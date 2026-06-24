"""
nexus_db.py
NEXUS v3 — SQLite Persistence Layer

One database file per run: <out_dir>/nexus.db

Replaces scattered JSON files with a single queryable store.
Key benefits:
  1. MCQ deduplication — check DB BEFORE calling the LLM; if a similar
     teaching case already exists, skip generation entirely (saves 1 API call)
  2. LLM call ledger — track exactly how many calls each round costs and why
  3. Eval / threshold history — all metrics queryable without loading JSON files
  4. Engram + principle persistence — queryable across runs
  5. Route weight history — watch how routes evolve over training

FAISS index is kept separately — SQLite cannot do ANN vector search.
All other persistent state lives here.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np


# ─── Embedding helpers ────────────────────────────────────────────────────────

def _encode(arr) -> Optional[bytes]:
    if arr is None:
        return None
    return np.array(arr, dtype=np.float32).tobytes()


def _decode(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


# ─── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nodes (
    id               TEXT PRIMARY KEY,
    parent_id        TEXT,
    trigger_cond     TEXT,
    prompt           TEXT,
    created_round    INTEGER DEFAULT 0,
    retired          INTEGER DEFAULT 0
);

-- One row per MCQ teaching case generated from a misclassification.
-- embedding stored as float32 blob for similarity deduplication.
CREATE TABLE IF NOT EXISTS mcq_cases (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id          TEXT    NOT NULL,
    round_num        INTEGER NOT NULL,
    text             TEXT    NOT NULL,
    true_label       TEXT,
    predicted_label  TEXT,
    correct_reasoning TEXT,
    error_type       TEXT,
    difficulty       TEXT,
    embedding        BLOB,
    ts               REAL    DEFAULT (strftime('%s','now'))
);

-- Wrong-answer explanations for each MCQ case.
CREATE TABLE IF NOT EXISTS mcq_distractors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mcq_id      INTEGER NOT NULL REFERENCES mcq_cases(id) ON DELETE CASCADE,
    label       TEXT,
    reasoning   TEXT,
    correction  TEXT,
    error_type  TEXT
);

-- Engram clusters formed from error patterns.
CREATE TABLE IF NOT EXISTS engram_clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       TEXT    NOT NULL,
    cluster_id    TEXT    NOT NULL,
    size          INTEGER DEFAULT 0,
    centroid      BLOB,
    principle     TEXT,
    created_round INTEGER,
    swr_fired     INTEGER DEFAULT 0,
    UNIQUE(node_id, cluster_id)
);

-- Principles extracted from SWR events.
CREATE TABLE IF NOT EXISTS principles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id        TEXT    NOT NULL,
    cluster_id     TEXT,
    principle      TEXT    NOT NULL,
    injected_round INTEGER
);

-- Route weight snapshots (one row per route per round).
CREATE TABLE IF NOT EXISTS route_weights (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id    TEXT NOT NULL,
    round_num  INTEGER NOT NULL,
    route_name TEXT NOT NULL,
    weight     REAL,
    accuracy   REAL
);

-- Per-round evaluation metrics.
CREATE TABLE IF NOT EXISTS eval_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    round_num      INTEGER NOT NULL,
    f1             REAL, precision REAL, recall REAL,
    batch_accuracy REAL,
    errors         INTEGER, swr_events INTEGER, grafts INTEGER,
    tree_nodes     INTEGER, ade_bias REAL,
    tp INTEGER, fp INTEGER, fn INTEGER, tn INTEGER,
    graft_happened INTEGER DEFAULT 0,
    swr_happened   INTEGER DEFAULT 0,
    ts             REAL DEFAULT (strftime('%s','now'))
);

-- Full threshold calibration sweep stored every time calibration runs.
CREATE TABLE IF NOT EXISTS threshold_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    round_num      INTEGER NOT NULL,
    bias_candidate REAL    NOT NULL,
    f1 REAL, precision REAL, recall REAL, fbeta REAL,
    selected       INTEGER DEFAULT 0
);

-- Every LLM API call logged with type and context.
-- Lets you audit exactly where quota is being spent.
CREATE TABLE IF NOT EXISTS llm_calls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    round_num  INTEGER,
    call_type  TEXT,   -- 'route', 'mcq_gen', 'engram_consolidate',
                       -- 'meta', 'graft_probe', 'child_propose'
    node_id    TEXT,
    skipped    INTEGER DEFAULT 0,  -- 1 = call was avoided via deduplication
    ts         REAL DEFAULT (strftime('%s','now'))
);

-- Homeostatic controller intervention log.
CREATE TABLE IF NOT EXISTS intervention_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    round_num      INTEGER NOT NULL,
    intervention   TEXT    NOT NULL,   -- e.g. 'principle_rollback', 'node_retirement'
    committed      INTEGER DEFAULT 0,  -- 1 = change was applied, 0 = probe failed
    delta_f1       REAL    DEFAULT 0.0,
    detail         TEXT,               -- optional JSON detail (e.g. feature flag proposal)
    ts             REAL    DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS idx_mcq_node      ON mcq_cases(node_id);
CREATE INDEX IF NOT EXISTS idx_mcq_error     ON mcq_cases(error_type);
CREATE INDEX IF NOT EXISTS idx_eval_round    ON eval_history(round_num);
CREATE INDEX IF NOT EXISTS idx_llm_round     ON llm_calls(round_num);
CREATE INDEX IF NOT EXISTS idx_llm_type      ON llm_calls(call_type);
CREATE INDEX IF NOT EXISTS idx_intervention  ON intervention_history(round_num);
"""


# ─── Database ─────────────────────────────────────────────────────────────────

class NexusDB:
    """
    SQLite persistence layer for NEXUS v3.

    Usage:
        db = NexusDB("run_v3_03/nexus.db")
        db.log_eval(rnd, metrics, ade_bias, ...)
        db.log_llm_call("route", rnd, node_id)
        mcq_id = db.find_similar_mcq(embedding, node_id)  # 0 extra LLM calls
    """

    def __init__(self, db_path: str):
        self.path = str(db_path)
        self._init()

    def _init(self):
        with self._cx() as cx:
            cx.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self):
        """Safe schema migrations — ALTER TABLE only if column doesn't exist."""
        migrations = [
            ("eval_history",         "graft_happened", "INTEGER DEFAULT 0"),
            ("eval_history",         "swr_happened",   "INTEGER DEFAULT 0"),
        ]
        with self._cx() as cx:
            for table, col, typedef in migrations:
                try:
                    cx.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # Column already exists — safe to ignore

    @contextmanager
    def _cx(self):
        """Per-call connection (thread-safe, WAL allows concurrent readers)."""
        cx = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        cx.row_factory = sqlite3.Row
        try:
            yield cx
            cx.commit()
        except Exception:
            cx.rollback()
            raise
        finally:
            cx.close()

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def upsert_node(
        self,
        node_id: str,
        parent_id: Optional[str],
        trigger: Optional[str],
        prompt: str,
        created_round: int = 0,
    ):
        with self._cx() as cx:
            cx.execute("""
                INSERT OR REPLACE INTO nodes (id, parent_id, trigger_cond, prompt, created_round)
                VALUES (?,?,?,?,?)
            """, (node_id, parent_id, trigger, prompt, created_round))

    def retire_node(self, node_id: str):
        with self._cx() as cx:
            cx.execute("UPDATE nodes SET retired=1 WHERE id=?", (node_id,))

    # ── MCQ deduplication ─────────────────────────────────────────────────────

    def find_similar_mcq(
        self,
        embedding: list[float],
        node_id: str,
        min_sim: float = 0.85,
    ) -> Optional[int]:
        """
        Before calling the LLM to generate a new MCQ, call this.

        Returns the existing mcq_id if a similar teaching case already exists
        (cosine similarity >= min_sim), otherwise returns None.

        If it returns a value → skip the LLM call, log it as skipped.
        If it returns None    → generate the MCQ, then call insert_mcq().

        Zero LLM calls. Pure Python + SQLite read.
        """
        q_vec = np.array(embedding, dtype=np.float32)
        with self._cx() as cx:
            rows = cx.execute(
                "SELECT id, embedding FROM mcq_cases WHERE node_id=? AND embedding IS NOT NULL",
                (node_id,),
            ).fetchall()

        best_id, best_sim = None, -1.0
        for row in rows:
            v = _decode(row["embedding"])
            if v is not None and len(v) == len(q_vec):
                sim = _cosine(q_vec, v)
                if sim > best_sim:
                    best_sim, best_id = sim, row["id"]

        return best_id if best_sim >= min_sim else None

    def insert_mcq(
        self,
        node_id: str,
        round_num: int,
        text: str,
        true_label: str,
        predicted_label: str,
        correct_reasoning: str,
        error_type: str,
        difficulty: str,
        embedding: Optional[list[float]],
        distractors: list[dict],
    ) -> int:
        """Store a newly generated MCQ. Returns the new mcq_id."""
        blob = _encode(embedding)
        with self._cx() as cx:
            cur = cx.execute("""
                INSERT INTO mcq_cases
                  (node_id, round_num, text, true_label, predicted_label,
                   correct_reasoning, error_type, difficulty, embedding)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (node_id, round_num, text, true_label, predicted_label,
                  correct_reasoning, error_type, difficulty, blob))
            mcq_id = cur.lastrowid
            for d in distractors:
                cx.execute("""
                    INSERT INTO mcq_distractors (mcq_id, label, reasoning, correction, error_type)
                    VALUES (?,?,?,?,?)
                """, (mcq_id,
                      d.get("label", ""), d.get("reasoning", ""),
                      d.get("correction", ""), d.get("error_type", "")))
        return mcq_id

    # ── LLM call ledger ───────────────────────────────────────────────────────

    def log_llm_call(
        self,
        call_type: str,
        round_num: int = 0,
        node_id: str = "",
        skipped: bool = False,
    ):
        """
        Record every LLM API call (or skipped call).
        call_type: 'route', 'mcq_gen', 'engram_consolidate',
                   'meta', 'graft_probe', 'child_propose'
        skipped=True: the call was avoided via deduplication.
        """
        with self._cx() as cx:
            cx.execute("""
                INSERT INTO llm_calls (round_num, call_type, node_id, skipped)
                VALUES (?,?,?,?)
            """, (round_num, call_type, node_id, int(skipped)))

    def llm_summary(self, round_num: Optional[int] = None) -> dict:
        """Returns {call_type: count} for actual + skipped calls."""
        with self._cx() as cx:
            if round_num is not None:
                rows = cx.execute("""
                    SELECT call_type, skipped, COUNT(*) as cnt
                    FROM llm_calls WHERE round_num=?
                    GROUP BY call_type, skipped
                """, (round_num,)).fetchall()
            else:
                rows = cx.execute("""
                    SELECT call_type, skipped, COUNT(*) as cnt
                    FROM llm_calls GROUP BY call_type, skipped
                """).fetchall()
        result: dict = {}
        for r in rows:
            k = r["call_type"] + ("_saved" if r["skipped"] else "")
            result[k] = r["cnt"]
        return result

    def total_calls(self) -> int:
        with self._cx() as cx:
            return cx.execute(
                "SELECT COUNT(*) FROM llm_calls WHERE skipped=0"
            ).fetchone()[0]

    def total_saved(self) -> int:
        with self._cx() as cx:
            return cx.execute(
                "SELECT COUNT(*) FROM llm_calls WHERE skipped=1"
            ).fetchone()[0]

    # ── Eval history ──────────────────────────────────────────────────────────

    def log_eval(
        self,
        round_num: int,
        metrics: dict,
        ade_bias: float,
        batch_accuracy: float = 0.0,
        errors: int = 0,
        swr_events: int = 0,
        grafts: int = 0,
        tree_nodes: int = 0,
        graft_happened: bool = False,
        swr_happened: bool = False,
    ):
        with self._cx() as cx:
            cx.execute("""
                INSERT INTO eval_history
                  (round_num, f1, precision, recall, batch_accuracy,
                   errors, swr_events, grafts, tree_nodes, ade_bias,
                   tp, fp, fn, tn, graft_happened, swr_happened)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                round_num,
                metrics.get("f1"), metrics.get("precision"), metrics.get("recall"),
                batch_accuracy, errors, swr_events, grafts, tree_nodes, ade_bias,
                metrics.get("tp", 0), metrics.get("fp", 0),
                metrics.get("fn", 0), metrics.get("tn", 0),
                int(graft_happened), int(swr_happened),
            ))

    def save_baseline(self, metrics: dict):
        """Cache baseline eval so restarts don't re-run 200+ LLM calls."""
        with self._cx() as cx:
            cx.execute("""
                INSERT OR REPLACE INTO eval_history
                  (round_num, f1, precision, recall, tp, fp, fn, tn, ade_bias)
                VALUES (0, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (metrics.get("f1"), metrics.get("precision"), metrics.get("recall"),
                  metrics.get("tp", 0), metrics.get("fp", 0),
                  metrics.get("fn", 0), metrics.get("tn", 0)))

    def get_baseline(self) -> Optional[dict]:
        """Return cached baseline, or None if not yet computed."""
        with self._cx() as cx:
            row = cx.execute(
                "SELECT * FROM eval_history WHERE round_num=0 LIMIT 1"
            ).fetchone()
        if row:
            return {"f1": row["f1"], "precision": row["precision"],
                    "recall": row["recall"], "tp": row["tp"], "fp": row["fp"],
                    "fn": row["fn"], "tn": row["tn"]}
        return None

    def get_f1_curve(self) -> list[tuple[int, float]]:
        with self._cx() as cx:
            rows = cx.execute(
                "SELECT round_num, f1 FROM eval_history ORDER BY round_num"
            ).fetchall()
        return [(r["round_num"], r["f1"]) for r in rows]

    # ── Threshold calibration ─────────────────────────────────────────────────

    def log_threshold_calibration(
        self,
        round_num: int,
        results: list[tuple],   # (bias, score, prec, rec, f1, fbeta)
        selected_bias: float,
    ):
        with self._cx() as cx:
            for bias, score, prec, rec, f1, fbeta in results:
                cx.execute("""
                    INSERT INTO threshold_history
                      (round_num, bias_candidate, f1, precision, recall, fbeta, selected)
                    VALUES (?,?,?,?,?,?,?)
                """, (round_num, bias, f1, prec, rec, fbeta,
                      1 if abs(bias - selected_bias) < 1e-6 else 0))

    # ── Engrams + principles ──────────────────────────────────────────────────

    def upsert_engram_cluster(
        self,
        node_id: str,
        cluster_id: str,
        size: int,
        centroid: Optional[list[float]],
        principle: Optional[str],
        created_round: int,
        swr_fired: bool = False,
    ):
        with self._cx() as cx:
            cx.execute("""
                INSERT INTO engram_clusters
                  (node_id, cluster_id, size, centroid, principle, created_round, swr_fired)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(node_id, cluster_id) DO UPDATE SET
                  size=excluded.size,
                  principle=excluded.principle,
                  swr_fired=excluded.swr_fired
            """, (node_id, cluster_id, size, _encode(centroid),
                  principle, created_round, int(swr_fired)))

    def insert_principle(
        self, node_id: str, cluster_id: str, principle: str, injected_round: int
    ):
        with self._cx() as cx:
            cx.execute("""
                INSERT INTO principles (node_id, cluster_id, principle, injected_round)
                VALUES (?,?,?,?)
            """, (node_id, cluster_id, principle, injected_round))

    # ── Route weights ─────────────────────────────────────────────────────────

    def log_route_weights(
        self,
        node_id: str,
        round_num: int,
        weights: dict,
        histories: dict,
    ):
        with self._cx() as cx:
            for route, w in weights.items():
                hist = histories.get(route, [])
                acc = sum(hist) / len(hist) if hist else 0.0
                cx.execute("""
                    INSERT INTO route_weights (node_id, round_num, route_name, weight, accuracy)
                    VALUES (?,?,?,?,?)
                """, (node_id, round_num, route, w, acc))

    # ── Analytics ─────────────────────────────────────────────────────────────

    def mcq_error_summary(self) -> dict:
        with self._cx() as cx:
            rows = cx.execute("""
                SELECT error_type, COUNT(*) as cnt FROM mcq_cases
                GROUP BY error_type ORDER BY cnt DESC
            """).fetchall()
        return {r["error_type"]: r["cnt"] for r in rows}

    # ── Intervention history ──────────────────────────────────────────────────

    def log_intervention(
        self,
        round_num:    int,
        intervention: str,
        committed:    bool,
        delta_f1:     float = 0.0,
        detail:       Optional[str] = None,
    ) -> None:
        with self._cx() as cx:
            cx.execute(
                """INSERT INTO intervention_history
                   (round_num, intervention, committed, delta_f1, detail)
                   VALUES (?,?,?,?,?)""",
                (round_num, intervention, int(committed), delta_f1, detail),
            )

    def get_intervention_history(self) -> list[dict]:
        with self._cx() as cx:
            rows = cx.execute(
                "SELECT * FROM intervention_history ORDER BY round_num"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_eval_history_for_monitor(self) -> list[dict]:
        """Returns eval history in the format expected by HealthMonitor."""
        with self._cx() as cx:
            rows = cx.execute(
                """SELECT round_num, f1, precision, recall,
                          COALESCE(graft_happened, 0) as graft_happened,
                          COALESCE(swr_happened,   0) as swr_happened
                   FROM eval_history ORDER BY round_num"""
            ).fetchall()
        return [dict(r) for r in rows]

    def print_summary(self, round_num: Optional[int] = None):
        import os
        size_kb = os.path.getsize(self.path) / 1024
        print(f"\n[DB] nexus.db  ({size_kb:.0f} KB)")
        with self._cx() as cx:
            print(f"  MCQ cases:    {cx.execute('SELECT COUNT(*) FROM mcq_cases').fetchone()[0]}")
            print(f"  Clusters:     {cx.execute('SELECT COUNT(*) FROM engram_clusters').fetchone()[0]}")
            print(f"  Principles:   {cx.execute('SELECT COUNT(*) FROM principles').fetchone()[0]}")
            print(f"  Eval rounds:  {cx.execute('SELECT COUNT(*) FROM eval_history').fetchone()[0]}")
        actual = self.total_calls()
        saved  = self.total_saved()
        total  = actual + saved
        pct    = (saved / total * 100) if total > 0 else 0
        print(f"  LLM calls:    {actual} actual  |  {saved} saved by dedup  ({pct:.0f}% reduction)")
        err = self.mcq_error_summary()
        if err:
            top = list(err.items())[:4]
            print(f"  Top errors:   {top}")
        if round_num is not None:
            rnd_summary = self.llm_summary(round_num)
            print(f"  Round {round_num} calls: {rnd_summary}")
