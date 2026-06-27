"""
nexus_db_apex.py
Three-timescale SQLite memory for NEXUS Apex Learner.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMESCALES
  Fast   (every case)  — errors: δ, rationale, error_type
  Medium (every round) — contrastive_pairs: boundary lessons
  Slow   (every 3R)    — structural_knowledge: causal model

KEY DESIGN CHANGES from nexus_db_v2.py
  • No boolean feature columns — learning is embedding-based
  • Errors store full text (denormalized) for fast prompt injection
  • Contrastive pairs store anchor + contrast texts + embedding
  • Structural knowledge table captures slow-timescale synthesis
  • Near-misses are first-class citizens (own table)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

# ── Error type taxonomy (keyword-based, zero LLM cost) ────────────────────────

_ERROR_TAXONOMY: list[tuple[str, list[str]]] = [
    ("temporal_confusion",      ["following", "after", "temporal", "sequence",
                                  "prior to", "before", "subsequent", "then"]),
    ("therapeutic_goal",        ["therapeutic", "treatment goal", "intended",
                                  "beneficial", "improvement", "efficacy", "response"]),
    ("report_context",          ["report", "case report", "case series",
                                  "documented", "published", "literature", "study"]),
    ("negation_confusion",      ["no ", "not ", "without", "denied",
                                  "absence", "negative", "rule out", "unlikely"]),
    ("causal_ambiguity",        ["unclear", "ambiguous", "uncertain", "possible",
                                  "might", "may indicate", "cannot determine"]),
    ("completeness_confusion",  ["short", "incomplete", "fragment", "header",
                                  "title", "label", "coding"]),
]


def classify_error_type(rationale: str) -> str:
    """Zero-cost keyword classification of WHY the LLM erred."""
    r = rationale.lower()
    for etype, keywords in _ERROR_TAXONOMY:
        if any(k in r for k in keywords):
            return etype
    return "general"


def compute_delta(confidence: float, is_wrong: bool) -> float:
    """
    Dopamine-inspired prediction error signal.
    δ = confidence × |surprise|
    High confidence + wrong prediction → large δ (large learning signal).
    Low confidence + wrong prediction → small δ (small learning signal).
    """
    return round(float(confidence) * float(is_wrong), 4)


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=MEMORY;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT    NOT NULL,
    true_label   TEXT    NOT NULL,
    split        TEXT    NOT NULL DEFAULT 'train',
    loaded_round INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cases_split ON cases(split);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_text ON cases(text, split);

CREATE TABLE IF NOT EXISTS errors (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL REFERENCES cases(id),
    text         TEXT    NOT NULL,
    round        INTEGER NOT NULL,
    predicted    TEXT    NOT NULL,
    true_label   TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    delta        REAL    NOT NULL DEFAULT 0,
    error_type   TEXT    NOT NULL DEFAULT 'general',
    rationale    TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_errors_round  ON errors(round);
CREATE INDEX IF NOT EXISTS idx_errors_type   ON errors(error_type);
CREATE INDEX IF NOT EXISTS idx_errors_delta  ON errors(delta DESC);

CREATE TABLE IF NOT EXISTS near_misses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL REFERENCES cases(id),
    text         TEXT    NOT NULL,
    true_label   TEXT    NOT NULL,
    round        INTEGER NOT NULL,
    confidence   REAL    NOT NULL,
    boundary_delta REAL  NOT NULL  -- (1 - confidence): how close to boundary
);

CREATE INDEX IF NOT EXISTS idx_nm_round ON near_misses(round);
CREATE INDEX IF NOT EXISTS idx_nm_delta ON near_misses(boundary_delta DESC);

CREATE TABLE IF NOT EXISTS contrastive_pairs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_text      TEXT    NOT NULL,
    anchor_label     TEXT    NOT NULL,
    contrast_text    TEXT    NOT NULL,
    contrast_label   TEXT    NOT NULL,
    error_type       TEXT    NOT NULL DEFAULT 'general',
    lesson           TEXT    NOT NULL DEFAULT '',
    key_distinction  TEXT    NOT NULL DEFAULT '',
    delta_weight     REAL    NOT NULL DEFAULT 0,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    is_core          INTEGER NOT NULL DEFAULT 0,
    is_absorbed      INTEGER NOT NULL DEFAULT 0,
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_round    INTEGER NOT NULL,
    last_seen_round  INTEGER NOT NULL,
    anchor_embedding BLOB               -- float32 numpy array, L2-normalised
);

CREATE INDEX IF NOT EXISTS idx_pairs_type    ON contrastive_pairs(error_type);
CREATE INDEX IF NOT EXISTS idx_pairs_active  ON contrastive_pairs(is_active, is_absorbed);
CREATE INDEX IF NOT EXISTS idx_pairs_core    ON contrastive_pairs(is_core);

CREATE TABLE IF NOT EXISTS structural_knowledge (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_created   INTEGER NOT NULL,
    causal_model    TEXT    NOT NULL DEFAULT '',
    key_factors     TEXT    NOT NULL DEFAULT '[]',   -- JSON list
    error_patterns  TEXT    NOT NULL DEFAULT '[]',   -- JSON list
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS round_stats (
    round            INTEGER PRIMARY KEY,
    train_f1         REAL,
    train_precision  REAL,
    train_recall     REAL,
    train_errors     INTEGER,
    eval_f1          REAL,
    eval_precision   REAL,
    eval_recall      REAL,
    threshold        REAL,
    plasticity       REAL,
    n_pairs          INTEGER,
    n_core           INTEGER,
    theta_corrections INTEGER,
    near_misses      INTEGER
);
"""


# ── DDL split into individual statements (APFS mount compatibility) ───────────

_DDL_STATEMENTS = [s.strip() for s in """
CREATE TABLE IF NOT EXISTS cases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT    NOT NULL,
    true_label   TEXT    NOT NULL,
    split        TEXT    NOT NULL DEFAULT 'train',
    loaded_round INTEGER NOT NULL DEFAULT 0
)
---
CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_text ON cases(text, split)
---
CREATE INDEX IF NOT EXISTS idx_cases_split ON cases(split)
---
CREATE TABLE IF NOT EXISTS errors (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL,
    text         TEXT    NOT NULL,
    round        INTEGER NOT NULL,
    predicted    TEXT    NOT NULL,
    true_label   TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    delta        REAL    NOT NULL DEFAULT 0,
    error_type   TEXT    NOT NULL DEFAULT 'general',
    rationale    TEXT    NOT NULL DEFAULT ''
)
---
CREATE INDEX IF NOT EXISTS idx_errors_round  ON errors(round)
---
CREATE INDEX IF NOT EXISTS idx_errors_type   ON errors(error_type)
---
CREATE INDEX IF NOT EXISTS idx_errors_delta  ON errors(delta)
---
CREATE TABLE IF NOT EXISTS near_misses (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id        INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    true_label     TEXT    NOT NULL,
    round          INTEGER NOT NULL,
    confidence     REAL    NOT NULL,
    boundary_delta REAL    NOT NULL
)
---
CREATE INDEX IF NOT EXISTS idx_nm_round ON near_misses(round)
---
CREATE INDEX IF NOT EXISTS idx_nm_delta ON near_misses(boundary_delta)
---
CREATE TABLE IF NOT EXISTS contrastive_pairs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_text      TEXT    NOT NULL,
    anchor_label     TEXT    NOT NULL,
    contrast_text    TEXT    NOT NULL,
    contrast_label   TEXT    NOT NULL,
    error_type       TEXT    NOT NULL DEFAULT 'general',
    lesson           TEXT    NOT NULL DEFAULT '',
    key_distinction  TEXT    NOT NULL DEFAULT '',
    delta_weight     REAL    NOT NULL DEFAULT 0,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    is_core          INTEGER NOT NULL DEFAULT 0,
    is_absorbed      INTEGER NOT NULL DEFAULT 0,
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_round    INTEGER NOT NULL,
    last_seen_round  INTEGER NOT NULL,
    anchor_embedding BLOB
)
---
CREATE INDEX IF NOT EXISTS idx_pairs_type   ON contrastive_pairs(error_type)
---
CREATE INDEX IF NOT EXISTS idx_pairs_active ON contrastive_pairs(is_active, is_absorbed)
---
CREATE INDEX IF NOT EXISTS idx_pairs_core   ON contrastive_pairs(is_core)
---
CREATE TABLE IF NOT EXISTS structural_knowledge (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    round_created   INTEGER NOT NULL,
    causal_model    TEXT    NOT NULL DEFAULT '',
    key_factors     TEXT    NOT NULL DEFAULT '[]',
    error_patterns  TEXT    NOT NULL DEFAULT '[]',
    is_active       INTEGER NOT NULL DEFAULT 1
)
---
CREATE TABLE IF NOT EXISTS round_stats (
    round             INTEGER PRIMARY KEY,
    train_f1          REAL,
    train_precision   REAL,
    train_recall      REAL,
    train_errors      INTEGER,
    eval_f1           REAL,
    eval_precision    REAL,
    eval_recall       REAL,
    threshold         REAL,
    plasticity        REAL,
    n_pairs           INTEGER,
    n_core            INTEGER,
    theta_corrections INTEGER,
    near_misses       INTEGER
)
""".split("---") if s.strip()]


# ── Database class ─────────────────────────────────────────────────────────────

class NexusApexDB:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._init_db()

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self):
        """Create all tables using individual execute() calls (avoids APFS executescript issues)."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA foreign_keys=ON")
            for stmt in _DDL_STATEMENTS:
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()
        finally:
            conn.close()

    # ── Case loading ───────────────────────────────────────────────────────────

    def load_cases(
        self,
        cases: list[dict],
        split: str = "train",
        round_num: int = 0,
    ) -> int:
        """Insert cases, ignoring duplicates (by text+split). Returns inserted count."""
        rows = [
            (c["text"], c.get("true_label", c.get("label", "")), split, round_num)
            for c in cases
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO cases(text, true_label, split, loaded_round)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get_cases(self, split: str = "train") -> list[dict]:
        """Return all cases for a split as dicts."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, text, true_label FROM cases WHERE split=?", (split,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_cases(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT split, COUNT(*) n FROM cases GROUP BY split"
            ).fetchall()
        return {r["split"]: r["n"] for r in rows}

    def get_seen_case_texts(self) -> set[str]:
        """For warm restart: texts that have already been classified."""
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT text FROM errors").fetchall()
        return {r["text"] for r in rows}

    # ── Fast timescale: error logging ─────────────────────────────────────────

    def add_error(
        self,
        case_id: int,
        text: str,
        round_num: int,
        predicted: str,
        true_label: str,
        confidence: float,
        rationale: str,
    ) -> float:
        """
        Log a misclassification. Computes δ and error_type automatically.
        Returns δ for the caller's accumulation.
        """
        delta      = compute_delta(confidence, True)
        error_type = classify_error_type(rationale)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO errors(case_id, text, round, predicted, true_label,"
                " confidence, delta, error_type, rationale)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (case_id, text, round_num, predicted, true_label,
                 confidence, delta, error_type, rationale),
            )
        return delta

    def add_near_miss(
        self,
        case_id: int,
        text: str,
        true_label: str,
        round_num: int,
        confidence: float,
    ) -> None:
        """Log a correct but uncertain prediction (boundary case)."""
        boundary_delta = round(1.0 - float(confidence), 4)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO near_misses(case_id, text, true_label, round,"
                " confidence, boundary_delta) VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, text, true_label, round_num, confidence, boundary_delta),
            )

    def get_errors_this_round(self, round_num: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM errors WHERE round=? ORDER BY delta DESC",
                (round_num,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_near_misses(self, n: int = 20) -> list[dict]:
        """Return the n most boundary-proximate near-misses across all rounds."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT text, true_label, round, confidence FROM near_misses"
                " ORDER BY boundary_delta DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_errors_by_type(
        self,
        round_num: int,
        min_delta: float = 0.3,
    ) -> dict[str, list[dict]]:
        """Group this round's errors by error_type, filtered by δ."""
        errors = [
            e for e in self.get_errors_this_round(round_num)
            if e["delta"] >= min_delta
        ]
        grouped: dict[str, list[dict]] = {}
        for e in errors:
            grouped.setdefault(e["error_type"], []).append(e)
        return grouped

    # ── Medium timescale: contrastive pairs ───────────────────────────────────

    def upsert_pair(
        self,
        anchor_text: str,
        anchor_label: str,
        contrast_text: str,
        contrast_label: str,
        error_type: str,
        lesson: str,
        key_distinction: str,
        delta: float,
        anchor_embedding: np.ndarray,
        round_num: int,
    ) -> int:
        """
        Add a new pair or update an existing one (matched by anchor_text + error_type).
        Auto-promotes to CORE at occurrence_count >= 3.
        Returns pair id.
        """
        emb_blob = anchor_embedding.astype(np.float32).tobytes()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id, occurrence_count, delta_weight FROM contrastive_pairs"
                " WHERE anchor_text=? AND error_type=?",
                (anchor_text, error_type),
            ).fetchone()
            if existing:
                new_count  = existing["occurrence_count"] + 1
                new_weight = existing["delta_weight"] + delta
                is_core    = 1 if new_count >= 3 else 0
                conn.execute(
                    "UPDATE contrastive_pairs SET occurrence_count=?, delta_weight=?,"
                    " is_core=?, last_seen_round=?, lesson=?, key_distinction=?"
                    " WHERE id=?",
                    (new_count, new_weight, is_core, round_num,
                     lesson, key_distinction, existing["id"]),
                )
                return existing["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO contrastive_pairs(anchor_text, anchor_label,"
                    " contrast_text, contrast_label, error_type, lesson,"
                    " key_distinction, delta_weight, occurrence_count, is_core,"
                    " is_absorbed, is_active, created_round, last_seen_round,"
                    " anchor_embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 0, 1, ?, ?, ?)",
                    (anchor_text, anchor_label, contrast_text, contrast_label,
                     error_type, lesson, key_distinction, delta,
                     round_num, round_num, emb_blob),
                )
                return cur.lastrowid

    def get_active_pairs(self) -> list[dict]:
        """All active, non-absorbed pairs for lesson retrieval."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM contrastive_pairs WHERE is_active=1 AND is_absorbed=0"
                " ORDER BY delta_weight DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_core_pairs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM contrastive_pairs WHERE is_core=1 AND is_active=1"
                " ORDER BY delta_weight DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_weighted_lessons(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        core_only: bool = False,
    ) -> list[dict]:
        """
        Retrieve top-k contrastive pairs by cosine similarity to query embedding.
        Embedding-based retrieval — NOT feature-signature-based.
        """
        pairs = self.get_core_pairs() if core_only else self.get_active_pairs()
        if not pairs:
            return []
        valid = [p for p in pairs if p.get("anchor_embedding")]
        if not valid:
            return pairs[:k]

        embeddings = np.stack([
            np.frombuffer(p["anchor_embedding"], dtype=np.float32)
            for p in valid
        ])
        # Cosine similarity (vectors are L2-normalised from embedder.py)
        sims = embeddings @ query_embedding.astype(np.float32)
        top_idx = np.argsort(sims)[-k:][::-1]
        result = []
        for i in top_idx:
            p = dict(valid[i])
            p["similarity"] = float(sims[i])
            p.pop("anchor_embedding", None)  # don't pass bytes upstream
            result.append(p)
        return result

    def mark_pairs_absorbed(self, pair_ids: list[int]) -> None:
        """Graduate pairs to structural knowledge (slow timescale)."""
        if not pair_ids:
            return
        placeholders = ",".join("?" * len(pair_ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE contrastive_pairs SET is_absorbed=1, is_active=0"
                f" WHERE id IN ({placeholders})",
                pair_ids,
            )

    def get_pairs_created_before(self, round_num: int, min_occurrences: int = 2) -> list[dict]:
        """Pairs old enough to have their absorption measured."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM contrastive_pairs WHERE created_round <= ?"
                " AND occurrence_count >= ? AND is_absorbed=0 AND is_active=1",
                (round_num - 2, min_occurrences),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Slow timescale: structural knowledge ──────────────────────────────────

    def get_structural_knowledge(self) -> Optional[dict]:
        """Return the most recent active causal model, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM structural_knowledge WHERE is_active=1"
                " ORDER BY round_created DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["key_factors"]    = json.loads(d.get("key_factors", "[]"))
        d["error_patterns"] = json.loads(d.get("error_patterns", "[]"))
        return d

    def store_structural_knowledge(
        self,
        round_num: int,
        causal_model: str,
        key_factors: list[str],
        error_patterns: list[str],
    ) -> None:
        """Store a new causal model, deactivating the previous one."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE structural_knowledge SET is_active=0"
            )
            conn.execute(
                "INSERT INTO structural_knowledge(round_created, causal_model,"
                " key_factors, error_patterns, is_active)"
                " VALUES (?, ?, ?, ?, 1)",
                (round_num, causal_model,
                 json.dumps(key_factors), json.dumps(error_patterns)),
            )

    # ── Round statistics ──────────────────────────────────────────────────────

    def save_round_stats(
        self,
        round_num: int,
        train_f1: float,
        train_precision: float,
        train_recall: float,
        train_errors: int,
        eval_f1: float,
        eval_precision: float,
        eval_recall: float,
        threshold: float,
        plasticity: float,
        n_pairs: int,
        n_core: int,
        theta_corrections: int,
        near_misses: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO round_stats VALUES"
                " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (round_num, train_f1, train_precision, train_recall, train_errors,
                 eval_f1, eval_precision, eval_recall, threshold, plasticity,
                 n_pairs, n_core, theta_corrections, near_misses),
            )

    def get_round_history(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM round_stats ORDER BY round"
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> str:
        counts = self.count_cases()
        history = self.get_round_history()
        with self._conn() as conn:
            n_pairs = conn.execute(
                "SELECT COUNT(*) FROM contrastive_pairs WHERE is_active=1"
            ).fetchone()[0]
            n_core = conn.execute(
                "SELECT COUNT(*) FROM contrastive_pairs WHERE is_core=1 AND is_active=1"
            ).fetchone()[0]
            n_errors = conn.execute(
                "SELECT COUNT(*) FROM errors"
            ).fetchone()[0]
            n_nm = conn.execute(
                "SELECT COUNT(*) FROM near_misses"
            ).fetchone()[0]

        f1s = " → ".join(f"{r['eval_f1']:.4f}" for r in history[-4:])
        has_model = self.get_structural_knowledge() is not None
        return (
            f"NexusApexDB: {counts}\n"
            f"  Pairs: {n_pairs} active ({n_core} CORE) | "
            f"Errors: {n_errors} | Near-misses: {n_nm}\n"
            f"  Causal model: {'YES' if has_model else 'none'} | "
            f"F1 trajectory: {f1s}"
        )
