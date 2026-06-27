"""
nexus_db_v2.py
SQLite-backed multidimensional case memory for NEXUS MCQ Learner.

═══════════════════════════════════════════════════════════════
 INTERNAL TOKENS (zero LLM cost)
 ─────────────────────────────────
 Every case is fingerprinted by two representations computed locally:

 1. Feature vector — 11 boolean values from features.py:
       has_induced, has_associated, has_toxicity, has_adverse,
       has_developed, has_following, has_reaction, has_report,
       has_negation, has_short, has_drug_name
    Stored as integer columns. Enables pure-SQL pattern detection.
    Feature signature = comma-joined names of TRUE features only.
    e.g.  "has_induced,has_negation"

 2. PubMedBERT embedding — 768-dim float32 vector (BLOB).
    Used for semantic similarity search (FAISS, in RAGIndex).
    Not stored in SQLite by default (too large); stored in FAISS index.

 LLM is called ONLY for:
   (a) Case classification (1 call per case)
   (b) MCQ generation for qualifying patterns (1 call per new pattern)
   (c) Optional: pattern description generation (1 call per new pattern)

═══════════════════════════════════════════════════════════════
 TABLES
 ───────
 cases         — every corpus case with feature vector + split label
 errors        — every misclassification, linked to case + pattern
 patterns      — recurring error signatures (feature-based)
 mcqs          — one lesson per pattern, with effectiveness tracking
 round_stats   — per-round evaluation metrics
 column_stats  — per-column per-round routing + BCM data
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from features import FEATURE_NAMES

# ── Feature signature helpers ─────────────────────────────────────────────────

def feature_vector(feat_dict: dict) -> dict:
    """Return {feature_name: 0/1} for all known features."""
    return {f: int(bool(feat_dict.get(f, False))) for f in FEATURE_NAMES}

def feature_signature(feat_dict: dict) -> str:
    """Compact string of only the TRUE features. Zero-cost internal token."""
    return ",".join(f for f in FEATURE_NAMES if feat_dict.get(f, False))

def signature_from_row(row: dict) -> str:
    return ",".join(f for f in FEATURE_NAMES if row.get(f, 0))


# ── Schema ────────────────────────────────────────────────────────────────────

_FEATURE_COLS_DDL = "\n    ".join(f"{f} INTEGER NOT NULL DEFAULT 0," for f in FEATURE_NAMES)

SCHEMA = f"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    true_label  TEXT NOT NULL,
    split       TEXT NOT NULL DEFAULT 'train',  -- train | eval | held_out
    {_FEATURE_COLS_DDL}
    feature_sig TEXT NOT NULL DEFAULT '',
    loaded_round INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cases_split ON cases(split);
CREATE INDEX IF NOT EXISTS idx_cases_sig   ON cases(feature_sig);

CREATE TABLE IF NOT EXISTS errors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id       INTEGER NOT NULL REFERENCES cases(id),
    round         INTEGER NOT NULL,
    column_id     TEXT    NOT NULL DEFAULT 'ROOT',
    predicted     TEXT    NOT NULL,
    confidence    REAL    NOT NULL DEFAULT 0.5,
    feature_sig   TEXT    NOT NULL DEFAULT '',
    error_type    TEXT    NOT NULL DEFAULT '',   -- FP | FN
    pattern_id    INTEGER REFERENCES patterns(id)
);

CREATE INDEX IF NOT EXISTS idx_errors_round   ON errors(round);
CREATE INDEX IF NOT EXISTS idx_errors_sig     ON errors(feature_sig);
CREATE INDEX IF NOT EXISTS idx_errors_pattern ON errors(pattern_id);

CREATE TABLE IF NOT EXISTS patterns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_sig      TEXT NOT NULL UNIQUE,
    description      TEXT NOT NULL DEFAULT '',
    error_type       TEXT NOT NULL DEFAULT 'MIXED',   -- FP | FN | MIXED
    first_round      INTEGER NOT NULL,
    last_round       INTEGER NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    total_errors     INTEGER NOT NULL DEFAULT 0,
    is_core          INTEGER NOT NULL DEFAULT 0,   -- 1 when seen 3+ rounds
    is_active        INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_patterns_sig  ON patterns(feature_sig);
CREATE INDEX IF NOT EXISTS idx_patterns_core ON patterns(is_core);

CREATE TABLE IF NOT EXISTS mcqs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id       INTEGER NOT NULL REFERENCES patterns(id),
    correct_answer   TEXT NOT NULL,
    correct_rationale TEXT NOT NULL,
    wrong_answers    TEXT NOT NULL DEFAULT '[]',  -- JSON list of {{answer, explanation}}
    example_text     TEXT NOT NULL DEFAULT '',
    round_created    INTEGER NOT NULL,
    round_revised    INTEGER,
    pre_error_rate   REAL,    -- error rate for this pattern BEFORE this MCQ
    post_error_rate  REAL,    -- error rate for this pattern AFTER this MCQ
    effectiveness    REAL,    -- pre - post; positive = helped
    is_active        INTEGER NOT NULL DEFAULT 1,
    version          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS round_stats (
    round           INTEGER PRIMARY KEY,
    phase           TEXT NOT NULL DEFAULT '',
    f1              REAL, precision_ REAL, recall REAL,
    tp INTEGER, fp INTEGER, fn INTEGER, tn INTEGER,
    n_cases_trained INTEGER NOT NULL DEFAULT 0,
    n_errors        INTEGER NOT NULL DEFAULT 0,
    n_new_patterns  INTEGER NOT NULL DEFAULT 0,
    n_core_patterns INTEGER NOT NULL DEFAULT 0,
    n_active_mcqs   INTEGER NOT NULL DEFAULT 0,
    firing_threshold REAL,
    notes           TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS column_stats (
    round     INTEGER NOT NULL,
    column_id TEXT    NOT NULL,
    routes    INTEGER NOT NULL DEFAULT 0,
    errors    INTEGER NOT NULL DEFAULT 0,
    bcm_event TEXT,
    theta_m   REAL,
    PRIMARY KEY (round, column_id)
);
"""


# ── Database class ────────────────────────────────────────────────────────────

class NexusDB:
    """
    Thread-safe SQLite database for NEXUS MCQ Learner.
    All pattern detection is pure SQL — zero LLM cost.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Initialize schema
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def transaction(self):
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── Case loading ──────────────────────────────────────────────────────────

    def load_cases(self, cases: list[dict], split: str = "train",
                   round_num: int = 0) -> int:
        """Bulk-insert cases into the cases table. Returns count inserted."""
        feat_cols = ", ".join(FEATURE_NAMES)
        placeholders = ", ".join("?" * (5 + len(FEATURE_NAMES)))
        sql = (
            f"INSERT OR IGNORE INTO cases "
            f"(text, true_label, split, {feat_cols}, feature_sig, loaded_round) "
            f"VALUES ({placeholders})"
        )
        rows = []
        for c in cases:
            from features import features as extract_features
            feats = extract_features(c["text"])
            fv = feature_vector(feats)
            sig = feature_signature(feats)
            rows.append((c["text"], c["label"], split,
                         *[fv[f] for f in FEATURE_NAMES], sig, round_num))
        with self.transaction() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def get_unseen_train_cases(self, n: int, seen_ids: set[int]) -> list[dict]:
        """Return up to n train cases not in seen_ids."""
        conn = self._conn()
        if seen_ids:
            ph = ",".join("?" * len(seen_ids))
            rows = conn.execute(
                f"SELECT * FROM cases WHERE split='train' AND id NOT IN ({ph}) "
                f"ORDER BY RANDOM() LIMIT ?",
                (*seen_ids, n)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cases WHERE split='train' ORDER BY RANDOM() LIMIT ?",
                (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_split_cases(self, split: str) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM cases WHERE split=?", (split,)
        ).fetchall()
        return [dict(r) for r in rows]

    def count_cases(self) -> dict:
        conn = self._conn()
        rows = conn.execute(
            "SELECT split, COUNT(*) as n FROM cases GROUP BY split"
        ).fetchall()
        return {r["split"]: r["n"] for r in rows}

    # ── Error recording ───────────────────────────────────────────────────────

    def record_errors(self, errors: list[dict], round_num: int) -> None:
        """
        Write misclassifications to errors table.
        Automatically assigns pattern_id based on feature_sig.
        """
        with self.transaction() as conn:
            for e in errors:
                error_type = "FP" if e["predicted"] == "ADE" else "FN"
                sig = e.get("feature_sig", "")
                # Look up existing pattern
                pat = conn.execute(
                    "SELECT id FROM patterns WHERE feature_sig=?", (sig,)
                ).fetchone()
                pattern_id = pat["id"] if pat else None

                conn.execute(
                    "INSERT INTO errors "
                    "(case_id, round, column_id, predicted, confidence, "
                    "feature_sig, error_type, pattern_id) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (e.get("case_id"), round_num, e.get("column_id", "ROOT"),
                     e["predicted"], e.get("confidence", 0.5),
                     sig, error_type, pattern_id)
                )

    # ── Pattern detection (pure SQL — zero LLM cost) ──────────────────────────

    def detect_patterns(self, round_num: int,
                        min_errors: int = 3) -> list[dict]:
        """
        Find error signatures with enough errors to qualify as a pattern.
        Pure SQL — zero LLM tokens.

        Returns list of dicts: {feature_sig, error_type, n_errors, is_new}
        """
        conn = self._conn()
        # Group errors from this round by feature signature
        rows = conn.execute("""
            SELECT
                feature_sig,
                COUNT(*) as n_errors,
                SUM(CASE WHEN error_type='FP' THEN 1 ELSE 0 END) as n_fp,
                SUM(CASE WHEN error_type='FN' THEN 1 ELSE 0 END) as n_fn
            FROM errors
            WHERE round = ? AND feature_sig != ''
            GROUP BY feature_sig
            HAVING COUNT(*) >= ?
            ORDER BY COUNT(*) DESC
        """, (round_num, min_errors)).fetchall()

        results = []
        for r in rows:
            sig = r["feature_sig"]
            n_fp, n_fn = r["n_fp"], r["n_fn"]
            error_type = "FP" if n_fp > n_fn else ("FN" if n_fn > n_fp else "MIXED")
            # Check if pattern already exists
            existing = conn.execute(
                "SELECT id, occurrence_count, last_round FROM patterns "
                "WHERE feature_sig=?", (sig,)
            ).fetchone()
            is_new = existing is None
            results.append({
                "feature_sig": sig,
                "error_type": error_type,
                "n_errors": r["n_errors"],
                "n_fp": n_fp,
                "n_fn": n_fn,
                "is_new": is_new,
                "pattern_id": existing["id"] if existing else None,
                "occurrence_count": (existing["occurrence_count"] + 1) if existing else 1,
            })
        return results

    def upsert_pattern(self, sig: str, error_type: str,
                       n_errors: int, round_num: int,
                       description: str = "") -> int:
        """Insert or update a pattern. Returns pattern_id."""
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id, occurrence_count, total_errors FROM patterns "
                "WHERE feature_sig=?", (sig,)
            ).fetchone()
            if existing:
                new_count = existing["occurrence_count"] + 1
                is_core = 1 if new_count >= 3 else 0
                conn.execute("""
                    UPDATE patterns SET
                        occurrence_count = ?,
                        total_errors = total_errors + ?,
                        last_round = ?,
                        is_core = ?,
                        error_type = ?
                    WHERE feature_sig = ?
                """, (new_count, n_errors, round_num, is_core, error_type, sig))
                return existing["id"]
            else:
                cur = conn.execute("""
                    INSERT INTO patterns
                    (feature_sig, description, error_type,
                     first_round, last_round, occurrence_count,
                     total_errors, is_core)
                    VALUES (?,?,?,?,?,1,?,0)
                """, (sig, description, error_type, round_num, round_num, n_errors))
                return cur.lastrowid

    def get_recurring_patterns(self, min_rounds: int = 2) -> list[dict]:
        """Patterns seen in 2+ distinct rounds — candidates for MCQ generation."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT p.*, m.id as mcq_id, m.is_active as mcq_active
            FROM patterns p
            LEFT JOIN mcqs m ON m.pattern_id = p.id AND m.is_active = 1
            WHERE p.occurrence_count >= ? AND p.is_active = 1
            ORDER BY p.occurrence_count DESC, p.total_errors DESC
        """, (min_rounds,)).fetchall()
        return [dict(r) for r in rows]

    def get_unaddressed_patterns(self, round_num: int) -> list[dict]:
        """Patterns with errors this round that have no active MCQ."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT DISTINCT p.*
            FROM patterns p
            LEFT JOIN mcqs m ON m.pattern_id = p.id AND m.is_active = 1
            WHERE p.is_active = 1
              AND m.id IS NULL
              AND p.occurrence_count >= 1
            ORDER BY p.occurrence_count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def update_pattern_error_links(self, round_num: int) -> None:
        """Link errors from this round to their pattern_id."""
        conn = self._conn()
        with self.transaction() as conn:
            conn.execute("""
                UPDATE errors SET pattern_id = (
                    SELECT p.id FROM patterns p
                    WHERE p.feature_sig = errors.feature_sig
                )
                WHERE round = ? AND pattern_id IS NULL
            """, (round_num,))

    # ── MCQ management ────────────────────────────────────────────────────────

    def add_mcq(self, pattern_id: int, correct_answer: str,
                correct_rationale: str, wrong_answers: list[dict],
                example_text: str, round_num: int,
                pre_error_rate: Optional[float] = None) -> int:
        """Insert a new MCQ. Returns mcq_id."""
        with self.transaction() as conn:
            # Deactivate old MCQ for this pattern if exists
            conn.execute(
                "UPDATE mcqs SET is_active=0 WHERE pattern_id=?", (pattern_id,)
            )
            cur = conn.execute("""
                INSERT INTO mcqs
                (pattern_id, correct_answer, correct_rationale,
                 wrong_answers, example_text, round_created,
                 pre_error_rate, is_active, version)
                VALUES (?,?,?,?,?,?,?,1,
                    COALESCE((SELECT MAX(version)+1 FROM mcqs WHERE pattern_id=?), 1)
                )
            """, (pattern_id, correct_answer, correct_rationale,
                  json.dumps(wrong_answers), example_text, round_num,
                  pre_error_rate, pattern_id))
            return cur.lastrowid

    def get_core_mcqs(self) -> list[dict]:
        """All active MCQs for CORE patterns (seen 3+ rounds). Always in prompt."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT m.*, p.feature_sig, p.error_type, p.occurrence_count,
                   p.description as pattern_description
            FROM mcqs m JOIN patterns p ON m.pattern_id = p.id
            WHERE m.is_active = 1 AND p.is_core = 1 AND p.is_active = 1
            ORDER BY p.occurrence_count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_contextual_mcqs(self, feature_sig: str,
                            max_mcqs: int = 4) -> list[dict]:
        """
        MCQs for patterns that are a SUBSET of the case's feature signature.
        e.g. if case has "has_induced,has_negation,has_short",
        patterns "has_induced,has_negation" and "has_negation" both match.
        """
        conn = self._conn()
        all_active = conn.execute("""
            SELECT m.*, p.feature_sig, p.error_type, p.occurrence_count
            FROM mcqs m JOIN patterns p ON m.pattern_id = p.id
            WHERE m.is_active = 1 AND p.is_active = 1
            ORDER BY p.occurrence_count DESC
        """).fetchall()

        case_features = set(feature_sig.split(",")) if feature_sig else set()
        matched = []
        for row in all_active:
            pat_feats = set(row["feature_sig"].split(",")) if row["feature_sig"] else set()
            if pat_feats and pat_feats.issubset(case_features):
                matched.append(dict(row))
            if len(matched) >= max_mcqs:
                break
        return matched

    def update_mcq_effectiveness(self, pattern_id: int,
                                 post_error_rate: float) -> None:
        """Record post-MCQ error rate and compute effectiveness."""
        with self.transaction() as conn:
            conn.execute("""
                UPDATE mcqs SET
                    post_error_rate = ?,
                    effectiveness = COALESCE(pre_error_rate, 0) - ?
                WHERE pattern_id = ? AND is_active = 1
            """, (post_error_rate, post_error_rate, pattern_id))

    def retire_ineffective_mcqs(self, min_effectiveness: float = -0.05) -> int:
        """Deactivate MCQs that made things worse (negative effectiveness)."""
        with self.transaction() as conn:
            cur = conn.execute("""
                UPDATE mcqs SET is_active = 0
                WHERE post_error_rate IS NOT NULL
                  AND effectiveness < ?
            """, (min_effectiveness,))
            return cur.rowcount

    # ── Stats ─────────────────────────────────────────────────────────────────

    def save_round_stats(self, round_num: int, stats: dict) -> None:
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO round_stats
                (round, phase, f1, precision_, recall,
                 tp, fp, fn, tn,
                 n_cases_trained, n_errors, n_new_patterns,
                 n_core_patterns, n_active_mcqs,
                 firing_threshold, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                round_num,
                stats.get("phase", ""),
                stats.get("f1"), stats.get("precision"), stats.get("recall"),
                stats.get("tp"), stats.get("fp"),
                stats.get("fn"), stats.get("tn"),
                stats.get("n_cases_trained", 0),
                stats.get("n_errors", 0),
                stats.get("n_new_patterns", 0),
                stats.get("n_core_patterns", 0),
                stats.get("n_active_mcqs", 0),
                stats.get("firing_threshold"),
                stats.get("notes", ""),
            ))

    def save_column_stats(self, round_num: int, col_stats: list[dict]) -> None:
        with self.transaction() as conn:
            for cs in col_stats:
                conn.execute("""
                    INSERT OR REPLACE INTO column_stats
                    (round, column_id, routes, errors, bcm_event, theta_m)
                    VALUES (?,?,?,?,?,?)
                """, (round_num, cs["column_id"], cs["routes"],
                      cs["errors"], cs.get("bcm_event"), cs.get("theta_m")))

    def get_round_history(self) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM round_stats ORDER BY round"
        ).fetchall()
        return [dict(r) for r in rows]

    def pattern_error_rate(self, pattern_id: int,
                           from_round: int, to_round: int) -> float:
        """Error rate for a pattern over a round range."""
        conn = self._conn()
        row = conn.execute("""
            SELECT COUNT(*) as n_errors FROM errors
            WHERE pattern_id = ? AND round BETWEEN ? AND ?
        """, (pattern_id, from_round, to_round)).fetchone()
        n_errors = row["n_errors"] if row else 0
        # Compare against total cases seen with this sig in that range
        pat = conn.execute(
            "SELECT feature_sig FROM patterns WHERE id=?", (pattern_id,)
        ).fetchone()
        if not pat:
            return 0.0
        sig = pat["feature_sig"]
        # Count cases with matching features in that range
        # (approximate: count by errors seen across all rounds)
        total = conn.execute("""
            SELECT COUNT(DISTINCT case_id) as n FROM errors
            WHERE feature_sig = ? AND round BETWEEN ? AND ?
        """, (sig, from_round, to_round)).fetchone()
        n_total = total["n"] if total and total["n"] else max(1, n_errors)
        return n_errors / n_total

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        conn = self._conn()
        counts = self.count_cases()
        n_patterns = conn.execute(
            "SELECT COUNT(*) FROM patterns WHERE is_active=1"
        ).fetchone()[0]
        n_core = conn.execute(
            "SELECT COUNT(*) FROM patterns WHERE is_core=1 AND is_active=1"
        ).fetchone()[0]
        n_mcqs = conn.execute(
            "SELECT COUNT(*) FROM mcqs WHERE is_active=1"
        ).fetchone()[0]
        n_errors = conn.execute(
            "SELECT COUNT(*) FROM errors"
        ).fetchone()[0]
        history = self.get_round_history()
        f1_str = " → ".join(f"{r['f1']:.4f}" for r in history if r.get("f1"))
        return (
            f"NexusDB: {counts}\n"
            f"  Patterns: {n_patterns} active ({n_core} CORE) | MCQs: {n_mcqs} | Errors: {n_errors}\n"
            f"  F1 trajectory: {f1_str}"
        )
