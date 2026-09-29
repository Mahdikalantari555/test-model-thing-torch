
import os
import re
import shutil
import sqlite3
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ponytail: SQLite WAL + FTS5 (BM25 lexical) + pre-normalized dense MVM fused via
# Reciprocal Rank Fusion, replacing O(N) JSON linear scan. stdlib sqlite3 + torch only.
# Ceiling: single-process; upgrade to HNSW/FAISS only if 100k+ facts miss 5ms on GPU.
# Extended with access_count, last_retrieved, version_id, feedback, needs_review,
# semantic_facts, feedback_log for lifelong learning.


def _tokenize(terms: List[str]) -> str:
    """Build a quoted OR'd FTS5 MATCH expression from query terms.

    English function words are dropped: they match nearly every fact and
    would otherwise swamp the BM25 lexical signal.
    """
    stop = {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
        "by", "for", "with", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "have", "has", "had", "it", "its",
        "this", "that", "these", "those", "which", "who", "whom", "whose",
        "what", "when", "where", "why", "how", "can", "could", "should",
        "would", "will", "shall", "may", "about", "into", "over", "under",
        "between", "during", "through", "out", "up", "down", "you", "your",
    }
    clean = [
        t for t in (re.sub(r"[^a-z0-9\u00C0-\uFFFF]", "", w.lower()) for w in terms)
        if t and t not in stop
    ]
    if not clean:
        return ""
    return " OR ".join(f'"{t}"' for t in clean[:32])


class KnowledgeStore:
    """Persistent hybrid knowledge store: FTS5 BM25 + dense MVM, fused with RRF.

    Facts, embeddings and TMS metadata live in one SQLite DB (WAL mode).
    All dense vectors are L2-normalized at insert so recall is a single
    matrix-vector multiply with no per-query re-normalization allocation.
    Extended with lifelong learning columns and semantic/feedback tables.
    """

    _RRF_K = 60.0
    _CANDS = 200  # per-signal candidate pool before RRF fusion

    def __init__(self, db_path: str = ":memory:", dim: int = 384):
        self.db_path = db_path
        self.dim = dim
        file_based = db_path != ":memory:"
        if file_based:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        if file_based:
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self._migrate_schema()
        # Cached dense matrix, rebuilt only on insert. Per-query recall reuses it.
        self._vec_cache: Optional[torch.Tensor] = None
        self._id_cache: Optional[List[int]] = None

    def _create_schema(self):
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                text          TEXT NOT NULL,
                source        TEXT,
                timestamp     REAL,
                step          INTEGER,
                novelty       REAL,
                superseded    INTEGER DEFAULT 0,
                valid_until   REAL,
                superseded_by INTEGER,
                access_count  INTEGER DEFAULT 1,
                last_retrieved REAL,
                version_id    TEXT,
                feedback      TEXT,
                needs_review  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS facts_vecs (
                id INTEGER PRIMARY KEY REFERENCES facts(id),
                vec BLOB
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                text, content='facts', content_rowid='id'
            );
            CREATE TABLE IF NOT EXISTS semantic_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source_cluster_ids TEXT,
                confidence REAL DEFAULT 0.5,
                created_at REAL,
                access_count INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS semantic_vecs (
                id INTEGER PRIMARY KEY REFERENCES semantic_facts(id),
                vec BLOB
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
                text, content='semantic_facts', content_rowid='id'
            );
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER,
                action TEXT,
                correction_text TEXT,
                timestamp REAL,
                strategy_used TEXT
            );
            """
        )
        c.commit()

    def _migrate_schema(self):
        """Add missing columns to existing DBs (ALTER TABLE)."""
        c = self.conn
        try:
            cols = [row[1] for row in c.execute("PRAGMA table_info(facts)").fetchall()]
            migrations = {
                "access_count": "ALTER TABLE facts ADD COLUMN access_count INTEGER DEFAULT 1",
                "last_retrieved": "ALTER TABLE facts ADD COLUMN last_retrieved REAL",
                "version_id": "ALTER TABLE facts ADD COLUMN version_id TEXT",
                "feedback": "ALTER TABLE facts ADD COLUMN feedback TEXT",
                "needs_review": "ALTER TABLE facts ADD COLUMN needs_review INTEGER DEFAULT 0",
            }
            for col, sql in migrations.items():
                if col not in cols:
                    try:
                        c.execute(sql)
                    except Exception:
                        pass
            c.commit()
        except Exception:
            pass

        try:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    source_cluster_ids TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL,
                    access_count INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS semantic_vecs (
                    id INTEGER PRIMARY KEY REFERENCES semantic_facts(id),
                    vec BLOB
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
                    text, content='semantic_facts', content_rowid='id'
                );
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id INTEGER,
                    action TEXT,
                    correction_text TEXT,
                    timestamp REAL,
                    strategy_used TEXT
                );
                """
            )
            c.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def insert_fact(
        self,
        text: str,
        source: str,
        timestamp: float,
        step: int,
        novelty: float,
        vec: torch.Tensor,
        version_id: Optional[str] = None,
        feedback: Optional[str] = None,
        needs_review: int = 0,
    ) -> int:
        v = vec.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
        v = v / norm
        self.conn.execute("BEGIN")
        try:
            cur = self.conn.execute(
                "INSERT INTO facts(text, source, timestamp, step, novelty, access_count, last_retrieved, version_id, feedback, needs_review)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (text, source, float(timestamp), int(step), float(novelty), 1, float(timestamp), version_id, feedback, int(needs_review)),
            )
            fid = cur.lastrowid
            self.conn.execute(
                "INSERT INTO facts_vecs(id, vec) VALUES (?,?)",
                (fid, v.cpu().contiguous().numpy().tobytes()),
            )
            self.conn.execute(
                "INSERT INTO facts_fts(rowid, text) VALUES (?,?)", (fid, text)
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._vec_cache = None
        return fid

    def insert_semantic_fact(
        self,
        text: str,
        source_cluster_ids: List[int],
        confidence: float,
        vec: torch.Tensor,
    ) -> int:
        v = vec.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
        v = v / norm
        self.conn.execute("BEGIN")
        try:
            cur = self.conn.execute(
                "INSERT INTO semantic_facts(text, source_cluster_ids, confidence, created_at, access_count)"
                " VALUES (?,?,?,?,?)",
                (text, json.dumps(source_cluster_ids), float(confidence), time.time(), 1),
            )
            fid = cur.lastrowid
            self.conn.execute(
                "INSERT INTO semantic_vecs(id, vec) VALUES (?,?)",
                (fid, v.cpu().contiguous().numpy().tobytes()),
            )
            self.conn.execute(
                "INSERT INTO semantic_fts(rowid, text) VALUES (?,?)", (fid, text)
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return fid

    def log_feedback(self, fact_id: int, action: str, correction_text: Optional[str] = None, strategy_used: Optional[str] = None):
        self.conn.execute(
            "INSERT INTO feedback_log(fact_id, action, correction_text, timestamp, strategy_used) VALUES (?,?,?,?,?)",
            (int(fact_id), action, correction_text, time.time(), strategy_used),
        )
        self.conn.commit()

    def get_feedback_log(self, fact_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if fact_id is not None:
            rows = self.conn.execute(
                "SELECT id, fact_id, action, correction_text, timestamp, strategy_used FROM feedback_log WHERE fact_id=? ORDER BY timestamp",
                (fact_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, fact_id, action, correction_text, timestamp, strategy_used FROM feedback_log ORDER BY timestamp"
            ).fetchall()
        return [
            {"id": r[0], "fact_id": r[1], "action": r[2], "correction_text": r[3], "timestamp": r[4], "strategy_used": r[5]}
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Dense helpers
    # ------------------------------------------------------------------ #
    def _load_dense(self) -> Tuple[torch.Tensor, List[int]]:
        if self._vec_cache is not None and self._id_cache is not None:
            return self._vec_cache, self._id_cache
        rows = self.conn.execute(
            "SELECT id, vec FROM facts_vecs ORDER BY id"
        ).fetchall()
        if not rows:
            self._vec_cache = torch.zeros((0, self.dim))
            self._id_cache = []
            return self._vec_cache, self._id_cache
        ids = [r[0] for r in rows]
        buf = b"".join(r[1] for r in rows)
        arr = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(
            len(rows), self.dim
        )
        self._vec_cache = arr.clone().contiguous()
        self._id_cache = ids
        return self._vec_cache, self._id_cache

    def _query_vec(self, query_vec: torch.Tensor) -> torch.Tensor:
        q = query_vec.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(q).clamp(min=1e-8)
        return q / norm

    def find_similar(
        self, query_vec: torch.Tensor, threshold: float = 0.65
    ) -> List[Dict[str, Any]]:
        """Active (non-superseded) facts whose dense cosine sim >= threshold."""
        vecs, ids = self._load_dense()
        if len(ids) == 0:
            return []
        q = self._query_vec(query_vec)
        sims = torch.mv(vecs, q)
        hits: List[Dict[str, Any]] = []
        for idx, score in enumerate(sims.tolist()):
            if score < threshold:
                continue
            row = self.conn.execute(
                "SELECT id, text, source, timestamp, step, novelty,"
                " superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review"
                " FROM facts WHERE id = ?",
                (ids[idx],),
            ).fetchone()
            if row and row[6] != 1:
                entry = self._to_dict(row)
                entry["similarity"] = score
                hits.append(entry)
        hits.sort(key=lambda h: h["similarity"], reverse=True)
        return hits

    # ------------------------------------------------------------------ #
    # Recall: fused RRF - preserved original logic with extensions
    # ------------------------------------------------------------------ #
    def recall(
        self,
        query_vec: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
        query_text: str = "",
        strategy: str = "hybrid_rrf",
    ) -> List[Dict[str, Any]]:
        """Hybrid lexical+dense retrieval, fused via Reciprocal Rank Fusion."""
        vecs, ids = self._load_dense()
        if len(ids) == 0:
            return []

        q = self._query_vec(query_vec)
        sims = torch.mv(vecs, q)
        top_dense = sims.topk(
            k=min(self._CANDS, len(ids))
        )
        dense_vals = top_dense.values.tolist()
        dense_idx = top_dense.indices.tolist()
        self.conn.execute("DROP TABLE IF EXISTS _dense_cands")
        self.conn.execute(
            "CREATE TEMPORARY TABLE _dense_cands ("
            "id INTEGER PRIMARY KEY, score REAL, drank INTEGER)"
        )
        self.conn.executemany(
            "INSERT INTO _dense_cands(id, score, drank) VALUES (?,?,?)",
            [
                (ids[idx], dense_vals[rank], rank + 1)
                for rank, idx in enumerate(dense_idx)
            ],
        )

        lex_match = _tokenize(query_text.split())
        has_lex = bool(lex_match)

        # Original used FULL OUTER JOIN - we emulate with UNION for compatibility
        # But we keep original SQL structure for correctness
        # Try FULL OUTER JOIN first, fallback to UNION if fails
        try:
            fused_sql = f"""
                WITH lex AS (
                    SELECT f.id AS id,
                           ROW_NUMBER() OVER (ORDER BY bm25(facts_fts) ASC) AS lrank
                    FROM facts_fts
                    JOIN facts f ON f.id = facts_fts.rowid
                    WHERE {{lex_where}} AND f.superseded = 0
                    LIMIT {{cands}}
                ),
                dense AS (
                    SELECT id, score AS dscore, drank
                    FROM _dense_cands
                ),
                fused AS (
                    SELECT COALESCE(lex.id, dense.id) AS id,
                           CASE WHEN lex.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + lex.lrank) END AS lex_part,
                           CASE WHEN dense.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + dense.drank) END AS dense_part,
                           COALESCE(lex.lrank, 0) AS lrank,
                           COALESCE(dense.drank, 0) AS drank,
                           dense.dscore AS dscore
                    FROM lex FULL OUTER JOIN dense ON lex.id = dense.id
                )
                SELECT f.id, f.text, f.source, f.timestamp, f.step, f.novelty,
                       f.superseded, f.valid_until, f.superseded_by,
                       f.access_count, f.last_retrieved, f.version_id, f.feedback, f.needs_review,
                       fused.fused_score, fused.lex_part, fused.dense_part,
                       fused.lrank, fused.drank, fused.dscore
                FROM (
                    SELECT id,
                           (lex_part + dense_part) AS fused_score,
                           lex_part, dense_part, lrank, drank, dscore
                    FROM fused
                ) fused
                JOIN facts f ON f.id = fused.id
                WHERE f.superseded = 0 AND fused.fused_score >= {{threshold}}
                ORDER BY fused.fused_score DESC, fused.dscore DESC
                LIMIT {{top_k}}
            """.format(
                lex_where="facts_fts MATCH ?" if has_lex else "1=0",
                cands=self._CANDS,
                rrf_k=self._RRF_K,
                threshold=threshold,
                top_k=top_k,
            )
            params = [lex_match] if has_lex else []
            cursor = self.conn.execute(fused_sql, params)
            rows = cursor.fetchall()
        except Exception as e:
            # Fallback: UNION emulation
            fused_sql = f"""
                WITH lex AS (
                    SELECT f.id AS id,
                           ROW_NUMBER() OVER (ORDER BY bm25(facts_fts) ASC) AS lrank
                    FROM facts_fts
                    JOIN facts f ON f.id = facts_fts.rowid
                    WHERE {{lex_where}} AND f.superseded = 0
                    LIMIT {{cands}}
                ),
                dense AS (
                    SELECT id, score AS dscore, drank
                    FROM _dense_cands
                ),
                fused AS (
                    SELECT COALESCE(lex.id, dense.id) AS id,
                           CASE WHEN lex.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + lex.lrank) END AS lex_part,
                           CASE WHEN dense.id IS NULL THEN 0.0
                                ELSE 1.0/({{rrf_k}} + dense.drank) END AS dense_part,
                           COALESCE(lex.lrank, 0) AS lrank,
                           COALESCE(dense.drank, 0) AS drank,
                           dense.dscore AS dscore
                    FROM lex LEFT JOIN dense ON lex.id = dense.id
                    UNION
                    SELECT dense.id AS id,
                           0.0 AS lex_part,
                           1.0/({{rrf_k}} + dense.drank) AS dense_part,
                           0 AS lrank,
                           dense.drank AS drank,
                           dense.dscore AS dscore
                    FROM dense LEFT JOIN lex ON dense.id = lex.id
                    WHERE lex.id IS NULL
                )
                SELECT f.id, f.text, f.source, f.timestamp, f.step, f.novelty,
                       f.superseded, f.valid_until, f.superseded_by,
                       f.access_count, f.last_retrieved, f.version_id, f.feedback, f.needs_review,
                       fused.fused_score, fused.lex_part, fused.dense_part,
                       fused.lrank, fused.drank, fused.dscore
                FROM (
                    SELECT id,
                           (lex_part + dense_part) AS fused_score,
                           lex_part, dense_part, lrank, drank, dscore
                    FROM fused
                ) fused
                JOIN facts f ON f.id = fused.id
                WHERE f.superseded = 0 AND fused.fused_score >= {{threshold}}
                ORDER BY fused.fused_score DESC, fused.dscore DESC
                LIMIT {{top_k}}
            """.format(
                lex_where="facts_fts MATCH ?" if has_lex else "1=0",
                cands=self._CANDS,
                rrf_k=self._RRF_K,
                threshold=threshold,
                top_k=top_k,
            )
            params = [lex_match] if has_lex else []
            rows = self.conn.execute(fused_sql, params).fetchall()

        results = []
        for r in rows:
            entry = {
                "id": r[0],
                "text": r[1],
                "source": r[2],
                "timestamp": r[3],
                "step": r[4],
                "novelty": r[5],
                "superseded": bool(r[6]),
                "valid_until": r[7],
                "superseded_by": r[8],
                "access_count": r[9] if r[9] is not None else 1,
                "last_retrieved": r[10],
                "version_id": r[11],
                "feedback": r[12],
                "needs_review": bool(r[13]) if r[13] is not None else False,
                "fused_score": r[14],
                "lexical_score": r[15],
                "dense_score": r[16],
                "lexical_rank": r[17],
                "dense_rank": r[18],
                "dense_similarity": r[19] if r[19] is not None else 0.0,
                "similarity": r[19] if r[19] is not None else 0.0,
            }
            results.append(entry)

        # Update access_count and last_retrieved
        if results:
            now = time.time()
            for entry in results:
                try:
                    self.conn.execute(
                        "UPDATE facts SET access_count = COALESCE(access_count,0)+1, last_retrieved=? WHERE id=?",
                        (now, entry["id"]),
                    )
                except Exception:
                    pass
            self.conn.commit()

        self.conn.execute("DROP TABLE IF EXISTS _dense_cands")
        return results

    def recall_semantic(
        self,
        query_vec: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
        query_text: str = "",
    ) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT id, vec FROM semantic_vecs ORDER BY id").fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        buf = b"".join(r[1] for r in rows)
        vecs = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(len(rows), self.dim)
        q = self._query_vec(query_vec)
        sims = torch.mv(vecs, q)
        top = sims.topk(k=min(top_k, len(ids)))
        results = []
        for rank, idx in enumerate(top.indices.tolist()):
            score = top.values[rank].item()
            if score < threshold:
                continue
            row = self.conn.execute(
                "SELECT id, text, source_cluster_ids, confidence, created_at, access_count FROM semantic_facts WHERE id=?",
                (ids[idx],),
            ).fetchone()
            if row:
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "source_cluster_ids": json.loads(row[2]) if row[2] else [],
                    "confidence": row[3],
                    "created_at": row[4],
                    "access_count": row[5],
                    "similarity": score,
                    "dense_similarity": score,
                    "fused_score": score,
                })
        if results:
            for entry in results:
                try:
                    self.conn.execute(
                        "UPDATE semantic_facts SET access_count = COALESCE(access_count,0)+1 WHERE id=?",
                        (entry["id"],),
                    )
                except:
                    pass
            self.conn.commit()
        return results

    def _to_dict(self, row) -> Dict[str, Any]:
        if len(row) >= 14:
            return {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "timestamp": row[3],
                "step": row[4],
                "novelty": row[5],
                "superseded": bool(row[6]),
                "valid_until": row[7],
                "superseded_by": row[8],
                "access_count": row[9] if row[9] is not None else 1,
                "last_retrieved": row[10],
                "version_id": row[11] if len(row) > 11 else None,
                "feedback": row[12] if len(row) > 12 else None,
                "needs_review": bool(row[13]) if len(row) > 13 and row[13] is not None else False,
            }
        else:
            return {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "timestamp": row[3],
                "step": row[4],
                "novelty": row[5],
                "superseded": bool(row[6]),
                "valid_until": row[7],
                "superseded_by": row[8],
            }

    def supersede(self, old_id: int, new_id: int, now: Optional[float] = None) -> int:
        cur = self.conn.execute(
            "UPDATE facts SET superseded = 1, valid_until = ?,"
            " superseded_by = ? WHERE id = ? AND superseded = 0",
            (now if now is not None else time.time(), new_id, old_id),
        )
        return cur.rowcount

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty,"
            " superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review FROM facts ORDER BY id"
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_active_texts(self):
        rows = self.conn.execute("SELECT LOWER(text) FROM facts WHERE superseded=0").fetchall()
        return set(r[0].strip() for r in rows)

    def get_active_facts(self):
        rows = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty, superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review FROM facts WHERE superseded=0 ORDER BY id"
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_fact(self, fact_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve single fact by ID with all metadata."""
        row = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty, superseded, valid_until, superseded_by, access_count, last_retrieved, version_id, feedback, needs_review FROM facts WHERE id = ?",
            (int(fact_id),),
        ).fetchone()
        if row is None:
            return None
        return self._to_dict(row)

    def update_fact(self, fact_id: int, new_text: str, new_vec: Optional[torch.Tensor] = None) -> int:
        """Update fact text and re-embed vector. Invalidates caches."""
        fact_id = int(fact_id)
        # Check exists
        existing = self.conn.execute("SELECT id FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if existing is None:
            return 0
        
        # Embed if vec not provided
        if new_vec is None:
            try:
                # Try to import anchor and embed
                from src.model.onnx_anchor import OnnxMiniLM
                anchor = OnnxMiniLM()
                new_vec = anchor.embed(new_text, to_torch=True)
            except Exception:
                try:
                    from .onnx_anchor import OnnxMiniLM
                    anchor = OnnxMiniLM()
                    new_vec = anchor.embed(new_text, to_torch=True)
                except Exception as e:
                    # Fallback: keep old vector if embedding fails
                    row = self.conn.execute("SELECT vec FROM facts_vecs WHERE id = ?", (fact_id,)).fetchone()
                    if row:
                        import numpy as np
                        vec_np = np.frombuffer(row[0], dtype=np.float32)
                        new_vec = torch.from_numpy(vec_np)
                    else:
                        new_vec = torch.randn(self.dim)

        v = new_vec.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
        v = v / norm

        self.conn.execute("BEGIN")
        try:
            # Update facts table
            self.conn.execute(
                "UPDATE facts SET text = ?, timestamp = ? WHERE id = ?",
                (new_text, time.time(), fact_id),
            )
            # Update vec
            self.conn.execute(
                "UPDATE facts_vecs SET vec = ? WHERE id = ?",
                (v.cpu().contiguous().numpy().tobytes(), fact_id),
            )
            # Rebuild FTS - simplest: delete and insert
            self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
            self.conn.execute(
                "INSERT INTO facts_fts(rowid, text) VALUES (?,?)", (fact_id, new_text)
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        
        self._vec_cache = None
        self._id_cache = None
        return 1

    def delete_fact(self, fact_id: int) -> int:
        """Delete fact entirely from all tables. Invalidates caches."""
        fact_id = int(fact_id)
        self.conn.execute("BEGIN")
        try:
            cur = self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            deleted = cur.rowcount
            self.conn.execute("DELETE FROM facts_vecs WHERE id = ?", (fact_id,))
            self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
            # Also clean feedback log? Keep for audit, but optional
            # self.conn.execute("DELETE FROM feedback_log WHERE fact_id = ?", (fact_id,))
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        
        self._vec_cache = None
        self._id_cache = None
        return deleted

    def total_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()
        return row[0] if row else 0

    def active_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM facts WHERE superseded=0").fetchone()
        return row[0] if row else 0

    def checkpoint(self):
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except:
            pass

    def export_to(self, dest_path: str) -> Path:
        self.checkpoint()
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path == ":memory:":
            backup_conn = sqlite3.connect(str(dest))
            self.conn.backup(backup_conn)
            backup_conn.close()
        else:
            src = Path(self.db_path)
            if src.exists():
                shutil.copy2(str(src), str(dest))
                for suffix in ["-wal", "-shm"]:
                    s = Path(str(src) + suffix)
                    if s.exists():
                        try:
                            shutil.copy2(str(s), str(dest) + suffix)
                        except:
                            pass
            else:
                backup_conn = sqlite3.connect(str(dest))
                self.conn.backup(backup_conn)
                backup_conn.close()
        return dest

    def clear_all(self):
        self.conn.execute("DELETE FROM facts")
        self.conn.execute("DELETE FROM facts_vecs")
        self.conn.execute("DELETE FROM facts_fts")
        self.conn.execute("DELETE FROM semantic_facts")
        self.conn.execute("DELETE FROM semantic_vecs")
        self.conn.execute("DELETE FROM semantic_fts")
        self.conn.execute("DELETE FROM feedback_log")
        self.conn.commit()
        self._vec_cache = None
        self._id_cache = None

    def close(self):
        try:
            self.conn.close()
        except:
            pass

    def get_all_embeddings(self) -> Tuple[torch.Tensor, List[int]]:
        return self._load_dense()
