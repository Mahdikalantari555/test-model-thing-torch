import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ponytail: SQLite WAL + FTS5 (BM25 lexical) + pre-normalized dense MVM fused via
# Reciprocal Rank Fusion, replacing O(N) JSON linear scan. stdlib sqlite3 + torch only.
# Ceiling: single-process; upgrade to HNSW/FAISS only if 100k+ facts miss 5ms on GPU.


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
                superseded_by INTEGER
            );
            CREATE TABLE IF NOT EXISTS facts_vecs (
                id INTEGER PRIMARY KEY REFERENCES facts(id),
                vec BLOB
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                text, content='facts', content_rowid='id'
            );
            """
        )
        c.commit()

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
    ) -> int:
        v = vec.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
        v = v / norm
        self.conn.execute("BEGIN")
        try:
            cur = self.conn.execute(
                "INSERT INTO facts(text, source, timestamp, step, novelty)"
                " VALUES (?,?,?,?,?)",
                (text, source, float(timestamp), int(step), float(novelty)),
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
        self._vec_cache = None  # invalidate; next recall rebuilds
        return fid

    # ------------------------------------------------------------------ #
    # Dense helpers
    # ------------------------------------------------------------------ #
    def _load_dense(self) -> tuple[torch.Tensor, List[int]]:
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
                " superseded, valid_until, superseded_by"
                " FROM facts WHERE id = ?",
                (ids[idx],),
            ).fetchone()
            if row and row[6] != 1:  # skip superseded
                entry = self._to_dict(row)
                entry["similarity"] = score
                hits.append(entry)
        hits.sort(key=lambda h: h["similarity"], reverse=True)
        return hits

    # ------------------------------------------------------------------ #
    # Recall: fused RRF in a single SQL CTE
    # ------------------------------------------------------------------ #
    def recall(
        self,
        query_vec: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
        query_text: str = "",
    ) -> List[Dict[str, Any]]:
        """Hybrid lexical+dense retrieval, fused via Reciprocal Rank Fusion.

        Returns active facts ranked by fused RRF score. Each result carries
        both lexical and dense contribution scores.
        """
        vecs, ids = self._load_dense()
        if len(ids) == 0:
            return []

        # Dense signal: top candidates from a single MVM (no re-norm allocation).
        q = self._query_vec(query_vec)
        sims = torch.mv(vecs, q)
        top_dense = sims.topk(
            k=min(self._CANDS, len(ids))
        )
        dense_vals = top_dense.values.tolist()
        dense_idx = top_dense.indices.tolist()
        # Temp table carries (id, score, drank) so fusion runs in-database.
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

        # Lexical signal: BM25 rank of FTS5 matches for the query text.
        lex_match = _tokenize(query_text.split())
        has_lex = bool(lex_match)
        fused_sql = """
            WITH lex AS (
                SELECT f.id AS id,
                       ROW_NUMBER() OVER (ORDER BY bm25(facts_fts) ASC) AS lrank
                FROM facts_fts
                JOIN facts f ON f.id = facts_fts.rowid
                WHERE {lex_where} AND f.superseded = 0
                LIMIT {cands}
            ),
            dense AS (
                SELECT id, score AS dscore, drank
                FROM _dense_cands
            ),
            fused AS (
                SELECT COALESCE(lex.id, dense.id) AS id,
                       CASE WHEN lex.id IS NULL THEN 0.0
                            ELSE 1.0/({rrf_k} + lex.lrank) END AS lex_part,
                       CASE WHEN dense.id IS NULL THEN 0.0
                            ELSE 1.0/({rrf_k} + dense.drank) END AS dense_part,
                       COALESCE(lex.lrank, 0) AS lrank,
                       COALESCE(dense.drank, 0) AS drank,
                       dense.dscore AS dscore
                FROM lex FULL OUTER JOIN dense ON lex.id = dense.id
            )
            SELECT f.id, f.text, f.source, f.timestamp, f.step, f.novelty,
                   f.superseded, f.valid_until, f.superseded_by,
                   fused.fused_score, fused.lex_part, fused.dense_part,
                   fused.lrank, fused.drank, fused.dscore
            FROM (
                SELECT id,
                       (lex_part + dense_part) AS fused_score,
                       lex_part, dense_part, lrank, drank, dscore
                FROM fused
            ) fused
            JOIN facts f ON f.id = fused.id
            WHERE f.superseded = 0 AND fused.fused_score >= {threshold}
            ORDER BY fused.fused_score DESC, fused.dscore DESC
            LIMIT {top_k}
        """.format(
            lex_where="facts_fts MATCH ?" if has_lex else "1=0",
            cands=self._CANDS,
            rrf_k=self._RRF_K,
            threshold=threshold,
            top_k=top_k,
        )
        params = [lex_match] if has_lex else []
        fused = self.conn.execute(fused_sql, params)
        results: List[Dict[str, Any]] = []
        for row in fused:
            results.append(
                {
                    "id": row[0],
                    "text": row[1],
                    "source": row[2],
                    "timestamp": row[3],
                    "step": row[4],
                    "novelty": row[5],
                    "superseded": bool(row[6]),
                    "valid_until": row[7],
                    "superseded_by": row[8],
                    "fused_score": row[9],
                    "lexical_score": row[10],
                    "dense_score": row[11],
                    "lexical_rank": row[12],
                    "dense_rank": row[13],
                    "dense_similarity": row[14],
                }
            )
        self.conn.execute("DROP TABLE IF EXISTS _dense_cands")
        return results

    # ------------------------------------------------------------------ #
    # Truth maintenance (TMS)
    # ------------------------------------------------------------------ #
    def supersede(self, old_id: int, new_id: int, now: Optional[float] = None) -> int:
        """Mark `old_id` as superseded by `new_id`, preserving the audit trail.

        Returns number of rows updated (0 if old_id unknown).
        """
        cur = self.conn.execute(
            "UPDATE facts SET superseded = 1, valid_until = ?,"
            " superseded_by = ? WHERE id = ? AND superseded = 0",
            (now if now is not None else time.time(), new_id, old_id),
        )
        return cur.rowcount

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """All facts, including superseded, in insertion order."""
        rows = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty,"
            " superseded, valid_until, superseded_by FROM facts ORDER BY id"
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def active_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE superseded = 0"
        ).fetchone()[0]

    def total_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def get_active_facts(self) -> List[Dict[str, Any]]:
        """Active (non-superseded) fact records, insertion order."""
        rows = self.conn.execute(
            "SELECT id, text, source, timestamp, step, novelty,"
            " superseded, valid_until, superseded_by"
            " FROM facts WHERE superseded = 0 ORDER BY id"
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get_active_texts(self) -> set:
        """Lowercased active fact texts, for dedup on teach."""
        rows = self.conn.execute(
            "SELECT lower(text) FROM facts WHERE superseded = 0"
        ).fetchall()
        return {r[0] for r in rows}

    @staticmethod
    def _to_dict(row) -> Dict[str, Any]:
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

    def clear_all(self):
        """Wipe every fact, vector and FTS entry (used by memory reset)."""
        self.conn.execute("BEGIN")
        try:
            self.conn.execute("DELETE FROM facts")
            self.conn.execute("DELETE FROM facts_vecs")
            # external-content FTS5: rebuild from (now empty) content table
            self.conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
            self.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'facts'")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        self._vec_cache = None
        self._id_cache = None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def checkpoint(self):
        """Fold WAL into the main DB and truncate, for clean export."""
        if self.db_path != ":memory:":
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.commit()

    def export_to(self, dest_path: str) -> Path:
        """Materialize a consistent WAL-mode DB at dest_path."""
        self.conn.commit()
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        src_conn = sqlite3.connect(str(dest))
        self.conn.backup(src_conn)
        src_conn.execute("PRAGMA journal_mode=WAL")
        src_conn.close()
        return dest

    def close(self):
        self.checkpoint()
        self.conn.close()
