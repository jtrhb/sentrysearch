"""PostgreSQL + pgvector vector store."""

import hashlib
import os
from datetime import datetime, timezone

import psycopg
from pgvector.psycopg import register_vector


class BackendMismatchError(RuntimeError):
    """Raised on backend mismatch (kept for backward compatibility)."""


def _make_chunk_id(source_file: str, start_time: float) -> str:
    """Deterministic chunk ID from source file + start time."""
    raw = f"{source_file}:{start_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_index(database_url: str | None = None) -> tuple[str | None, str | None]:
    """Return ('gemini', None) if indexed data exists, else (None, None)."""
    try:
        url = database_url or os.environ.get("DATABASE_URL", "")
        if not url:
            return None, None
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS("
                    "  SELECT 1 FROM information_schema.tables"
                    "  WHERE table_name = 'chunks'"
                    ")"
                )
                if not cur.fetchone()[0]:
                    return None, None
                cur.execute("SELECT EXISTS(SELECT 1 FROM chunks LIMIT 1)")
                return ("gemini", None) if cur.fetchone()[0] else (None, None)
    except Exception:
        return None, None


def detect_backend(database_url: str | None = None) -> str | None:
    """Return the backend that has indexed data, or None."""
    backend, _ = detect_index(database_url)
    return backend


class SentryStore:
    """Persistent vector store backed by PostgreSQL + pgvector."""

    def __init__(self, database_url: str | None = None, **kwargs):
        """Initialize store.

        Args:
            database_url: PostgreSQL connection string. Falls back to
                          DATABASE_URL environment variable.
            **kwargs: Ignored (accepts legacy backend/model/db_path params).
        """
        self._url = database_url or os.environ["DATABASE_URL"]
        self._conn = psycopg.connect(self._url)
        register_vector(self._conn)
        self._ensure_schema()

    def _ensure_schema(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    embedding vector(3072) NOT NULL,
                    source_file TEXT NOT NULL,
                    start_time DOUBLE PRECISION NOT NULL,
                    end_time DOUBLE PRECISION NOT NULL,
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_source_file_idx
                ON chunks (source_file)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    current_path TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    filename TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    asset_id TEXT PRIMARY KEY,
                    character_consistency DOUBLE PRECISION,
                    scene_consistency DOUBLE PRECISION,
                    ai_score DOUBLE PRECISION,
                    max_similarity DOUBLE PRECISION,
                    similar_to TEXT,
                    category_scores JSONB NOT NULL DEFAULT '{}',
                    issues JSONB NOT NULL DEFAULT '[]',
                    severity_counts JSONB NOT NULL DEFAULT '{}',
                    quality_grade TEXT,
                    grade_score DOUBLE PRECISION,
                    overall_score DOUBLE PRECISION,
                    summary TEXT,
                    details JSONB NOT NULL DEFAULT '{}',
                    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        self._conn.commit()

    @property
    def collection(self):
        """Not applicable for pgvector — exists for interface compatibility."""
        return None

    def get_backend(self) -> str:
        return "gemini"

    def get_model(self) -> str | None:
        return None

    def check_backend(self, backend: str) -> None:
        if backend != "gemini":
            raise BackendMismatchError(
                f"Only gemini backend is supported, got: {backend}"
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Store a single chunk embedding with metadata."""
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (id, embedding, source_file, start_time, end_time, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    source_file = EXCLUDED.source_file,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    indexed_at = EXCLUDED.indexed_at
                """,
                (
                    chunk_id, embedding, metadata["source_file"],
                    float(metadata["start_time"]), float(metadata["end_time"]), now,
                ),
            )
        self._conn.commit()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Batch-store chunks. Each dict must have 'embedding' and metadata keys."""
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            for chunk in chunks:
                chunk_id = _make_chunk_id(chunk["source_file"], chunk["start_time"])
                cur.execute(
                    """
                    INSERT INTO chunks (id, embedding, source_file, start_time, end_time, indexed_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        indexed_at = EXCLUDED.indexed_at
                    """,
                    (
                        chunk_id, chunk["embedding"], chunk["source_file"],
                        float(chunk["start_time"]), float(chunk["end_time"]), now,
                    ),
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Return top N results ranked by cosine similarity."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            if cur.fetchone()[0] == 0:
                return []

            cur.execute(
                """
                SELECT source_file, start_time, end_time,
                       1 - distance AS score, distance
                FROM (
                    SELECT source_file, start_time, end_time,
                           embedding <=> %s AS distance
                    FROM chunks
                    ORDER BY distance
                    LIMIT %s
                ) ranked
                """,
                (query_embedding, n_results),
            )
            rows = cur.fetchall()

        return [
            {
                "source_file": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "score": float(row[3]),
                "distance": float(row[4]),
            }
            for row in rows
        ]

    def is_indexed(self, source_file: str) -> bool:
        """Check whether any chunks from source_file are already stored."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM chunks WHERE source_file = %s)",
                (source_file,),
            )
            return cur.fetchone()[0]

    def remove_file(self, source_file: str) -> int:
        """Remove all chunks for a given source file. Returns count removed."""
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE source_file = %s", (source_file,),
            )
            count = cur.rowcount
        self._conn.commit()
        return count

    def get_stats(self) -> dict:
        """Return store statistics."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            total = cur.fetchone()[0]

            if total == 0:
                return {"total_chunks": 0, "unique_source_files": 0, "source_files": []}

            cur.execute(
                "SELECT DISTINCT source_file FROM chunks ORDER BY source_file"
            )
            source_files = [row[0] for row in cur.fetchall()]

        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "source_files": source_files,
        }

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    def register_asset(
        self,
        asset_id: str,
        current_path: str,
        original_path: str | None = None,
        filename: str | None = None,
    ) -> None:
        """Insert or update an asset record."""
        now = datetime.now(timezone.utc)
        if original_path is None:
            original_path = current_path
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO assets
                    (asset_id, current_path, original_path, filename,
                     status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, 'active', %s, %s)
                ON CONFLICT (asset_id) DO UPDATE SET
                    current_path = EXCLUDED.current_path,
                    original_path = EXCLUDED.original_path,
                    filename = EXCLUDED.filename,
                    status = 'active',
                    updated_at = EXCLUDED.updated_at
                """,
                (asset_id, current_path, original_path, filename, now, now),
            )
        self._conn.commit()

    def update_asset_path(self, asset_id: str, new_path: str) -> None:
        """Update the current_path and updated_at for an asset."""
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE assets SET current_path = %s, updated_at = %s "
                "WHERE asset_id = %s",
                (new_path, now, asset_id),
            )
        self._conn.commit()

    def get_asset(self, asset_id: str) -> dict | None:
        """Get an asset by its ID. Returns dict or None."""
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT asset_id, current_path, original_path, filename,
                          status, created_at, updated_at
                   FROM assets WHERE asset_id = %s""",
                (asset_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "asset_id": row[0],
            "current_path": row[1],
            "original_path": row[2],
            "filename": row[3],
            "status": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    def get_asset_by_path(self, path: str) -> dict | None:
        """Get an asset by its current_path. Returns dict or None."""
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT asset_id, current_path, original_path, filename,
                          status, created_at, updated_at
                   FROM assets WHERE current_path = %s""",
                (path,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "asset_id": row[0],
            "current_path": row[1],
            "original_path": row[2],
            "filename": row[3],
            "status": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    def reconcile_asset(self, asset_id: str, new_path: str) -> bool:
        """If asset exists, update its path and set status to 'active'.

        Returns True if the asset was found and updated, False otherwise.
        """
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE assets SET current_path = %s, status = 'active', "
                "updated_at = %s WHERE asset_id = %s",
                (new_path, now, asset_id),
            )
            updated = cur.rowcount > 0
        self._conn.commit()
        return updated

    def mark_asset_stale(self, path: str) -> None:
        """Mark an asset as stale by its current_path."""
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE assets SET status = 'stale', updated_at = %s "
                "WHERE current_path = %s",
                (now, path),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    def save_evaluation(self, asset_id: str, evaluation: dict) -> None:
        """Upsert an evaluation record (combines old scores + critique)."""
        import json

        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluations
                    (asset_id, character_consistency, scene_consistency,
                     ai_score, max_similarity, similar_to,
                     category_scores, issues, severity_counts,
                     quality_grade, grade_score, overall_score,
                     summary, details, evaluated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (asset_id) DO UPDATE SET
                    character_consistency = EXCLUDED.character_consistency,
                    scene_consistency = EXCLUDED.scene_consistency,
                    ai_score = EXCLUDED.ai_score,
                    max_similarity = EXCLUDED.max_similarity,
                    similar_to = EXCLUDED.similar_to,
                    category_scores = EXCLUDED.category_scores,
                    issues = EXCLUDED.issues,
                    severity_counts = EXCLUDED.severity_counts,
                    quality_grade = EXCLUDED.quality_grade,
                    grade_score = EXCLUDED.grade_score,
                    overall_score = EXCLUDED.overall_score,
                    summary = EXCLUDED.summary,
                    details = EXCLUDED.details,
                    evaluated_at = EXCLUDED.evaluated_at
                """,
                (
                    asset_id,
                    evaluation.get("character_consistency"),
                    evaluation.get("scene_consistency"),
                    evaluation.get("ai_score"),
                    evaluation.get("max_similarity"),
                    evaluation.get("similar_to"),
                    json.dumps(evaluation.get("category_scores", {})),
                    json.dumps(evaluation.get("issues", [])),
                    json.dumps(evaluation.get("severity_counts", {})),
                    evaluation.get("quality_grade"),
                    evaluation.get("grade_score"),
                    evaluation.get("overall_score"),
                    evaluation.get("summary"),
                    json.dumps(evaluation.get("details", {})),
                    now,
                ),
            )
        self._conn.commit()

    def get_evaluation(self, asset_id: str) -> dict | None:
        """Get evaluation for a specific asset. Returns dict or None."""
        import json

        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT asset_id, character_consistency, scene_consistency,
                          ai_score, max_similarity, similar_to,
                          category_scores, issues, severity_counts,
                          quality_grade, grade_score, overall_score,
                          summary, details, evaluated_at
                   FROM evaluations WHERE asset_id = %s""",
                (asset_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None

        def _parse(val):
            return json.loads(val) if isinstance(val, str) else val

        return {
            "asset_id": row[0],
            "character_consistency": row[1],
            "scene_consistency": row[2],
            "ai_score": row[3],
            "max_similarity": row[4],
            "similar_to": row[5],
            "category_scores": _parse(row[6]),
            "issues": _parse(row[7]),
            "severity_counts": _parse(row[8]),
            "quality_grade": row[9],
            "grade_score": row[10],
            "overall_score": row[11],
            "summary": row[12],
            "details": _parse(row[13]),
            "evaluated_at": row[14],
        }

    def get_evaluations(
        self,
        min_overall: float | None = None,
        max_ai_score: float | None = None,
        max_grade: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query evaluations with optional filters."""
        import json

        conditions: list[str] = []
        params: list = []

        if min_overall is not None:
            conditions.append("overall_score >= %s")
            params.append(min_overall)
        if max_ai_score is not None:
            conditions.append("ai_score <= %s")
            params.append(max_ai_score)
        if max_grade is not None:
            conditions.append("grade_score <= %s")
            thresholds = {"A": 100, "B": 89, "C": 74, "D": 59, "F": 39}
            params.append(thresholds.get(max_grade.upper(), 100))

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT asset_id, character_consistency, scene_consistency,
                           ai_score, max_similarity, similar_to,
                           category_scores, issues, severity_counts,
                           quality_grade, grade_score, overall_score,
                           summary, details, evaluated_at
                    FROM evaluations {where}
                    ORDER BY overall_score DESC
                    LIMIT %s""",
                params,
            )
            rows = cur.fetchall()

        def _parse(val):
            return json.loads(val) if isinstance(val, str) else val

        return [
            {
                "asset_id": row[0],
                "character_consistency": row[1],
                "scene_consistency": row[2],
                "ai_score": row[3],
                "max_similarity": row[4],
                "similar_to": row[5],
                "category_scores": _parse(row[6]),
                "issues": _parse(row[7]),
                "severity_counts": _parse(row[8]),
                "quality_grade": row[9],
                "grade_score": row[10],
                "overall_score": row[11],
                "summary": row[12],
                "details": _parse(row[13]),
                "evaluated_at": row[14],
            }
            for row in rows
        ]

    def close(self):
        """Close the database connection."""
        self._conn.close()
