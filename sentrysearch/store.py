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
                CREATE TABLE IF NOT EXISTS video_critiques (
                    source_file TEXT PRIMARY KEY,
                    issues JSONB NOT NULL DEFAULT '[]',
                    category_scores JSONB NOT NULL DEFAULT '{}',
                    severity_counts JSONB NOT NULL DEFAULT '{}',
                    quality_grade TEXT,
                    grade_score DOUBLE PRECISION,
                    summary TEXT,
                    critiqued_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_scores (
                    source_file TEXT PRIMARY KEY,
                    character_consistency DOUBLE PRECISION,
                    scene_consistency DOUBLE PRECISION,
                    ai_score DOUBLE PRECISION,
                    max_similarity DOUBLE PRECISION,
                    similar_to TEXT,
                    overall_score DOUBLE PRECISION,
                    details JSONB NOT NULL DEFAULT '{}',
                    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    # Video critiques
    # ------------------------------------------------------------------

    def save_critique(self, source_file: str, critique: dict) -> None:
        """Save or update a video critique."""
        import json

        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_critiques
                    (source_file, issues, category_scores, severity_counts,
                     quality_grade, grade_score, summary, critiqued_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_file) DO UPDATE SET
                    issues = EXCLUDED.issues,
                    category_scores = EXCLUDED.category_scores,
                    severity_counts = EXCLUDED.severity_counts,
                    quality_grade = EXCLUDED.quality_grade,
                    grade_score = EXCLUDED.grade_score,
                    summary = EXCLUDED.summary,
                    critiqued_at = EXCLUDED.critiqued_at
                """,
                (
                    source_file,
                    json.dumps(critique.get("issues", [])),
                    json.dumps(critique.get("category_scores", {})),
                    json.dumps(critique.get("severity_counts", {})),
                    critique.get("quality_grade"),
                    critique.get("grade_score"),
                    critique.get("summary"),
                    now,
                ),
            )
        self._conn.commit()

    def get_critique(self, source_file: str) -> dict | None:
        """Get critique for a specific video."""
        import json

        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT issues, category_scores, severity_counts,
                          quality_grade, grade_score, summary
                   FROM video_critiques WHERE source_file = %s""",
                (source_file,),
            )
            row = cur.fetchone()
        if row is None:
            return None

        def _parse(val):
            return json.loads(val) if isinstance(val, str) else val

        return {
            "source_file": source_file,
            "issues": _parse(row[0]),
            "category_scores": _parse(row[1]),
            "severity_counts": _parse(row[2]),
            "quality_grade": row[3],
            "grade_score": row[4],
            "summary": row[5],
        }

    def get_critiques(
        self,
        max_grade: str | None = None,
        min_issues: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query video critiques with optional filters."""
        import json

        grade_order = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}
        conditions = []
        params = []

        if max_grade is not None:
            conditions.append("grade_score <= %s")
            # Map grade letter to score threshold
            thresholds = {"A": 100, "B": 89, "C": 74, "D": 59, "F": 39}
            params.append(thresholds.get(max_grade.upper(), 100))

        params.append(limit)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._conn.cursor() as cur:
            cur.execute(
                f"""SELECT source_file, issues, category_scores,
                           severity_counts, quality_grade, grade_score, summary
                    FROM video_critiques {where}
                    ORDER BY grade_score ASC
                    LIMIT %s""",
                params,
            )
            rows = cur.fetchall()

        def _parse(val):
            return json.loads(val) if isinstance(val, str) else val

        results = []
        for row in rows:
            entry = {
                "source_file": row[0],
                "issues": _parse(row[1]),
                "category_scores": _parse(row[2]),
                "severity_counts": _parse(row[3]),
                "quality_grade": row[4],
                "grade_score": row[5],
                "summary": row[6],
            }
            if min_issues is not None:
                if len(entry["issues"]) < min_issues:
                    continue
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Video scores
    # ------------------------------------------------------------------

    def save_score(self, source_file: str, scores: dict) -> None:
        """Save or update video scores."""
        import json

        now = datetime.now(timezone.utc)
        consistency = scores.get("consistency", {})
        ai = scores.get("ai_detection", {})
        similarity = scores.get("similarity", {})

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_scores
                    (source_file, character_consistency, scene_consistency,
                     ai_score, max_similarity, similar_to, overall_score,
                     details, scored_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_file) DO UPDATE SET
                    character_consistency = EXCLUDED.character_consistency,
                    scene_consistency = EXCLUDED.scene_consistency,
                    ai_score = EXCLUDED.ai_score,
                    max_similarity = EXCLUDED.max_similarity,
                    similar_to = EXCLUDED.similar_to,
                    overall_score = EXCLUDED.overall_score,
                    details = EXCLUDED.details,
                    scored_at = EXCLUDED.scored_at
                """,
                (
                    source_file,
                    consistency.get("character"),
                    consistency.get("scene"),
                    ai.get("score"),
                    similarity.get("max_similarity"),
                    similarity.get("similar_to"),
                    scores.get("overall_score"),
                    json.dumps(scores),
                    now,
                ),
            )
        self._conn.commit()

    def get_score(self, source_file: str) -> dict | None:
        """Get scores for a specific video."""
        import json

        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT details FROM video_scores WHERE source_file = %s",
                (source_file,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        details = row[0]
        if isinstance(details, str):
            details = json.loads(details)
        details["source_file"] = source_file
        return details

    def get_scores(
        self,
        min_overall: float | None = None,
        max_ai_score: float | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query video scores with optional filters."""
        import json

        conditions = []
        params = []

        if min_overall is not None:
            conditions.append("overall_score >= %s")
            params.append(min_overall)
        if max_ai_score is not None:
            conditions.append("ai_score <= %s")
            params.append(max_ai_score)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT source_file, details FROM video_scores "
                f"{where} ORDER BY overall_score DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            details = row[1]
            if isinstance(details, str):
                details = json.loads(details)
            details["source_file"] = row[0]
            results.append(details)
        return results

    def close(self):
        """Close the database connection."""
        self._conn.close()
