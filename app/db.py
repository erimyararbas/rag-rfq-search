import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS rfqs (
    rfq_id text PRIMARY KEY,
    date date,
    customer text,
    customer_industry text,
    customer_country text,
    language text,
    title text,
    material text,
    process text,
    quantity int,
    status text,
    family_id text
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id serial PRIMARY KEY,
    rfq_id text NOT NULL REFERENCES rfqs(rfq_id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    content text NOT NULL,
    embedding vector(768),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_rfq_id_idx ON chunks (rfq_id);
"""


def connect(url: str | None = None, register: bool = True) -> psycopg.Connection:
    """Open a connection. Never creates the extension: that is init_schema's job.

    register=False is for callers (ingest, test fixtures) that call init_schema next,
    which creates the extension and then registers the vector type.
    """
    conn = psycopg.connect(url or get_settings().database_url, connect_timeout=5)
    if not register:
        return conn
    try:
        register_vector(conn)
    except psycopg.ProgrammingError as e:  # pgvector's "vector type not found"
        conn.close()
        raise RuntimeError(
            "pgvector extension/type not found in this database; "
            "run ingest first (python -m app.ingest) to initialise the schema"
        ) from e
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(SCHEMA)
    conn.commit()
    register_vector(conn)
