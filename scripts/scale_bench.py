"""Scale test (protocol fixed before the run; summarised in the README scale-test section).

Subcommands: estimate (embed throughput + size gate), ingest (resumable load, HNSW build,
VACUUM ANALYZE), bench (HNSW versus exact kNN). Uses its own database, `rag_scale`.
"""
import argparse
import json
import re
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_rfq  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.data.generate import generate_queries  # noqa: E402
from app.data.scale import chunk_stats, generate_scale_rfqs  # noqa: E402

# 50 batches (20,000 RFQs) were estimated at 87.9 min of embedding on CPU, over the
# 60-min gate in the protocol; reduced by explicit decision to 33 batches (13,200 RFQs).
N_BATCHES = 33
SHIPPED_DEPTH = 50  # app.retrieval.DEPTH
from app.db import connect, init_schema  # noqa: E402
from app.ingest import BATCH, RFQ_COLS  # noqa: E402
from app.retrieval import dense_candidates  # noqa: E402

SCALE_DB = "rag_scale"
INDEX = "chunks_embedding_hnsw"
EF_GRID = (10, 20, 40, 80, 200, 400)
N_QUERIES = 200
QUERY_SEED = 19
WARMUP = 10
K = 10
APP_DEPTH = 10
SAMPLE_N = 256
SAMPLE_SEED = 5
COMMIT_EVERY = 50
MAINT_MEM = "1GB"
OUT_DIR = ROOT / "eval" / "results"

EXACT_SQL = "SELECT chunk_id, rfq_id FROM chunks ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s"


# ---------------------------------------------------------------- pure helpers
def swap_db_url(url: str, dbname: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path="/" + dbname))


def recall_at_k(found: list, truth: list, k: int = 10) -> float:
    """Share of the first k truth items that appear in the first k found items."""
    t = list(truth)[:k]
    if not t:
        return 1.0
    return len(set(t) & set(list(found)[:k])) / len(t)


def percentile(vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100); 0.0 for empty input."""
    if not vals:
        return 0.0
    s = sorted(vals)
    pos = (len(s) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def latency_stats(ms: list[float]) -> dict:
    return {
        "mean_ms": sum(ms) / len(ms) if ms else 0.0,
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
    }


def collapse_to_rfq(rfq_ids: list[str]) -> list[str]:
    """Best chunk per RFQ: keep the first occurrence of each RFQ id (input is best-first)."""
    return list(dict.fromkeys(rfq_ids))


def stratified_sample(is_multi: list[bool], n: int, seed: int) -> list[int]:
    """n indices; chunks of multi-chunk RFQs are included in proportion to their share."""
    total = len(is_multi)
    if n >= total:
        return list(range(total))
    rng = random.Random(seed)
    multi = [i for i, m in enumerate(is_multi) if m]
    single = [i for i, m in enumerate(is_multi) if not m]
    n_multi = min(len(multi), round(n * len(multi) / total))
    n_single = n - n_multi
    if n_single > len(single):
        n_single = len(single)
        n_multi = n - n_single
    return sorted(rng.sample(multi, n_multi) + rng.sample(single, n_single))


def corpus_chunks(rfqs: list[dict]) -> tuple[list[str], list[bool]]:
    texts, multi = [], []
    for r in rfqs:
        cs = chunk_rfq(r)
        texts.extend(cs)
        multi.extend([len(cs) > 1] * len(cs))
    return texts, multi


def rfq_batches(rfqs: list[dict], max_chunks: int = BATCH):
    """Yield lists of (rfq, [chunks]); never splits an RFQ, at most max_chunks chunks per batch."""
    cur, n = [], 0
    for r in rfqs:
        cs = chunk_rfq(r)
        if cur and n + len(cs) > max_chunks:
            yield cur
            cur, n = [], 0
        cur.append((r, cs))
        n += len(cs)
    if cur:
        yield cur


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(name: str, data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(name: str, out_dir: Path) -> dict | None:
    p = out_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- database
def ensure_db(url: str) -> None:
    dbname = urlsplit(url).path.lstrip("/")
    with psycopg.connect(swap_db_url(url, "rag"), autocommit=True, connect_timeout=5) as c:
        if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", [dbname]).fetchone():
            c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


def open_scale_conn(url: str):
    """Connect; on first use create the schema and drop the HNSW index (built after the load)."""
    conn = connect(url, register=False)
    if conn.execute("SELECT to_regclass('public.chunks')").fetchone()[0] is None:
        init_schema(conn)
        conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
        conn.commit()
    else:
        from pgvector.psycopg import register_vector
        register_vector(conn)
        conn.commit()
    return conn


def index_exists(conn) -> bool:
    ok = conn.execute("SELECT to_regclass(%s)", [f"public.{INDEX}"]).fetchone()[0] is not None
    conn.commit()
    return ok


def ingest_corpus(conn, rfqs: list[dict], embedder, commit_every: int = COMMIT_EVERY,
                  log=print, progress_every: int = 1000) -> dict:
    """Resumable load: skips RFQs already present; commits every `commit_every` batches.

    Each batch holds whole RFQs (<= BATCH chunks), so a committed state never has an RFQ
    with a partial chunk set. Does not touch the HNSW index.
    """
    present = {r[0] for r in conn.execute("SELECT rfq_id FROM rfqs").fetchall()}
    conn.commit()
    pending = [r for r in rfqs if r["rfq_id"] not in present]
    total_chunks = sum(len(chunk_rfq(r)) for r in pending)
    ph = ", ".join(["%s"] * len(RFQ_COLS))
    t0 = time.perf_counter()
    done = 0
    next_log = progress_every
    for bi, batch in enumerate(rfq_batches(pending), start=1):
        texts = [c for _, cs in batch for c in cs]
        vecs = embedder.embed_passages(texts)
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO rfqs ({', '.join(RFQ_COLS)}) VALUES ({ph})",
                [[r[c] for c in RFQ_COLS] for r, _ in batch])
            it = iter(vecs)
            cur.executemany(
                "INSERT INTO chunks (rfq_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                [(r["rfq_id"], i, c, next(it)) for r, cs in batch for i, c in enumerate(cs)])
        done += len(texts)
        if bi % commit_every == 0:
            conn.commit()
        if done >= next_log:
            el = time.perf_counter() - t0
            eta = el / done * (total_chunks - done)
            log(f"  {done}/{total_chunks} chunks  elapsed {el / 60:.1f} min  ETA {eta / 60:.1f} min")
            next_log = (done // progress_every + 1) * progress_every
    conn.commit()
    return {
        "rfqs_present_before": len(present),
        "rfqs_loaded": len(pending),
        "chunks_loaded": done,
        "load_seconds": time.perf_counter() - t0,
    }


def build_index(conn, maintenance_work_mem: str = MAINT_MEM) -> float:
    conn.commit()
    conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
    conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
    t0 = time.perf_counter()
    conn.execute(f"CREATE INDEX {INDEX} ON chunks USING hnsw (embedding vector_cosine_ops)")
    conn.commit()
    return time.perf_counter() - t0


def db_sizes(conn, dbname: str) -> dict:
    out = {"database_bytes": conn.execute("SELECT pg_database_size(%s)", [dbname]).fetchone()[0]}
    for rel in ("chunks", INDEX, "chunks_tsv_gin"):
        a, b = conn.execute(
            "SELECT pg_relation_size(%s::regclass), pg_total_relation_size(%s::regclass)",
            [rel, rel]).fetchone()
        out[rel] = {"relation_bytes": a, "total_bytes": b}
    conn.commit()
    return out


# ---------------------------------------------------------------- bench core
def _plan(conn, vec, k: int = K) -> str:
    rows = conn.execute("EXPLAIN " + EXACT_SQL, {"v": vec, "k": k}).fetchall()
    return "\n".join(r[0] for r in rows)


def exact_topk(conn, vec, k: int) -> tuple[list[tuple[int, str]], str]:
    """Exact kNN: index and bitmap scans off, plan checked to be a sequential scan."""
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")
        plan = _plan(conn, vec, k=k)
        if "Seq Scan on chunks" not in plan or INDEX in plan:
            raise AssertionError("exact query did not use a sequential scan:\n" + plan)
        rows = conn.execute(EXACT_SQL, {"v": vec, "k": k}).fetchall()
        raise psycopg.Rollback()
    return rows, plan


class PrecomputedEmbedder:
    """Maps query text to its precomputed vector, so search timing excludes embedding."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def embed_query(self, text: str) -> list[float]:
        return self.table[text]


def scan_node(plan: str) -> str:
    """The scan that feeds the LIMIT: HNSW index scan or sequential scan."""
    if f"Index Scan using {INDEX}" in plan:
        return "hnsw"
    if "Seq Scan on chunks" in plan:
        return "seq scan"
    return "other"


def short_plan(plan: str) -> str:
    return re.sub(r"'\[[^']*\]'::vector", "'[...]'::vector", plan)


def timed_hnsw(conn, vecs: list, ef: int, warmup: int,
               force_index: bool = False) -> tuple[list[list[int]], list[float], str]:
    """Run all queries at one ef_search inside a rolled-back transaction.

    pgvector's cost estimate grows with ef_search, so at default settings the planner may
    switch to a sequential scan; the plan's scan node is returned. force_index disables
    sequential scans so the HNSW index itself is measured."""
    found, ms = [], []
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(ef)])
        if force_index:
            conn.execute("SET LOCAL enable_seqscan = off")
        node = scan_node(_plan(conn, vecs[0]))
        for v in vecs[:warmup]:
            conn.execute(EXACT_SQL, {"v": v, "k": K}).fetchall()
        for v in vecs:
            t0 = time.perf_counter()
            rows = conn.execute(EXACT_SQL, {"v": v, "k": K}).fetchall()
            ms.append((time.perf_counter() - t0) * 1000)
            found.append([r[0] for r in rows])
        raise psycopg.Rollback()
    return found, ms, node


def run_bench(conn, texts: list[str], vecs: list[list[float]], ef_grid=EF_GRID,
              warmup: int = WARMUP, assert_default_plan: bool = True, log=print) -> dict:
    """HNSW versus exact kNN on one connection. Returns the result dict (no file output)."""
    # exact reference: chunk level (k = 10) and app level (best chunk per RFQ, from 4 * depth chunks)
    exact_ids, exact_ms, exact_rfq = [], [], []
    plan_exact = ""
    for v in vecs:
        t0 = time.perf_counter()
        rows, plan_exact = exact_topk(conn, v, K)
        exact_ms.append((time.perf_counter() - t0) * 1000)
        exact_ids.append([r[0] for r in rows])
        rows40, _ = exact_topk(conn, v, APP_DEPTH * 4)
        exact_rfq.append(collapse_to_rfq([r[1] for r in rows40])[:APP_DEPTH])
    log(f"exact done, mean {sum(exact_ms) / len(exact_ms):.1f} ms")

    # planner check at default settings, no forcing
    plan_default = _plan(conn, vecs[0])
    conn.rollback()
    uses_index = f"Index Scan using {INDEX}" in plan_default
    if assert_default_plan and not uses_index:
        raise AssertionError("default-settings plan does not use the HNSW index:\n" + plan_default)

    rows_ef = []
    for force in (False, True):
        for ef in ef_grid:
            found, ms, node = timed_hnsw(conn, vecs, ef, warmup, force_index=force)
            rec = [recall_at_k(f, t, K) for f, t in zip(found, exact_ids)]
            rows_ef.append({"ef_search": ef, "planner": "seqscan off" if force else "default",
                            "scan": node, "recall_at_10": sum(rec) / len(rec), **latency_stats(ms)})
            log(f"ef={ef} {rows_ef[-1]['planner']} ({node}): recall@10 "
                f"{rows_ef[-1]['recall_at_10']:.4f}, mean {rows_ef[-1]['mean_ms']:.2f} ms")

    # The shipped /search path uses depth 50, so dense_candidates sets ef_search = 200.
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(SHIPPED_DEPTH * 4)])
        plan_shipped = _plan(conn, vecs[0], k=SHIPPED_DEPTH * 4)
        raise psycopg.Rollback()

    emb = PrecomputedEmbedder(dict(zip(texts, vecs)))
    for t in texts[:warmup]:
        dense_candidates(conn, emb, t, depth=APP_DEPTH)
    app_ms, app_rec = [], []
    for t, truth in zip(texts, exact_rfq):
        t0 = time.perf_counter()
        got = dense_candidates(conn, emb, t, depth=APP_DEPTH)
        app_ms.append((time.perf_counter() - t0) * 1000)
        app_rec.append(recall_at_k([g[0] for g in got], truth, K))
    conn.rollback()
    return {
        "exact": latency_stats(exact_ms),
        "plan_exact": plan_exact,
        "plan_default": plan_default,
        "default_plan_uses_hnsw": uses_index,
        "shipped_depth": SHIPPED_DEPTH, "plan_shipped_depth": plan_shipped,
        "shipped_depth_scan": scan_node(plan_shipped),
        "ef_rows": rows_ef,
        "app_level": {
            "depth": APP_DEPTH, "ef_search_set_by_code": APP_DEPTH * 4,
            "recall_at_10": sum(app_rec) / len(app_rec), **latency_stats(app_ms),
        },
    }


# ---------------------------------------------------------------- report
def mb(b: int) -> str:
    return f"{b / 1024 / 1024:.1f}"


def render_md(scale: dict, estimate: dict | None, ingest: dict | None) -> str:
    L = ["# Scale test: HNSW versus exact kNN", ""]
    L.append(f"Commit `{scale['commit']}`, {scale['timestamp']}. Protocol: "
             "fixed before the run, summarised in the README scale-test section.")
    L += ["", "## Corpus", ""]
    c = scale["corpus"]
    L.append(f"- RFQs: {c['rfqs']}; chunks: {c['chunks']}; RFQs with more than one chunk: "
             f"{c['multi_chunk_rfqs']}; max chunks per RFQ: {c['max_chunks_per_rfq']}")
    if estimate:
        L.append(f"- Embedding throughput (E5, CPU, sample of {estimate['sample_chunks']} chunks): "
                 f"{estimate['chunks_per_s']:.1f} chunks/s; estimated full embed "
                 f"{estimate['estimated_minutes']:.1f} min")
    if ingest:
        L += ["", "## Ingest", "",
              f"- Embed and insert wall time: {ingest['load_seconds_total'] / 60:.1f} min ({ingest.get('load_seconds_source', 'timed by this script')})",
              f"- HNSW build: {ingest['index_build_seconds'] / 60:.2f} min "
              f"(m = 16, ef_construction = 64, maintenance_work_mem = {ingest['maintenance_work_mem']})",
              f"- VACUUM ANALYZE: {ingest['vacuum_seconds']:.1f} s", "",
              "| object | relation MB | total MB |", "|---|---|---|"]
        s = ingest["sizes"]
        for rel in ("chunks", INDEX, "chunks_tsv_gin"):
            L.append(f"| {rel} | {mb(s[rel]['relation_bytes'])} | {mb(s[rel]['total_bytes'])} |")
        L += ["", f"Database size: {mb(s['database_bytes'])} MB."]
    cfg = scale["config"]
    L += ["", "## HNSW versus exact kNN", "",
          f"{cfg['n_queries']} queries (seed {cfg['query_seed']}), recall@10 against the exact "
          f"top 10 chunk ids; latency is SQL time after {cfg['warmup']} warm-up queries.", "",
          "`planner = default` is what the application gets; the scan column is the plan the "
          "planner chose. `seqscan off` forces the HNSW index, to measure the index itself.", "",
          "| ef_search | planner | scan | recall@10 | mean ms | p50 ms | p95 ms |",
          "|---|---|---|---|---|---|---|"]
    for r in scale["ef_rows"]:
        L.append(f"| {r['ef_search']} | {r.get('planner', 'default')} | {r.get('scan', '?')} | "
                 f"{r['recall_at_10']:.4f} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
    e = scale["exact"]
    L.append(f"| - | exact | seq scan | 1.0000 | {e['mean_ms']:.2f} | {e['p50_ms']:.2f} | {e['p95_ms']:.2f} |")
    a = scale["app_level"]
    L += ["", "App-level check: `dense_candidates(depth=10)` (shipped code path, "
          f"ef_search {a['ef_search_set_by_code']}) versus the exact RFQ-level top 10:", "",
          "| path | RFQ recall@10 | mean ms | p50 ms | p95 ms |", "|---|---|---|---|---|",
          f"| dense_candidates | {a['recall_at_10']:.4f} | {a['mean_ms']:.2f} | "
          f"{a['p50_ms']:.2f} | {a['p95_ms']:.2f} |", "",
          "Planner at default settings:", "", "```"]
    L += short_plan(scale["plan_default"]).splitlines()[:3]
    L += ["```", ""]
    if "plan_shipped_depth" in scale:
        L += [f"Shipped `/search` path (depth {scale['shipped_depth']}, so `ef_search` "
              f"{scale['shipped_depth'] * 4} and LIMIT {scale['shipped_depth'] * 4}) at default "
              f"settings: **{scale['shipped_depth_scan']}**.", "", "```"]
        L += short_plan(scale["plan_shipped_depth"]).splitlines()[:3]
        L += ["```", ""]
    L += [ "Latency is single-client, warm cache, on one laptop CPU; indicative only."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- commands
def cmd_estimate(args) -> None:
    from app.embeddings import E5Embedder

    rfqs = generate_scale_rfqs(n_batches=N_BATCHES)
    stats = chunk_stats(rfqs)
    print(f"RFQs {stats['rfqs']}, chunks {stats['chunks']}, multi-chunk RFQs "
          f"{stats['multi_chunk_rfqs']}, max chunks per RFQ {stats['max_chunks_per_rfq']}")
    texts, multi = corpus_chunks(rfqs)
    idx = stratified_sample(multi, SAMPLE_N, SAMPLE_SEED)
    sample = [texts[i] for i in idx]
    t0 = time.perf_counter()
    emb = E5Embedder(get_settings().embedding_model)
    load_s = time.perf_counter() - t0
    emb.embed_passages(sample[:BATCH])  # untimed warm-up
    t0 = time.perf_counter()
    for i in range(0, len(sample), BATCH):
        emb.embed_passages(sample[i:i + BATCH])
    el = time.perf_counter() - t0
    cps = len(sample) / el
    minutes = stats["chunks"] / cps / 60
    out = {
        "corpus": stats, "sample_chunks": len(sample),
        "sample_multi_chunks": sum(1 for i in idx if multi[i]),
        "batch_size": BATCH, "model_load_seconds": load_s, "embed_seconds": el,
        "chunks_per_s": cps, "estimated_minutes": minutes, "commit": git_commit(),
        "timestamp": now_iso(),
    }
    write_json("scale_estimate.json", out, Path(args.out_dir))
    print(f"chunks/s {cps:.1f} (model load {load_s:.1f} s)")
    print(f"ESTIMATE_MINUTES={minutes:.1f}")


def cmd_ingest(args) -> None:
    url = args.db_url
    ensure_db(url)
    rfqs = generate_scale_rfqs(n_batches=N_BATCHES)
    stats = chunk_stats(rfqs)
    conn = open_scale_conn(url)
    out_dir = Path(args.out_dir)
    prev = read_json("scale_ingest.json", out_dir) or {}
    present = conn.execute("SELECT count(*) FROM rfqs").fetchone()[0]
    conn.commit()
    all_present = present == len(rfqs)
    if all_present and index_exists(conn) and not args.rebuild_index:
        print("all RFQs present and index exists; nothing to do (use --rebuild-index)")
        return
    load = {"rfqs_present_before": present, "rfqs_loaded": 0, "chunks_loaded": 0, "load_seconds": 0.0}
    if not all_present:
        from app.embeddings import E5Embedder

        if index_exists(conn):  # resumed run with a leftover index: drop it, load, rebuild
            conn.execute(f"DROP INDEX {INDEX}")
            conn.commit()
        print(f"loading {len(rfqs) - present} RFQs ({present} already present)")
        embedder = E5Embedder(get_settings().embedding_model)
        load = ingest_corpus(conn, rfqs, embedder)
    prior = prev.get("load_seconds_total", 0.0) if present else 0.0
    # Persist the load timing before the index build, so a failed build does not lose it.
    write_json("scale_ingest.json", {**prev, "this_run": load,
                                     "load_seconds_total": prior + load["load_seconds"],
                                     "partial": True}, out_dir)
    print("building HNSW index")
    build_s = build_index(conn)
    print(f"index built in {build_s / 60:.2f} min")
    conn.close()
    with psycopg.connect(url, autocommit=True) as vc:
        t0 = time.perf_counter()
        vc.execute("VACUUM ANALYZE")
        vac_s = time.perf_counter() - t0
    conn = connect(url)
    counts = conn.execute("SELECT (SELECT count(*) FROM rfqs), (SELECT count(*) FROM chunks)").fetchone()
    sizes = db_sizes(conn, urlsplit(url).path.lstrip("/"))
    out = {
        "corpus": stats, "db_counts": {"rfqs": counts[0], "chunks": counts[1]},
        "this_run": load, "load_seconds_total": prior + load["load_seconds"],
        "load_seconds_source": prev.get("load_seconds_source", "timed by this script"),
        "index_build_seconds": build_s, "maintenance_work_mem": MAINT_MEM,
        "index_params": {"m": 16, "ef_construction": 64}, "vacuum_seconds": vac_s,
        "sizes": sizes, "commit": git_commit(), "timestamp": now_iso(),
    }
    write_json("scale_ingest.json", out, out_dir)
    print(f"done: {counts[0]} RFQs, {counts[1]} chunks")


def cmd_bench(args) -> None:
    from app.embeddings import E5Embedder

    out_dir = Path(args.out_dir)
    queries = generate_queries(seed=QUERY_SEED, n=N_QUERIES)
    texts = [q["text"] for q in queries]
    emb = E5Embedder(get_settings().embedding_model)
    vecs = [emb.embed_query(t) for t in texts]
    conn = connect(args.db_url)
    counts = conn.execute("SELECT (SELECT count(*) FROM rfqs), (SELECT count(*) FROM chunks)").fetchone()
    conn.rollback()
    res = run_bench(conn, texts, vecs)
    ver = conn.execute(
        "SELECT version(), (SELECT extversion FROM pg_extension WHERE extname = 'vector')").fetchone()
    scale = {
        "config": {
            "db": urlsplit(args.db_url).path.lstrip("/"), "n_queries": len(texts),
            "query_seed": QUERY_SEED, "ef_grid": list(EF_GRID), "warmup": WARMUP, "k": K,
            "embedding_model": get_settings().embedding_model, "postgres": ver[0], "pgvector": ver[1],
        },
        "corpus": {**chunk_stats(generate_scale_rfqs(n_batches=N_BATCHES)), "db_rfqs": counts[0], "db_chunks": counts[1]},
        **res, "commit": git_commit(), "timestamp": now_iso(),
    }
    write_json("scale.json", scale, out_dir)
    md = render_md(scale, read_json("scale_estimate.json", out_dir), read_json("scale_ingest.json", out_dir))
    (out_dir / "scale.md").write_text(md, encoding="utf-8")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-url", default=swap_db_url(get_settings().database_url, SCALE_DB))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    p = sub.add_parser("ingest")
    p.add_argument("--rebuild-index", action="store_true")
    sub.add_parser("bench")
    args = ap.parse_args()
    {"estimate": cmd_estimate, "ingest": cmd_ingest, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
