# Scale test: HNSW versus exact kNN

Commit `293234e`, 2026-11-26T17:05:17+00:00. Protocol: fixed before the run, summarised in the README scale-test section.

## Corpus

- RFQs: 13200; chunks: 15189; RFQs with more than one chunk: 1989; max chunks per RFQ: 2
- Embedding throughput (E5, CPU, sample of 256 chunks): 4.2 chunks/s; estimated full embed 60.4 min

## Ingest

- Embed and insert wall time: 56.6 min (derived, not timed: from the end of the estimate run (2026-11-26T16:03:40Z) to the index-build error in the Postgres log (2026-11-26T17:00:14Z); includes container start, corpus generation and E5 model load (about 10-20 s); the first build failed because /dev/shm was 64 MB)
- HNSW build: 0.05 min (m = 16, ef_construction = 64, maintenance_work_mem = 1GB)
- VACUUM ANALYZE: 0.6 s

| object | relation MB | total MB |
|---|---|---|
| chunks | 19.7 | 151.4 |
| chunks_embedding_hnsw | 59.3 | 59.3 |
| chunks_tsv_gin | 7.3 | 7.4 |

Database size: 162.3 MB.

## HNSW versus exact kNN

200 queries (seed 19), recall@10 against the exact top 10 chunk ids; latency is SQL time after 10 warm-up queries.

`planner = default` is what the application gets; the scan column is the plan the planner chose. `seqscan off` forces the HNSW index, to measure the index itself.

| ef_search | planner | scan | recall@10 | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| 10 | default | hnsw | 0.7860 | 2.44 | 2.37 | 3.48 |
| 20 | default | hnsw | 0.8795 | 2.02 | 2.09 | 2.56 |
| 40 | default | hnsw | 0.9410 | 2.17 | 2.21 | 2.74 |
| 80 | default | hnsw | 0.9720 | 2.68 | 2.56 | 3.74 |
| 200 | default | seq scan | 1.0000 | 39.38 | 38.35 | 45.86 |
| 400 | default | seq scan | 1.0000 | 40.29 | 39.08 | 46.82 |
| 10 | seqscan off | hnsw | 0.7860 | 2.04 | 2.12 | 2.57 |
| 20 | seqscan off | hnsw | 0.8795 | 2.04 | 2.13 | 2.58 |
| 40 | seqscan off | hnsw | 0.9410 | 2.20 | 2.23 | 2.79 |
| 80 | seqscan off | hnsw | 0.9720 | 2.56 | 2.50 | 3.42 |
| 200 | seqscan off | hnsw | 0.9990 | 3.29 | 3.23 | 4.22 |
| 400 | seqscan off | hnsw | 0.9995 | 4.57 | 4.33 | 6.69 |
| - | exact | seq scan | 1.0000 | 42.97 | 41.28 | 49.86 |

App-level check: `dense_candidates(depth=10)` (shipped code path, ef_search 40) versus the exact RFQ-level top 10:

| path | RFQ recall@10 | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|
| dense_candidates | 0.9385 | 3.36 | 3.25 | 4.40 |

Planner at default settings:

```
Limit  (cost=1143.29..1169.37 rows=10 width=23)
  ->  Index Scan using chunks_embedding_hnsw on chunks  (cost=1143.29..40751.78 rows=15189 width=23)
        Order By: (embedding <=> '[...]'::vector)
```

Shipped `/search` path (depth 50, so `ef_search` 200 and LIMIT 200) at default settings: **seq scan**.

```
Limit  (cost=3362.32..3362.82 rows=200 width=23)
  ->  Sort  (cost=3362.32..3400.29 rows=15189 width=23)
        Sort Key: ((embedding <=> '[...]'::vector))
```

Latency is single-client, warm cache, on one laptop CPU; indicative only.
