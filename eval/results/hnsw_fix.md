# HNSW planner fix: acceptance checks

Commit `8a336bd`, 2026-11-26T18:25:49+00:00. Protocol fixed before the run. Change: `SET LOCAL enable_seqscan = off` inside the rolled-back transaction of `dense_candidates`, so the planner keeps the HNSW index when one exists.

## Check 1: small corpus unchanged

`dense_candidates(depth=50)` versus exact kNN (exact top 200 chunks, best chunk per RFQ, first 50), 400 RFQs: same RFQ ids in the same order for **140 / 140** queries (80 family + 60 code).

Scan node with the fix settings: hnsw.

Result: **PASS** (criterion: all 140 identical).

## Check 2: large corpus uses the index

200 queries (seed 29), 13200 RFQs. Reference: exact RFQ-level top 50. Latency is SQL time after 10 warm-up queries, single client, indicative only.

| path | scan | RFQ recall@50 | RFQ recall@10 | mean ms | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| before | seq scan | 1.0000 | 1.0000 | 44.97 | 43.98 | 52.22 |
| after | hnsw | 0.9926 | 0.9960 | 6.17 | 5.85 | 8.49 |

`before`: ef_search 200, default planner. `after`: shipped `dense_candidates`.

Plan after the change:

```
Limit  (cost=3993.33..4477.34 rows=200 width=23)
  ->  Index Scan using chunks_embedding_hnsw on chunks  (cost=3993.33..40751.78 rows=15189 width=23)
        Order By: (embedding <=> '[...]'::vector)
```

Result: **PASS** (criteria: plan after is the HNSW index scan; mean recall@50 >= 0.98).
