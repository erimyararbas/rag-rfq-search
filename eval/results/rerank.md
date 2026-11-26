Protocol (fixed before any rerank run; not changed after seeing results)
- Queries and corpus: the existing 80 queries (seed 7) and 400 RFQs (seed 42), unchanged.
- Split: by family, split_families with seed 11 (the same dev and test families as the hybrid sweep).
- First stage: the shipped default, hybrid with the settings defaults (w_lex = 0.0, k = 60, so it equals dense), top 50.
- Reranker input: query text and the best chunk text of each candidate, as stored in the DB. No instruction prefix. Model BAAI/bge-reranker-v2-m3, max_length 512, CPU.
- Depth grid: {20, 30, 50}. For each depth d the reranker actually scores the first d candidates (a separate, timed call per depth). The reranked top d forms the final ranking; metrics are computed on its top 20 (for d = 20 rerank only reorders the top 20).
- Selection: dev MRR@20. Tie-break: higher dev hit@5, then smaller depth.
- Reported: baseline and every depth on dev and on test; the full set (80) with same-language and cross-language subsets, labelled as including the dev half. Test numbers are reported whatever they show. No second selection round.
- Latency: one untimed warm-up call, then per query the wall time of the reranker call per depth (mean, p50, p95), plus first-stage latency. Model load time and torch.get_num_threads() are recorded. CPU only.
- Product decision fixed in advance: rerank stays opt-in (rerank=true) whatever the result, because of its CPU latency. RERANK_DEPTH is set to the selected depth.

Dev: 20 families, 40 queries. Test: 20 families, 40 queries.

Selected depth: 50 (dev MRR@20 0.706, dev hit@5 0.825).

Dev split (selection)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| baseline (first stage) | 0.725 | 0.330 | 0.264 | 0.587 |
| rerank depth 20 | 0.775 | 0.375 | 0.276 | 0.685 |
| rerank depth 30 | 0.775 | 0.400 | 0.285 | 0.699 |
| rerank depth 50 | 0.825 | 0.410 | 0.303 | 0.706 |

Test split (held out)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| baseline (first stage) | 0.725 | 0.385 | 0.246 | 0.689 |
| rerank depth 20 | 0.775 | 0.440 | 0.285 | 0.711 |
| rerank depth 30 | 0.775 | 0.460 | 0.305 | 0.707 |
| rerank depth 50 | 0.775 | 0.490 | 0.337 | 0.706 |

Full set (n = 80; includes the dev half that selected the depth)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| baseline (first stage) | 0.725 | 0.358 | 0.255 | 0.638 |
| rerank depth 20 | 0.775 | 0.408 | 0.280 | 0.698 |
| rerank depth 30 | 0.775 | 0.430 | 0.295 | 0.703 |
| rerank depth 50 | 0.800 | 0.450 | 0.320 | 0.706 |

Full set, same-language queries (includes the dev half)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| baseline (first stage) | 0.829 | 0.531 | 0.400 | 0.737 |
| rerank depth 20 | 0.800 | 0.560 | 0.430 | 0.774 |
| rerank depth 30 | 0.829 | 0.606 | 0.442 | 0.782 |
| rerank depth 50 | 0.914 | 0.640 | 0.462 | 0.825 |

Full set, cross-language queries (n = 45; includes the dev half)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| baseline (first stage) | 0.644 | 0.222 | 0.143 | 0.561 |
| rerank depth 20 | 0.756 | 0.289 | 0.164 | 0.639 |
| rerank depth 30 | 0.733 | 0.293 | 0.181 | 0.642 |
| rerank depth 50 | 0.711 | 0.302 | 0.209 | 0.614 |

Latency (per query; reranker on CPU)

| stage | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| first stage (hybrid, top 50) | 68.9 | 57.2 | 132.9 |
| rerank depth 20 | 10600.7 | 10510.2 | 12942.5 |
| rerank depth 30 | 14738.3 | 14786.6 | 17415.7 |
| rerank depth 50 | 22711.5 | 22081.7 | 26344.1 |

Reranker load time: 8.6 s. torch threads: 8.
