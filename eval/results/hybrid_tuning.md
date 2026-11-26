Protocol (split and grid defined after a first full-set run; fixed before the tuning run and not changed after seeing its results)
- Split: by family. Sorted unique family_ids of eval/queries.jsonl, shuffled with random.Random(11); the first half is dev, the rest is test. Both queries of a family land in the same split.
- Fusion: weighted RRF, score(d) = w_dense/(k + rank_dense(d)) + w_lex/(k + rank_lex(d)), with w_dense = 1.0. Candidate depth per list: 50.
- Grid: w_lex in {0.0, 0.1, 0.25, 0.5, 0.75, 1.0} x k in {10, 60} (12 configs).
- Selection metric: dev MRR@20. Tie-break: higher dev hit@5, then smaller w_lex, then k = 60.
- Compared on dev and test: dense, lexical, hybrid (w_lex = 1, k = 60, the previous default) and hybrid (selected config). Test numbers are reported whatever they show. No second selection round.
- The selected config becomes the default in Settings, even if its w_lex is 0.0 (which would mean lexical adds nothing on this data).
- The split and grid were defined after a first full-set run had shown equal-weight hybrid underperforming. Test families were held out from the selection step, not from all observation.

Dev: 20 families, 40 queries. Test: 20 families, 40 queries.

Grid on dev (12 configs)

| w_lex | k | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|---|
| 0.0 | 10 | 0.725 | 0.330 | 0.264 | 0.587 |
| 0.0 | 60 | 0.725 | 0.330 | 0.264 | 0.587 |
| 0.1 | 10 | 0.725 | 0.335 | 0.264 | 0.577 |
| 0.1 | 60 | 0.675 | 0.320 | 0.253 | 0.557 |
| 0.25 | 10 | 0.725 | 0.335 | 0.257 | 0.554 |
| 0.25 | 60 | 0.650 | 0.310 | 0.241 | 0.563 |
| 0.5 | 10 | 0.700 | 0.320 | 0.251 | 0.562 |
| 0.5 | 60 | 0.625 | 0.290 | 0.216 | 0.519 |
| 0.75 | 10 | 0.650 | 0.290 | 0.235 | 0.563 |
| 0.75 | 60 | 0.575 | 0.275 | 0.206 | 0.527 |
| 1.0 | 10 | 0.575 | 0.260 | 0.205 | 0.558 |
| 1.0 | 60 | 0.550 | 0.250 | 0.201 | 0.532 |

Selected: w_lex = 0.0, k = 60 (dev MRR@20 0.587).

Dev split (for transparency)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.725 | 0.330 | 0.264 | 0.587 |
| lexical | 0.500 | 0.175 | 0.147 | 0.343 |
| hybrid (w_lex=1.0, k=60, previous default) | 0.550 | 0.250 | 0.201 | 0.532 |
| hybrid selected (w_lex=0.0, k=60) | 0.725 | 0.330 | 0.264 | 0.587 |

Test split (held out)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.725 | 0.385 | 0.246 | 0.689 |
| lexical | 0.475 | 0.195 | 0.150 | 0.379 |
| hybrid (w_lex=1.0, k=60, previous default) | 0.675 | 0.270 | 0.213 | 0.546 |
| hybrid selected (w_lex=0.0, k=60) | 0.725 | 0.385 | 0.246 | 0.689 |

Selected w_lex is 0.0: lexical adds nothing on this data.
