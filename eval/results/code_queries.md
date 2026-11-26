Protocol (fixed before any run on this query set; not changed after seeing results)
- Question: does lexical retrieval or hybrid fusion help when the query contains exact material codes or standard numbers?
- Codes: material grades (material field) and standard designations that appear in the RFQ text. Thread sizes and tolerances are excluded.
- Relevance: an RFQ is relevant to a query if its text (title, description, technical notes) contains every code of the query as an exact, case-insensitive match with token boundaries (code not preceded by a word character, '.' or '-'; not followed by a word character, '-', or '.' plus a digit). Computed from data/rfqs.jsonl at generation time and stored in the query file; family_id is not used.
- Query set: eval/code_queries.jsonl, generator seed 23. 60 queries, 20 each of single (one bare code), pair (material + standard, bare), sentence (the pair in a request sentence; 8 en, 6 de, 6 tr). Kept only if 1 to 30 relevant RFQs and a new code tuple within its type.
- Systems (nothing tuned on this set): dense; lexical; hybrid with equal weights (w_lex = 1.0, k = 60); shipped default (hybrid with the settings defaults, w_lex = 0.0, equal to dense); rerank = shipped default top 50 reranked with BAAI/bge-reranker-v2-m3 (max_length 512, CPU) at the depth already selected on the family dev split (50).
- Metrics: hit@5, P@5, R@10, MRR@20 (top 20 retrieved), mean n_relevant. Every query is reported; there is no split.
- Latency: mean per system; rerank = first stage plus the reranker call (timed after one untimed warm-up).
- Results are reported whatever they show. README wording about exact codes is updated from these numbers only.

All queries (n = 60; mean n_relevant 5.08)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.650 | 0.303 | 0.529 | 0.498 |
| lexical | 0.867 | 0.537 | 0.767 | 0.815 |
| hybrid (equal weights) | 0.867 | 0.480 | 0.780 | 0.807 |
| shipped default | 0.650 | 0.303 | 0.529 | 0.498 |
| rerank (default top 50) | 0.967 | 0.490 | 0.738 | 0.946 |

Type: single (n = 20; mean n_relevant 10.20)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.600 | 0.510 | 0.409 | 0.585 |
| lexical | 1.000 | 0.970 | 0.862 | 1.000 |
| hybrid (equal weights) | 1.000 | 0.810 | 0.779 | 1.000 |
| shipped default | 0.600 | 0.510 | 0.409 | 0.585 |
| rerank (default top 50) | 1.000 | 0.720 | 0.553 | 1.000 |

Type: pair (n = 20; mean n_relevant 2.35)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.700 | 0.240 | 0.733 | 0.509 |
| lexical | 1.000 | 0.460 | 1.000 | 1.000 |
| hybrid (equal weights) | 1.000 | 0.450 | 1.000 | 1.000 |
| shipped default | 0.700 | 0.240 | 0.733 | 0.509 |
| rerank (default top 50) | 1.000 | 0.420 | 0.938 | 1.000 |

Type: sentence (n = 20; mean n_relevant 2.70)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.650 | 0.160 | 0.443 | 0.399 |
| lexical | 0.600 | 0.180 | 0.439 | 0.446 |
| hybrid (equal weights) | 0.600 | 0.180 | 0.560 | 0.422 |
| shipped default | 0.650 | 0.160 | 0.443 | 0.399 |
| rerank (default top 50) | 0.900 | 0.330 | 0.724 | 0.838 |

Sentence queries, language en (n = 8; mean n_relevant 2.50)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.500 | 0.125 | 0.344 | 0.438 |
| lexical | 0.500 | 0.100 | 0.208 | 0.281 |
| hybrid (equal weights) | 0.500 | 0.125 | 0.385 | 0.264 |
| shipped default | 0.500 | 0.125 | 0.344 | 0.438 |
| rerank (default top 50) | 0.750 | 0.200 | 0.490 | 0.656 |

Sentence queries, language de (n = 6; mean n_relevant 2.17)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.833 | 0.233 | 0.694 | 0.528 |
| lexical | 0.500 | 0.200 | 0.444 | 0.444 |
| hybrid (equal weights) | 0.667 | 0.267 | 0.667 | 0.506 |
| shipped default | 0.833 | 0.233 | 0.694 | 0.528 |
| rerank (default top 50) | 1.000 | 0.433 | 1.000 | 1.000 |

Sentence queries, language tr (n = 6; mean n_relevant 3.50)

| system | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.667 | 0.133 | 0.325 | 0.218 |
| lexical | 0.833 | 0.267 | 0.742 | 0.667 |
| hybrid (equal weights) | 0.667 | 0.167 | 0.687 | 0.548 |
| shipped default | 0.667 | 0.133 | 0.325 | 0.218 |
| rerank (default top 50) | 1.000 | 0.400 | 0.762 | 0.917 |

Mean latency per query

| system | mean ms |
|---|---|
| dense | 54.0 |
| lexical | 3.1 |
| hybrid (equal weights) | 49.3 |
| shipped default | 47.3 |
| rerank (default top 50) | 23344.5 |

Reranker call alone (depth 50): mean 23297.1 ms. Reranker load time: 4.3 s. torch threads: 8.
