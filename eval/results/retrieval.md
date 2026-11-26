Mean relevant RFQs per query: 10.00

Overall (n = 80)

| mode | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.725 | 0.358 | 0.255 | 0.638 |
| lexical | 0.487 | 0.185 | 0.148 | 0.361 |
| hybrid | 0.725 | 0.358 | 0.255 | 0.638 |

Same-language queries (n = 35)

| mode | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.829 | 0.531 | 0.400 | 0.737 |
| lexical | 0.686 | 0.274 | 0.232 | 0.505 |
| hybrid | 0.829 | 0.531 | 0.400 | 0.737 |

Cross-language queries (n = 45)

| mode | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.644 | 0.222 | 0.143 | 0.561 |
| lexical | 0.333 | 0.116 | 0.084 | 0.248 |
| hybrid | 0.644 | 0.222 | 0.143 | 0.561 |

Held-out test split (families from split_families, seed 11) (n = 40)

| mode | hit@5 | P@5 | R@10 | MRR@20 |
|---|---|---|---|---|
| dense | 0.725 | 0.385 | 0.246 | 0.689 |
| lexical | 0.475 | 0.195 | 0.150 | 0.379 |
| hybrid | 0.725 | 0.385 | 0.246 | 0.689 |

Mean query latency per mode (ms): dense 54.7, lexical 6.5, hybrid 53.8
