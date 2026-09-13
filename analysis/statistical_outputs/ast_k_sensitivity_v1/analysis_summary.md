# AST retrieval K-sensitivity analysis

## Scope and configuration

This is a sensitivity/robustness experiment over K={5,10,15,20}; K=20 remains the primary paper depth. No K was selected after observing the results. The benchmark, AST graph, Qdrant collection, embeddings, reranker, graph expansion, deduplication, thresholds, and corrected rank-at-K accounting were held fixed.

The existing implementation would accept `dynamic_k_min=10` at K=5 but would force the first staged pool to at least 10 candidates, exceeding the requested final depth. The repository's original paper sensitivity protocol explicitly fixes `dynamic_k_min=5` across all four K values; that recovered protocol was used. Consequently, the K=20 sensitivity staged row (`min=5`) is supplementary and does not replace the authoritative primary K=20 staged row (`min=10`).

## Query-weighted results

| K | Method | Hits@K | Recall@K | MRR@K | Pair savings | CE-time savings |
|---:|---|---:|---:|---:|---:|---:|
| 5 | Vector-only | 271/315 | 0.860317 | 0.741693 | 0.0% | 0.0% |
| 5 | Vector + Cross-Encoder | 271/315 | 0.860317 | 0.708466 | 0.0% | 0.0% |
| 5 | AST-KG + Cross-Encoder static-K | 263/315 | 0.834921 | 0.687460 | 0.0% | 0.0% |
| 5 | AST-KG + Staged Confidence | 264/315 | 0.838095 | 0.694180 | 59.0% | 58.3% |
| 10 | Vector-only | 286/315 | 0.907937 | 0.747750 | 0.0% | 0.0% |
| 10 | Vector + Cross-Encoder | 286/315 | 0.907937 | 0.705411 | 0.0% | 0.0% |
| 10 | AST-KG + Cross-Encoder static-K | 284/315 | 0.901587 | 0.696544 | 0.0% | 0.0% |
| 10 | AST-KG + Staged Confidence | 286/315 | 0.907937 | 0.703534 | 62.0% | 62.7% |
| 15 | Vector-only | 296/315 | 0.939683 | 0.750293 | 0.0% | 0.0% |
| 15 | Vector + Cross-Encoder | 296/315 | 0.939683 | 0.702621 | 0.0% | 0.0% |
| 15 | AST-KG + Cross-Encoder static-K | 291/315 | 0.923810 | 0.701704 | 0.0% | 0.0% |
| 15 | AST-KG + Staged Confidence | 294/315 | 0.933333 | 0.701231 | 58.7% | 56.3% |
| 20 | Vector-only | 301/315 | 0.955556 | 0.751194 | 0.0% | 0.0% |
| 20 | Vector + Cross-Encoder | 301/315 | 0.955556 | 0.697488 | 0.0% | 0.0% |
| 20 | AST-KG + Cross-Encoder static-K | 298/315 | 0.946032 | 0.704563 | 0.0% | 0.0% |
| 20 | AST-KG + Staged Confidence | 301/315 | 0.955556 | 0.697188 | 51.8% | 53.4% |

Savings are staged-confidence savings relative to static KG at the same K. Zeros for other methods mean not applicable/no staged saving. Absolute CE time was not retained by the existing ladder's output schema; the measured same-run CE-time savings were retained.

## Target-balanced results

| K | Method | Query recall | Target-balanced recall | Query MRR | Target-balanced MRR |
|---:|---|---:|---:|---:|---:|
| 5 | Vector-only | 0.860317 | 0.867661 | 0.741693 | 0.748929 |
| 5 | Vector + Cross-Encoder | 0.860317 | 0.867661 | 0.708466 | 0.714498 |
| 5 | AST-KG + Cross-Encoder static-K | 0.834921 | 0.842811 | 0.687460 | 0.693296 |
| 5 | AST-KG + Staged Confidence | 0.838095 | 0.844198 | 0.694180 | 0.699530 |
| 10 | Vector-only | 0.907937 | 0.911697 | 0.747750 | 0.754455 |
| 10 | Vector + Cross-Encoder | 0.907937 | 0.911697 | 0.705411 | 0.711076 |
| 10 | AST-KG + Cross-Encoder static-K | 0.901587 | 0.910772 | 0.696544 | 0.702692 |
| 10 | AST-KG + Staged Confidence | 0.907937 | 0.913546 | 0.703534 | 0.709347 |
| 15 | Vector-only | 0.939683 | 0.942210 | 0.750293 | 0.756879 |
| 15 | Vector + Cross-Encoder | 0.939683 | 0.942210 | 0.702621 | 0.708335 |
| 15 | AST-KG + Cross-Encoder static-K | 0.923810 | 0.933426 | 0.701704 | 0.707952 |
| 15 | AST-KG + Staged Confidence | 0.933333 | 0.937587 | 0.701231 | 0.707060 |
| 20 | Vector-only | 0.955556 | 0.954693 | 0.751194 | 0.757590 |
| 20 | Vector + Cross-Encoder | 0.955556 | 0.954693 | 0.697488 | 0.703015 |
| 20 | AST-KG + Cross-Encoder static-K | 0.946032 | 0.954230 | 0.704563 | 0.710558 |
| 20 | AST-KG + Staged Confidence | 0.955556 | 0.954693 | 0.697188 | 0.702710 |

## Paired Recall@K comparisons

| K | KG method vs vector | Both hit | Vector only | KG only | Both miss | KG-vector (pp) | Exact p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | AST-KG + Cross-Encoder static-K | 253 | 18 | 10 | 34 | -2.540 | 0.184933 |
| 5 | AST-KG + Staged Confidence | 258 | 13 | 6 | 38 | -2.222 | 0.167068 |
| 10 | AST-KG + Cross-Encoder static-K | 275 | 11 | 9 | 20 | -0.635 | 0.823803 |
| 10 | AST-KG + Staged Confidence | 282 | 4 | 4 | 25 | +0.000 | 1 |
| 15 | AST-KG + Cross-Encoder static-K | 286 | 10 | 5 | 14 | -1.587 | 0.301758 |
| 15 | AST-KG + Staged Confidence | 293 | 3 | 1 | 18 | -0.635 | 0.625 |
| 20 | AST-KG + Cross-Encoder static-K | 294 | 7 | 4 | 10 | -0.952 | 0.548828 |
| 20 | AST-KG + Staged Confidence | 300 | 1 | 1 | 13 | +0.000 | 1 |

## K=20 primary cross-check

- Vector-only: sensitivity Recall@20=0.955556, MRR@20=0.751194; primary Recall@20=0.955556, MRR@20=0.751194.
- Vector + Cross-Encoder: sensitivity Recall@20=0.955556, MRR@20=0.697488; primary Recall@20=0.955556, MRR@20=0.697488.
- AST-KG + Cross-Encoder static-K: sensitivity Recall@20=0.946032, MRR@20=0.704563; primary Recall@20=0.946032, MRR@20=0.704563.
- AST-KG + Staged Confidence: sensitivity Recall@20=0.955556, MRR@20=0.697188; primary Recall@20=0.955556, MRR@20=0.697188.

Vector-only, Vector+CE, and static KG reproduced the authoritative K=20 per-query gold ranks exactly. The staged sensitivity configuration changed the gold rank on 0/315 queries relative to the primary `min=10` staged run; the authoritative primary artifact remains unchanged.

## Conclusions

1. **Does AST-KG significantly improve Recall at any tested K?** No. Neither static nor staged AST-KG has a significant positive McNemar result at any tested K.

2. **Is any apparent KG advantage consistent across multiple K values?** No. Static KG is below vector recall at every K, while staged is below vector at K=5 and K=15 and tied at K=10 and K=20. There is no repeated positive recall advantage over vector-only.

3. **Does vector-only remain stronger on MRR?** Yes. Vector-only has the highest MRR at every tested K.

4. **Does staged confidence preserve retrieval quality at lower K?** Relative to static KG, yes descriptively: staged recall equals or exceeds static recall at every tested K, while saving substantial CE work. This does not imply improvement over vector-only.

5. **Are the K=20 conclusions robust to retrieval depth?** Yes. Across all tested depths, vector-only keeps the strongest MRR and neither AST-KG method shows a statistically reliable recall improvement over vector-only. K=20 remains the primary configuration; no alternative K is recommended from this sensitivity analysis.

K=40 was not run. It was optional, is unnecessary for the requested main sweep, and would coincide with the staged `dynamic_k_max=40` ceiling.
