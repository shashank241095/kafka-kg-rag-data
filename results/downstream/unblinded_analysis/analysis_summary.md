# Unblinded downstream analysis

## Scope and provenance

This report analyzes the frozen 100-query AST downstream experiment. Unblinding passed all structural checks: 300 scored answers, 100 queries, exactly three distinct retrieval conditions per query, no missing mappings, and an identical before/after score-value fingerprint.

The final reported labels were manually assigned by the author under blinded Method A/B/C identities using the fixed rubric; see the 2026-09-11 correction at the top of `AI_SCORING_NOTICE.md`. No score, generation, retrieval result, benchmark row, private mapping, or legacy result was modified by this analysis.

## New AST downstream results

| Method | Root-cause correctness | Evidence grounding | Hallucination rate |
|---|---:|---:|---:|
| Vector-only | 58/100 (58.0%) | 65/100 (65.0%) | 54/100 (54.0%) |
| AST-KG + Cross-Encoder static-K | 57/100 (57.0%) | 69/100 (69.0%) | 57/100 (57.0%) |
| AST-KG + Cross-Encoder + Staged Confidence | 52/100 (52.0%) | 63/100 (63.0%) | 54/100 (54.0%) |

Hallucination = 1 means an unsupported causal claim was introduced; lower is better.

### 95% Wilson confidence intervals

| Method | Root-cause correctness | Evidence grounding | Hallucination |
|---|---:|---:|---:|
| Vector-only | 58.0% [48.2%, 67.2%] | 65.0% [55.3%, 73.6%] | 54.0% [44.3%, 63.4%] |
| AST-KG + Cross-Encoder static-K | 57.0% [47.2%, 66.3%] | 69.0% [59.4%, 77.2%] | 57.0% [47.2%, 66.3%] |
| AST-KG + Cross-Encoder + Staged Confidence | 52.0% [42.3%, 61.5%] | 63.0% [53.2%, 71.8%] | 54.0% [44.3%, 63.4%] |

## Orientation against legacy-parser results

| Method | New AST RCA | Legacy RCA | New AST grounding | Legacy grounding | New AST hallucination | Legacy hallucination |
|---|---:|---:|---:|---:|---:|---:|
| Vector-only | 58.0% | 50.0% | 65.0% | 96.0% | 54.0% | 4.0% |
| AST-KG + Cross-Encoder static-K (legacy: KG + CrossEncoder) | 57.0% | 62.0% | 69.0% | 100.0% | 57.0% | 0.0% |
| AST-KG + Cross-Encoder + Staged Confidence (legacy: KG + StagedConfidence) | 52.0% | 61.0% | 63.0% | 98.0% | 54.0% | 2.0% |

This is not a strict apples-to-apples comparison and changes relative to the legacy values are not statistical evidence. The graph construction changed from regex to AST; the indexed corpus changed; the benchmark changed from 321 to an audited 315 queries; this 100-query sample comes from the revised 103-target benchmark; and the new scoring is blinded.

## Paired Root-cause correctness

| Comparison (A vs B) | Both 1 | n10 (A=1,B=0) | n01 (A=0,B=1) | Both 0 | A-B (pp) | Absolute difference (pp) | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| AST-KG + Cross-Encoder static-K vs Vector-only | 53 | 4 | 5 | 38 | -1.0 | 1.0 | 1 |
| AST-KG + Cross-Encoder + Staged Confidence vs Vector-only | 48 | 4 | 10 | 38 | -6.0 | 6.0 | 0.179565 |
| AST-KG + Cross-Encoder static-K vs AST-KG + Cross-Encoder + Staged Confidence | 52 | 5 | 0 | 43 | +5.0 | 5.0 | 0.0625 |

## Paired Evidence grounding

| Comparison (A vs B) | Both 1 | n10 (A=1,B=0) | n01 (A=0,B=1) | Both 0 | A-B (pp) | Absolute difference (pp) | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| AST-KG + Cross-Encoder static-K vs Vector-only | 59 | 10 | 6 | 25 | +4.0 | 4.0 | 0.454498 |
| AST-KG + Cross-Encoder + Staged Confidence vs Vector-only | 54 | 9 | 11 | 26 | -2.0 | 2.0 | 0.823803 |
| AST-KG + Cross-Encoder static-K vs AST-KG + Cross-Encoder + Staged Confidence | 62 | 7 | 1 | 30 | +6.0 | 6.0 | 0.0703125 |

## Paired Hallucination

| Comparison (A vs B) | Both hallucinate | A only | B only | Neither | A-B (pp) | Absolute difference (pp) | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| AST-KG + Cross-Encoder static-K vs Vector-only | 42 | 15 | 12 | 31 | +3.0 | 3.0 | 0.701108 |
| AST-KG + Cross-Encoder + Staged Confidence vs Vector-only | 43 | 11 | 11 | 35 | +0.0 | 0.0 | 1 |
| AST-KG + Cross-Encoder static-K vs AST-KG + Cross-Encoder + Staged Confidence | 48 | 9 | 6 | 37 | +3.0 | 3.0 | 0.607239 |

For hallucination, a negative A-B difference means A has fewer hallucinations and is better. Static had +3.0 pp versus vector; staged had +0.0 pp versus vector; and static had +3.0 pp versus staged.

## Paired bootstrap 95% confidence intervals

Fixed seed 42, 10,000 query-level paired bootstrap replicates. Values are percentage-point differences.

| Metric | Difference | Point estimate | 95% CI |
|---|---|---:|---:|
| Root-cause correctness | AST-KG + Cross-Encoder static-K - Vector-only | -1.0 | [-7.0, +5.0] |
| Root-cause correctness | AST-KG + Cross-Encoder + Staged Confidence - Vector-only | -6.0 | [-13.0, +1.0] |
| Root-cause correctness | AST-KG + Cross-Encoder + Staged Confidence - AST-KG + Cross-Encoder static-K | -5.0 | [-10.0, -1.0] |
| Evidence grounding | AST-KG + Cross-Encoder static-K - Vector-only | +4.0 | [-4.0, +12.0] |
| Evidence grounding | AST-KG + Cross-Encoder + Staged Confidence - Vector-only | -2.0 | [-11.0, +7.0] |
| Evidence grounding | AST-KG + Cross-Encoder + Staged Confidence - AST-KG + Cross-Encoder static-K | -6.0 | [-12.0, -1.0] |
| Hallucination | AST-KG + Cross-Encoder static-K - Vector-only | +3.0 | [-7.0, +13.0] |
| Hallucination | AST-KG + Cross-Encoder + Staged Confidence - Vector-only | +0.0 | [-9.0, +9.0] |
| Hallucination | AST-KG + Cross-Encoder + Staged Confidence - AST-KG + Cross-Encoder static-K | -3.0 | [-10.0, +5.0] |

## Root-cause correctness query-level changes

- Static KG rescues (4): L014-Q2, L024-Q1, L035-Q2, L061-Q1

- Static KG losses (5): L002-Q1, L050-Q3, L055-Q2, L092-Q2, L093-Q2

- Staged KG rescues (4): L014-Q2, L024-Q1, L035-Q2, L061-Q1

- Staged KG losses (10): L002-Q1, L020-Q2, L037-Q2, L040-Q1, L050-Q3, L055-Q2, L063-Q2, L075-Q3, L092-Q2, L093-Q2

The companion `query_level_comparisons.csv` contains query text, expected target, all three correctness labels, retrieval ranks, top-5/top-20 flags, and all three top-5 evidence lists for every changed query.

## Retrieval rank versus downstream correctness

These are descriptive associations, not causal estimates.

| Method | Gold top 5 | Gold top 20 | RCA correct | Top5+correct | Top5+incorrect | Outside top5+correct | Outside top5+incorrect |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector-only | 86/100 | 96/100 | 58/100 | 54 | 32 | 4 | 10 |
| AST-KG + Cross-Encoder static-K | 85/100 | 97/100 | 57/100 | 53 | 32 | 4 | 11 |
| AST-KG + Cross-Encoder + Staged Confidence | 88/100 | 96/100 | 52/100 | 50 | 38 | 2 | 10 |

| Method | Top20+correct | Top20+incorrect | Outside top20+correct | Outside top20+incorrect |
|---|---:|---:|---:|---:|
| Vector-only | 57 | 39 | 1 | 3 |
| AST-KG + Cross-Encoder static-K | 57 | 40 | 0 | 3 |
| AST-KG + Cross-Encoder + Staged Confidence | 52 | 44 | 0 | 4 |

## Evidence-based conclusions

1. **Does AST-KG static significantly improve root-cause correctness over vector-only?** No statistically significant difference. The observed change was -1.0 pp; exact McNemar p=1.

2. **Does AST-KG staged significantly improve root-cause correctness over vector-only?** No statistically significant difference. The observed change was -6.0 pp; exact McNemar p=0.179565.

3. **Does staged confidence preserve downstream quality relative to static KG?** Staged minus static correctness was -5.0 pp; exact McNemar p=0.0625; the paired bootstrap 95% CI was [-10.0, -1.0] pp. The exact test did not cross 0.05, but the bootstrap interval excludes zero. The evidence is therefore borderline and method-dependent, and preservation cannot be claimed; this was not an equivalence or non-inferiority test.

4. **Does AST-KG improve evidence grounding?** Static: No statistically significant difference. The observed change was +4.0 pp; exact McNemar p=0.454498. Staged: No statistically significant difference. The observed change was -2.0 pp; exact McNemar p=0.823803.

5. **Does AST-KG reduce hallucination?** Static: No statistically significant difference. The observed change was +3.0 pp; exact McNemar p=0.701108. Staged: No statistically significant difference. The observed change was +0.0 pp; exact McNemar p=1.

6. **Given that retrieval Recall@20 did not improve, is there evidence that structural context improves downstream reasoning anyway?** No statistically reliable evidence from root-cause correctness: neither AST-KG condition significantly improved downstream correctness. The contingency tables are descriptive and cannot establish a structural-context effect.

## Reproducibility

- Blinded score SHA-256: `a57fd8e1a7fbf23060b151257c9d3ccea229ba45a7178b7e9ca9c8a81e6cafe6`

- Private mapping SHA-256: `4b09910c284784eb0cea3a80b0e7efd2d9af2b1ed9da559d21b2b77c98acd565`

- Score-value fingerprint before/after: `12bd88dbb02a5403b8f6e8fd938e529857336e338386d2ae57e66d4090f69fc2`

- Exact McNemar tests use a two-sided exact binomial test over discordant pairs.

- Binomial intervals are 95% Wilson intervals.

- Paired bootstrap intervals use seed 42 and 10,000 query-level replicates.

- The exact McNemar test is the primary binary paired test. Percentile-bootstrap intervals need not invert that discrete exact test, which explains the borderline staged-versus-static discrepancy.
