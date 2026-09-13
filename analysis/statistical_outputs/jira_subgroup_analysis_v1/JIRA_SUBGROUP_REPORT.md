# Exploratory JIRA-seeded subgroup analysis (n = 21)

**Status: EXPLORATORY ROBUSTNESS CHECK. Not a primary experiment, not confirmatory, and not
yet reflected in the manuscript.** No retrieval, embedding, OpenIE or LLM call was executed;
every number is recomputed from frozen per-query artifacts. No primary artifact was modified.

## 1. Subset identification

Membership was derived **two independent ways and required to agree**:

1. `benchmark_incident_jira_seed_flat.json` (21 records, `source_type: "jira"`), matched into
   the final benchmark on the `(question, file, line)` signature.
2. The frozen `statistical_analysis/per_query_results.csv` `subset == JIRA` label.

Both yielded the identical 21 query IDs. Count verified: **21**.

### The decisive structural fact

The 21 JIRA queries cover only **5 unique `(file, line)` targets**:

| Queries | Target |
|---|---|
| 7 | `clients/src/main/java/org/apache/kafka/clients/FetchSessionHandler.java:618` |
| 6 | `core/src/main/scala/kafka/log/LogManager.scala:228` |
| 4 | `core/src/main/scala/kafka/server/ReplicaManager.scala:2172` |
| 3 | `metadata/src/main/java/org/apache/kafka/controller/ReplicationControlManager.java:1698` |
| 1 | `core/src/main/scala/kafka/server/AbstractFetcherManager.scala:203` |

One target supplies 7 of 21 queries (33%). The effective number of independent retrieval
problems is therefore closer to **5 than to 21**, which constrains every inference below far
more than the nominal sample size suggests.

## 2. Sources (all frozen, SHA-256 verified)

| Artifact | SHA-256 (first 16) |
|---|---|
| `results/ast_k_sensitivity_v1/per_query_results/k_5/vector_only.json` | `cf4008fa1297b552…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_10/vector_only.json` | `86191f482f8014ba…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_15/vector_only.json` | `f0b56f2b4b3b0447…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_20/vector_only.json` | `c4410d3e7cced629…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_5/vector_crossencoder.json` | `0ee6289e9b0f3d1e…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_10/vector_crossencoder.json` | `8f3972836314a641…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_15/vector_crossencoder.json` | `097fcb424a1e7ac3…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_20/vector_crossencoder.json` | `42dd1f05dc125571…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_5/kg_crossencoder_static_k.json` | `3d80bbffe3199810…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_10/kg_crossencoder_static_k.json` | `7aab192e6d9d1005…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_15/kg_crossencoder_static_k.json` | `a5e81d225afb598c…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_20/kg_crossencoder_static_k.json` | `cdb5fa32b021a014…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_5/kg_crossencoder_stagedconfidence.json` | `cfd96354b5adbab4…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_10/kg_crossencoder_stagedconfidence.json` | `fc229b0be04c1f31…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_15/kg_crossencoder_stagedconfidence.json` | `2071c33b8201d781…` |
| `results/ast_k_sensitivity_v1/per_query_results/k_20/kg_crossencoder_stagedconfidence.json` | `832a0ee682fde614…` |
| `artifacts/hipporag/retriever_only_per_query.jsonl` | `bfa5fd93002425ac…` |
| `artifacts/hipporag/full_per_query.jsonl` | `705ccc0aec280efe…` |
| `artifacts/hipporag/doc_ensemble_true/per_query.jsonl` | `f430325f0dc79748…` |

## 3. Full-benchmark reconstruction check

Recomputing the 315-query K=20 metrics from these same files reproduces the frozen manuscript
values exactly for **all seven conditions** (Hits, Recall and MRR to <5e-7). The subgroup
numbers therefore come from the same pipeline that produced the paper.

## 4. JIRA-only results at K = 20

| Method | Hits@20 | Recall@20 | tb-Recall@20 | MRR@20 | tb-MRR@20 |
|---|---|---|---|---|---|
| Vector-only | 21/21 | 1.0000 | 1.0000 | 0.6301 | 0.7134 |
| Vector + CE | 21/21 | 1.0000 | 1.0000 | 0.6371 | 0.7268 |
| AST-KG static | 16/21 | 0.7619 | 0.8571 | 0.6286 | 0.7217 |
| AST-KG staged | 21/21 | 1.0000 | 1.0000 | 0.6371 | 0.7268 |
| ColBERTv2-only | 16/21 | 0.7619 | 0.8571 | 0.6050 | 0.7089 |
| HippoRAG (ensemble off) | 5/21 | 0.2381 | 0.3286 | 0.0159 | 0.0275 |
| HippoRAG + doc ensemble [POST-HOC] | 8/21 | 0.3810 | 0.4905 | 0.0480 | 0.0594 |

## 5. Paired comparisons — EXPLORATORY, SEVERELY UNDERPOWERED

| Comparison | ΔRecall@20 | ΔMRR@20 | A-only | B-only | discordant | exact two-sided p |
|---|---|---|---|---|---|---|
| AST-KG static - Vector-only | -0.2381 | -0.0015 | 5 | 0 | 5 | 0.0625 |
| AST-KG staged - Vector-only | +0.0000 | +0.0070 | 0 | 0 | 0 | — |
| AST-KG staged - AST-KG static | +0.2381 | +0.0085 | 0 | 5 | 5 | 0.0625 |
| HippoRAG (ensemble off) - ColBERTv2-only | -0.5238 | -0.5891 | 11 | 0 | 11 | 0.0009765625 |
| HippoRAG + doc ensemble [POST-HOC] - ColBERTv2-only | -0.3810 | -0.5570 | 8 | 0 | 8 | 0.0078125 |
| Vector + CE - Vector-only (MRR only) | N/A (identical candidate set by construction) | +0.0070 |  |  |  | — |

**All five discordant pairs in the AST-KG static comparisons fall on a single target,**
`FetchSessionHandler.java:618`. They are not independent observations, so the nominal
`p = 0.0625` overstates the evidence and must not be read as a near-significant effect.
The same target accounts for all five ColBERTv2-only misses.

Staged AST-KG versus Vector-only has **zero discordant pairs**, so no test was performed.
Vector + CE reranks the identical candidate set, so its Recall@20 is identical by
construction and no Recall test is reported; only the MRR difference is meaningful.

## 6. K sensitivity (JIRA only)

| Method | K=5 | K=10 | K=15 | K=20 |
|---|---|---|---|---|
| Vector-only | 15/21 (0.599) | 18/21 (0.620) | 19/21 (0.625) | 21/21 (0.630) |
| Vector + CE | 15/21 (0.635) | 18/21 (0.644) | 19/21 (0.637) | 21/21 (0.637) |
| AST-KG static | 15/21 (0.631) | 15/21 (0.621) | 15/21 (0.621) | 16/21 (0.629) |
| AST-KG staged | 16/21 (0.647) | 17/21 (0.633) | 18/21 (0.631) | 21/21 (0.637) |
| ColBERTv2-only | 14/21 (0.595) | 15/21 (0.602) | 15/21 (0.602) | 16/21 (0.605) |
| HippoRAG (ensemble off) | 0/21 (0.000) | 1/21 (0.005) | 2/21 (0.008) | 5/21 (0.016) |
| HippoRAG + doc ensemble [POST-HOC] | 2/21 (0.024) | 4/21 (0.037) | 5/21 (0.040) | 8/21 (0.048) |

## 7. Per-query ranks

See `jira_query_level_results.csv`. `MISS` means the gold target was not in the top 20.

Notable individual cases:

- **All four `FetchSessionHandler.java:618` queries that Vector-only ranks 11, 4, 9, 16, 20**
  are lost by AST-KG static and by ColBERTv2-only, while staged AST-KG retains every one.
- `KAFKA-9357-Q3` (same target) is the single case where static AST-KG *keeps* the target
  (rank 6) while staged pushes it to 17 — the only static-better-than-staged query.
- HippoRAG with ensembling off retrieves only 5 of 21 and never ranks a target first;
  ensembling on recovers 3 further queries but still trails its backbone.

## 8. Interpretation (publication-safe)

The factually supported branch is **C (mixed / too small), with a strong caveat**:

> The 21-query JIRA-seeded subgroup yields mixed method-level differences and is too small
> for strong inferential conclusions. It nevertheless provides an issue-seeded robustness
> check complementary to the larger controlled benchmark. Because the 21 queries cover only
> five unique source targets, and because every discordant pair in the AST-KG comparisons
> falls on a single target, the subgroup should be read as approximately five independent
> retrieval problems rather than twenty-one.

What the subgroup *does* support descriptively:

- Vector-only remains fully competitive: it retrieves every JIRA target within K=20.
- Staged AST-KG matches Vector-only on Recall@20 and does not show a clear advantage;
  static AST-KG is worse, driven entirely by one target.
- The external ordering is preserved: both HippoRAG conditions remain well below their
  matched ColBERTv2-only control, and ensembling-on improves on ensembling-off.

What it **cannot** support:

- Any significance claim. n=21 with 5 effective clusters.
- Any claim that AST-KG static is reliably worse; the effect is one target's worth of queries.
- Any claim that benchmark construction does or does not matter — the subgroup is too
  target-concentrated to separate provenance from target difficulty.

