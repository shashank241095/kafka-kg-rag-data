# Data and Reproducibility Artifacts

This repository accompanies the manuscript:

> **Evaluating AST-Derived Knowledge Graph Retrieval for AIOps Incident Diagnosis in Apache Kafka**
> Shashank Mishra, Shweta Mishra

It contains the **frozen research artifacts** supporting the reported results:
the audited retrieval benchmark, the archived pre-audit benchmark, the frozen
source-linked log corpus, per-query retrieval outputs for every evaluated
condition, downstream evaluation records, statistical outputs, configurations,
and the pre-specified analysis plan.

Artifacts are byte-identical to those used to produce the manuscript. Nothing
here was regenerated, relabelled, or recomputed for publication.

---

## What is and is not redistributed

- **Apache Kafka source code is NOT redistributed.** The benchmark targets and
  the corpus reference `(file, line)` locations inside the evaluated Apache
  Kafka source snapshot. The corpus records extracted log-template strings and
  their ownership metadata, which are short extracts of Apache-2.0-licensed
  Kafka source; see `NOTICE` and `LICENSE-DATA`.
- **HippoRAG source code is NOT redistributed.** The exact upstream version used
  is recorded in `configs/resolved_config.json`:
  repository `https://github.com/OSU-NLP-Group/HippoRAG`, tag `v1.0.0`,
  commit `b144c46df14cabe5f5822d8caded4bec5f709461`.
- **No API credentials are included.** No `.env`, key, or token is present
  anywhere in this repository or its history.
- No model weights, virtual environments, caches, or private submission files
  are included.

## Benchmark: 321 archived vs 315 final

| | Queries | Unique targets |
|---|---|---|
| Archived original | **321** | 105 |
| Final audited | **315** | 103 |

An AST audit identified **six queries whose expected targets were legacy
regular-expression false positives rather than real logger calls**:
`L081-Q1`, `L081-Q2`, `L081-Q3` at `ResultOrError.java:89`, and `L084-Q1`,
`L084-Q2`, `L084-Q3` at `ResultOrError.java:87`, both under
`metadata/src/main/java/org/apache/kafka/controller/`. Removing only those six
yields the final 315-query, 103-target benchmark. **The 321-query archive is
preserved unchanged** in `benchmarks/archived_321/` so the correction is
traceable rather than a silent relabelling. The audit records are in
`benchmarks/audit/`.

## Directory structure

```
benchmarks/
  final_315/    benchmark_incident_v2_plus_v3_plus_jira_flat_ast_cleaned_proposed.json   <- the 315-query benchmark
  archived_321/ benchmark_incident_v2_plus_v3_plus_jira_flat.json                        <- the 321-query archive
                benchmark_incident_v2_flat.json, benchmark_incident_v3_flat.json,
                benchmark_incident_jira_seed_flat.json                                    <- the three source subsets
  audit/        benchmark_ast_audit_report.json, benchmark_ast_revised_audit_report.json
corpus/         frozen_corpus_1600.jsonl      <- the 1,600 source-linked log records
                doc_id_mapping.csv, frozen_checksums_before.csv, frozen_checksums_after.csv
results/
  internal/     per_query/ (Vector-only, Vector+CE, AST-KG static, AST-KG staged)
                corrected_metrics/, statistical_analysis/, results_table.csv
  colbert/      retriever_only_per_query.{csv,jsonl}, stock-path cross-check, timing
  hipporag/     full_per_query.{csv,jsonl}, evaluation.json, graph_diagnostics.json,
                table_main.csv, table_paired.csv, table_k_sensitivity.csv,
                doc_ensemble_true_posthoc/    <- POST-HOC sensitivity, see below
  downstream/   selected_queries.json, raw_generations.json, SCORING_RUBRIC.md,
                scoring_blinded_completed.csv, unblinded_analysis/
analysis/
  scripts/            metric, statistical, audit and comparison scripts
  statistical_outputs/ ast_k_sensitivity_v1/, jira_subgroup_analysis_v1/
configs/        resolved_config.json (HippoRAG), environment.md, store_verification.json
docs/           PRE_SPECIFIED_ANALYSIS_PLAN.md, serialization_entity_audit.md
```

## Which artifact supports which reported result

| Manuscript item | Artifact |
|---|---|
| **Table 1** — main K=20 retrieval | `results/hipporag/table_main.csv` (all six primary conditions); internal four also in `results/internal/corrected_metrics/results_table.csv` (with the staged cross-encoder savings); post-hoc row from `results/hipporag/doc_ensemble_true_posthoc/aggregate_metrics.csv` |
| **Table 2** — extraction and graph diagnostics | `results/hipporag/graph_diagnostics.json` |
| **Table 3** — K-sensitivity | `analysis/statistical_outputs/ast_k_sensitivity_v1/k_sensitivity_results.csv` (internal), `results/hipporag/table_k_sensitivity.csv` (external), `results/hipporag/doc_ensemble_true_posthoc/k_sensitivity.csv` (post-hoc) |
| **Table 4** — paired retrieval statistics | `results/hipporag/table_paired.csv`, `results/internal/statistical_analysis/statistical_analysis.json`, `results/hipporag/doc_ensemble_true_posthoc/paired_statistics.csv` |
| **Table 5** — downstream evaluation | `results/downstream/unblinded_analysis/aggregate_results.csv`, `paired_tests.json`, `confidence_intervals.json` |
| **Appendix table** — JIRA-seeded subgroup | `analysis/statistical_outputs/jira_subgroup_analysis_v1/jira_subgroup_metrics.csv` |
| Per-query rankings, all conditions | `results/internal/per_query/`, `results/colbert/`, `results/hipporag/` |
| Pre-specified analysis plan | `docs/PRE_SPECIFIED_ANALYSIS_PLAN.md` |

**Figure 1** is a hand-drawn schematic of the experimental design and has no
underlying data file.

## The HippoRAG document-ensemble condition is POST-HOC

`results/hipporag/doc_ensemble_true_posthoc/` holds a **post-hoc sensitivity
analysis**, run after the primary result was observed, varying exactly one
documented option (`doc_ensemble` from `false` to `true`) while reusing every
frozen input unchanged. It is **not** a pre-specified condition and is labelled
as post-hoc wherever it appears in the manuscript. The pre-specified external
condition is `doc_ensemble = false`.

## Reproducing the analyses

The statistical analyses are recomputable from the frozen per-query outputs in
this repository using `analysis/scripts/`, with no network or API access:

- `analyze_ast_retrieval_statistics.py` — exact McNemar tests, Wilson intervals,
  paired and target-cluster bootstrap (seed 42, 10,000 replicates)
- `analyze_ast_k_sensitivity.py` — K-sensitivity metrics
- `analyze_downstream_unblinded.py` — downstream aggregation and paired tests
- `compare_doc_ensemble.py` — post-hoc ensemble comparison
- `evaluate.py` — retrieval metric computation
- `audit_ast_benchmark.py` — benchmark/AST audit
- `checksum_frozen.py` — freeze verification

**Full end-to-end reruns are NOT possible from this repository alone.**
Regenerating retrieval or generation outputs additionally requires: an Apache
Kafka source checkout, a running Neo4j and Qdrant instance, OpenAI API access
(`text-embedding-3-large`, `gpt-4.1-mini-2025-04-14`, `gpt-3.5-turbo-1106`), the
`BAAI/bge-reranker-v2-m3` cross-encoder, and a HippoRAG v1.0.0 checkout. Some
scripts contain default paths and local service endpoints
(`bolt://localhost:7688`, `localhost:6333`) from the original run environment,
and some frozen JSON artifacts record the absolute paths of that environment.
These are preserved deliberately so the artifacts stay byte-identical to those
used for the manuscript.

Software and environment details are in `configs/environment.md`.

## Licensing

Three-part, because the contents have different origins:

| Content | License |
|---|---|
| Original code in `analysis/scripts/` | **MIT** — see `LICENSE` |
| Original research data (benchmarks, results, statistical outputs, downstream records) | **CC BY 4.0** — see `LICENSE-DATA` |
| Extracts of Apache Kafka source (log-template strings, `(file, line)` targets, class/method identifiers in `corpus/` and the benchmark files) | **Apache License 2.0**, attribution in `NOTICE` |

No license here is applied to third-party code. Apache Kafka and HippoRAG remain
under their own licenses and are not redistributed.

## Citation

See `CITATION.cff`. The accompanying manuscript is under review; it has no DOI
yet, so none is claimed here.
