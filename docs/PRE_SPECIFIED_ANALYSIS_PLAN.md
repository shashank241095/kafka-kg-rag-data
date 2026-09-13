# Pre-Registered Interpretation and Diagnostics Plan

**Written before any HippoRAG graph was built and before any HippoRAG retrieval result was
observed.** Its purpose is to fix in advance both how a result will be read and which graph
diagnostics will be reported, so that neither can be selected after the fact.

Git HEAD at time of writing: `609e93c0b56f91599503e24f3c55dd8c5dbd6ec8`
Frozen tag: `frozen-pre-hipporag`

---

## 1. Domain mismatch statement

> Our Kafka benchmark differs substantially from the passage-oriented, knowledge-integration
> settings for which HippoRAG was developed. Each retrieval unit is a short source-linked
> operational record rather than a conventional prose passage. The HippoRAG experiment
> therefore evaluates transfer of its text-derived graph mechanism to exact source-location
> retrieval; poor performance, if observed, must not be interpreted as evidence that
> text-derived graph retrieval is generally ineffective.

## 2. What is *not* predeclared

The following are **hypotheses to be measured, not predictions**. None is assumed, and none
may be asserted unless the diagnostics in §3 support it:

- that HippoRAG will underperform;
- that OpenIE extraction will be degenerate or sparse;
- that the induced graph will be disconnected;
- that short documents inherently produce poor OpenIE output.

A result in HippoRAG's favour will be reported exactly as readily as a result against it.

## 3. Pre-registered graph diagnostics

These will be reported whatever the retrieval outcome, and are **descriptive only**. They
will not be used to tune the graph, the threshold, or any other parameter.

**Entities**
- total extracted entities
- unique entities
- entities per document: mean, median, p25, p75, min, max

**Relations / triples**
- total extracted triples
- per-document distribution: mean, median, p25, p75, min, max

**Extraction coverage**
- percentage of documents producing zero entities
- percentage of documents producing zero relations

**Graph structure**
- node count
- edge count
- graph density (if meaningful for the induced graph type)
- number of connected components
- largest connected component size
- largest connected component as a percentage of nodes
- singleton-node count
- isolated-document / isolated-entity statistics where the implementation exposes them

**Retrieval-time behaviour** (recorded from HippoRAG's own `self.statistics` counters)
- number of queries resolved by the `ppr` path
- number resolved by the `doc` path (no entities found in the query)
- number resolved by the `ppr_doc_ensemble` path

## 4. Pre-registered interpretation branches

Exactly one branch will be written up, chosen by the observed data:

| Case | Observation | Permitted claim |
|---|---|---|
| **A** | Full HippoRAG > ColBERTv2-only | Graph propagation contributed positively under this transfer setting. |
| **B** | Full HippoRAG ≈ ColBERTv2-only | No reliable incremental graph benefit was established. |
| **C** | Full HippoRAG < ColBERTv2-only | HippoRAG graph propagation reduced exact-target retrieval under this configuration. |
| **D** | Diagnostics show extremely sparse or disconnected extraction | Scope explicitly: *"The observed behavior may reflect mismatch between passage-oriented OpenIE graph construction and short source-linked operational retrieval units."* |

"≈" means the target-cluster bootstrap 95% CI for the difference contains 0 **and** the exact
two-sided McNemar test on paired Recall@20 is non-significant at the 0.05 level. This
threshold is fixed here, before any result.

**Never permitted, in any branch:** the generalization *"text-derived graph retrieval does not
work."* A single-corpus, single-configuration transfer result cannot support it.

## 5. Pre-committed comparison structure

The primary external-baseline comparison is:

    Full HippoRAG (ColBERTv2)   vs   ColBERTv2-only

because both conditions share the same retrieval backbone, checkpoint, document text and
harness. The graph contribution is defined as their difference. A comparison of HippoRAG
against `text-embedding-3-large` is confounded by retriever identity and will **not** be used
to attribute any effect to the graph; it is reported only as a cross-system context number.

The 51.8% cross-encoder pair-workload saving and the 51.4% measured cross-encoder reranking
time saving apply **only** to AST-KG staged vs AST-KG static. They will not be attached to any
HippoRAG row.

## 6. Fixed analysis parameters

| Item | Value | Fixed by |
|---|---|---|
| Gold matching | exact canonical `(file,line)` | frozen protocol |
| K values | 5, 10, 15, 20 (primary 20) | frozen protocol |
| Metrics | Hits@K, Recall@K, MRR@K, target-balanced Recall/MRR | frozen definitions, imported from `scripts/analyze_ast_retrieval_statistics.py` |
| Paired recall test | exact two-sided McNemar | frozen implementation |
| CI method | target-cluster bootstrap, 103 clusters, 10,000 replicates, seed 42 | frozen implementation |
| Target balancing | average within each `(file,line)`, then average the 103 targets equally | frozen definition |
| HippoRAG config | `sim_threshold=0.8`, `damping=0.5`, `doc_ensemble=f`, `max_steps=1`, `recognition_threshold=0.9`, `extraction_type=ner`, `graph_alg=ppr` | official, see `environment.md` §4-5 |

No parameter above may be revised after results are seen.

---

## Addendum — 2026-09-09, written before any HippoRAG retrieval result existed

Recorded while OpenIE extraction was still running (stage 1 of 6), before `full_per_query.jsonl`
existed and before any HippoRAG retrieval metric had been computed. Two items in §3 need a
correction of *instrument*, not of *intent*.

**1. The retrieval-time path counters cannot come from HippoRAG's own counters.**
§3 committed to reading the `ppr` / `doc` / `ppr_doc_ensemble` counts from `self.statistics`.
Inspecting `src/hipporag.py` shows all three increments (`:240`, `:243`, `:259`) sit inside
`if self.doc_ensemble or self.dpr_only:` (`:236`). Under the pre-registered main configuration
(`doc_ensemble=False`, `dpr_only=False`) that branch is never entered, so `self.statistics` is
**empty by construction**. The same quantity is therefore measured from the per-query record
instead: a query with ≥1 extracted entity takes the PPR path (`:213-232`), a query with zero
extracted entities takes the uniform fallback. Counts, percentages and the per-query entity
distribution are reported. This substitutes an equivalent instrument for an inoperative one;
it does not change what is being measured or add any freedom to the analysis.

**2. One additional diagnostic is added, pre-committed here.**
`src/hipporag.py:232-233` sets `ppr_doc_prob` to a **uniform** vector when a query yields no
entities, and `:262` then takes `doc_prob = ppr_doc_prob`. All 1,600 documents tie, and
`np.argsort(..., kind='mergesort')[::-1]` returns descending document indices — a
content-independent ranking. We therefore additionally report:

- number of queries whose ranking is confirmed degenerate (zero entities **and** returned
  ranking exactly equal to descending document indices);
- Hits@20 split by whether the query produced entities.

This is upstream v1 behaviour and will not be modified, patched, or worked around. It is
reported as an observation. If it proves material to the aggregate, that will be stated
explicitly as a property of the method under this transfer setting rather than presented as a
defect of the benchmark or corrected out of the result.

No metric definition, statistical test, K value, configuration parameter or interpretation
branch from §1-§6 is altered.
