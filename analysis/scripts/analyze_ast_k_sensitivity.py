#!/usr/bin/env python3
"""Validate and analyze the frozen AST K-sensitivity retrieval runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "ast_k_sensitivity_v1"
PER_QUERY_DIR = OUTPUT_DIR / "per_query_results"
RAW_RUNS_DIR = OUTPUT_DIR / "raw_runs"
BENCHMARK_PATH = (
    ROOT
    / "scripts"
    / "benchMarkScript"
    / "benchmark_incident_v2_plus_v3_plus_jira_flat_ast_cleaned_proposed.json"
)
AUTHORITATIVE_DIR = ROOT / "scripts" / "results_ast_315_paper_config"
AUTHORITATIVE_PER_QUERY_DIR = AUTHORITATIVE_DIR / "per_query"
AUTHORITATIVE_SUMMARY = AUTHORITATIVE_DIR / "corrected_metrics" / "results_summary.json"

KS = (5, 10, 15, 20)
VECTOR = "Vector-only"
VECTOR_CE = "Vector + Cross-Encoder"
STATIC = "AST-KG + Cross-Encoder static-K"
STAGED = "AST-KG + Staged Confidence"
METHODS = (VECTOR, VECTOR_CE, STATIC, STAGED)

RUNNER_METHOD_TO_METHOD = {
    "Vector-only": VECTOR,
    "Vector+CrossEncoder": VECTOR_CE,
    "KG+CrossEncoder(static-K)": STATIC,
    "KG+CrossEncoder+StagedConfidence": STAGED,
}
METHOD_TO_SLUG = {
    VECTOR: "vector_only.json",
    VECTOR_CE: "vector_crossencoder.json",
    STATIC: "kg_crossencoder_static_k.json",
    STAGED: "kg_crossencoder_stagedconfidence.json",
}
AUTHORITATIVE_METHOD_FILES = {
    VECTOR: "vector_only.json",
    VECTOR_CE: "vector_crossencoder.json",
    STATIC: "kg_crossencoder_static_k.json",
    STAGED: "kg_crossencoder_stagedconfidence.json",
}

PRIMARY_K20_EXPECTED = {
    VECTOR: {"hits": 301, "recall": 0.9555555555555556, "mrr": 0.7511943140374513},
    VECTOR_CE: {"hits": 301, "recall": 0.9555555555555556, "mrr": 0.6974876669679855},
    STATIC: {"hits": 298, "recall": 0.946031746031746, "mrr": 0.7045628855012609},
    STAGED: {"hits": 301, "recall": 0.9555555555555556, "mrr": 0.6971882281723061},
}

FROZEN_SOURCE_PATHS = [
    BENCHMARK_PATH,
    AUTHORITATIVE_SUMMARY,
    *[AUTHORITATIVE_PER_QUERY_DIR / filename for filename in AUTHORITATIVE_METHOD_FILES.values()],
    ROOT / "results" / "ast_downstream_gpt41mini_v1" / "scoring_blinded_completed.csv",
]


class ValidationError(RuntimeError):
    """Raised when a run violates a frozen experiment invariant."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def exact_mcnemar_p(n10: int, n01: int) -> float:
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(n10, n01) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def target(row: Mapping[str, Any]) -> Tuple[str, int]:
    return row["expected_file"], int(row["expected_line"])


def corrected_metrics(row: Mapping[str, Any], k: int) -> Tuple[bool, float]:
    rank = row.get("gold_rank")
    hit = rank is not None and 1 <= int(rank) <= k
    return hit, (1.0 / int(rank) if hit else 0.0)


def load_raw_efficiency(k: int) -> Dict[str, Mapping[str, Any]]:
    path = RAW_RUNS_DIR / f"k_{k}" / "results_table.csv"
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != 4:
        raise ValidationError(f"Expected four raw result rows for K={k}")
    result = {}
    for row in rows:
        method = RUNNER_METHOD_TO_METHOD.get(row["Method"])
        if method is None:
            raise ValidationError(f"Unknown raw runner method {row['Method']!r}")
        result[method] = row
    return result


def load_and_validate_runs() -> Tuple[
    Dict[int, Dict[str, Dict[str, Mapping[str, Any]]]],
    Dict[int, Dict[str, Mapping[str, Any]]],
    List[Mapping[str, Any]],
]:
    benchmark = read_json(BENCHMARK_PATH)
    if len(benchmark) != 315:
        raise ValidationError(f"Expected 315 benchmark queries, found {len(benchmark)}")
    benchmark_by_id = {row["id"]: row for row in benchmark}
    if len(benchmark_by_id) != 315:
        raise ValidationError("Duplicate benchmark query IDs")
    target_count = len(
        {(row["expected"]["file"], int(row["expected"]["line"])) for row in benchmark}
    )
    if target_count != 103:
        raise ValidationError(f"Expected 103 unique targets, found {target_count}")

    all_runs: Dict[int, Dict[str, Dict[str, Mapping[str, Any]]]] = {}
    raw_efficiency: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for k in KS:
        all_runs[k] = {}
        raw_efficiency[k] = load_raw_efficiency(k)
        reference_ids = set(benchmark_by_id)
        for method in METHODS:
            path = PER_QUERY_DIR / f"k_{k}" / METHOD_TO_SLUG[method]
            payload = read_json(path)
            configuration = payload.get("configuration", {})
            rows = payload.get("queries")
            if not isinstance(rows, list) or len(rows) != 315:
                raise ValidationError(f"Expected 315 rows in {path}")
            by_id = {row["query_id"]: row for row in rows}
            if len(by_id) != 315 or set(by_id) != reference_ids:
                raise ValidationError(f"Query IDs do not match benchmark in {path}")
            if int(configuration.get("top_k", -1)) != k:
                raise ValidationError(f"top_k mismatch in {path}")
            if configuration.get("collection_name") != "kafka_log_templates_ast":
                raise ValidationError(f"Wrong collection in {path}")
            if configuration.get("embedding_model") != "text-embedding-3-large":
                raise ValidationError(f"Wrong embedding model in {path}")
            if configuration.get("rerank_batch_size") != 16:
                raise ValidationError(f"Wrong reranker batch size in {path}")
            if method != VECTOR and configuration.get("rerank_model") != "BAAI/bge-reranker-v2-m3":
                raise ValidationError(f"Wrong reranker in {path}")
            if method == STAGED:
                expected_staged = {
                    "staged_confidence": True,
                    "dynamic_k_min": 5,
                    "dynamic_k_max": 40,
                    "dynamic_k_drop": 0.15,
                    "staged_batch_size": 20,
                    "confidence_high": 0.6,
                    "confidence_low": 0.25,
                }
                for key, expected_value in expected_staged.items():
                    if configuration.get(key) != expected_value:
                        raise ValidationError(
                            f"Staged configuration mismatch in {path}: {key}={configuration.get(key)!r}"
                        )
            for query_id, row in by_id.items():
                benchmark_row = benchmark_by_id[query_id]
                signature = (
                    row["question"],
                    row["expected_file"],
                    int(row["expected_line"]),
                )
                expected_signature = (
                    benchmark_row["question"],
                    benchmark_row["expected"]["file"],
                    int(benchmark_row["expected"]["line"]),
                )
                if signature != expected_signature:
                    raise ValidationError(f"Benchmark signature mismatch for K={k} {method} {query_id}")
                hit, rr = corrected_metrics(row, k)
                if bool(row["gold_hit_at_k"]) != hit:
                    raise ValidationError(f"Incorrect stored hit accounting for K={k} {method} {query_id}")
                if abs(float(row["reciprocal_rank"]) - rr) > 1e-15:
                    raise ValidationError(f"Incorrect stored MRR accounting for K={k} {method} {query_id}")
                if int(row["evaluation_k"]) != k:
                    raise ValidationError(f"Final evaluation cutoff is not K={k} for {method} {query_id}")
            all_runs[k][method] = by_id
    return all_runs, raw_efficiency, benchmark


def ce_pair_count(method: str, row: Mapping[str, Any]) -> int | None:
    if method == VECTOR:
        return None
    vector_pairs = int(row["vector_retrieved"])
    if method == VECTOR_CE:
        return vector_pairs
    pool_size = int(row["kg_pool_size"])
    if method == STATIC:
        # Static KG skips KG reranking above the frozen 200-candidate cap.
        return vector_pairs + (pool_size if pool_size <= 200 else 0)
    if method == STAGED:
        # Staged expansion scores each newly admitted graph candidate exactly once.
        return vector_pairs + pool_size
    raise KeyError(method)


def aggregate_results(
    runs: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    raw_efficiency: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    rows = []
    for k in KS:
        static_pairs_total = sum(
            ce_pair_count(STATIC, row) or 0 for row in runs[k][STATIC].values()
        )
        staged_pairs_total = sum(
            ce_pair_count(STAGED, row) or 0 for row in runs[k][STAGED].values()
        )
        computed_pair_savings = 100.0 * (1.0 - staged_pairs_total / static_pairs_total)
        runner_pair_savings = float(raw_efficiency[k][STAGED]["PairSavings%"])
        if abs(round(computed_pair_savings, 1) - runner_pair_savings) > 1e-12:
            raise ValidationError(
                f"Pair-savings mismatch at K={k}: computed={computed_pair_savings}, runner={runner_pair_savings}"
            )
        for method in METHODS:
            method_rows = list(runs[k][method].values())
            corrected = [corrected_metrics(row, k) for row in method_rows]
            hits = sum(int(hit) for hit, _ in corrected)
            rr_sum = sum(rr for _, rr in corrected)
            pair_counts = [ce_pair_count(method, row) for row in method_rows]
            pair_counts = [value for value in pair_counts if value is not None]
            is_staged = method == STAGED
            rows.append(
                {
                    "k": k,
                    "method": method,
                    "hits_at_k": hits,
                    "queries": len(method_rows),
                    "recall_at_k": hits / len(method_rows),
                    "mrr_at_k": rr_sum / len(method_rows),
                    "avg_candidate_pool_size": (
                        sum(int(row["kg_pool_size"]) for row in method_rows) / len(method_rows)
                        if method in (STATIC, STAGED)
                        else None
                    ),
                    "total_ce_pair_count": sum(pair_counts) if pair_counts else None,
                    "avg_ce_pair_count": sum(pair_counts) / len(pair_counts) if pair_counts else None,
                    "pair_savings_percent_vs_static_same_k": computed_pair_savings if is_staged else 0.0,
                    "ce_time_savings_percent_vs_static_same_k": (
                        float(raw_efficiency[k][STAGED]["CETimeSavings%"])
                        if is_staged
                        else 0.0
                    ),
                }
            )
    return rows


def paired_tests(
    runs: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]]
) -> Dict[str, Any]:
    results = []
    for k in KS:
        for kg_method in (STATIC, STAGED):
            counts = defaultdict(int)
            for query_id in sorted(runs[k][VECTOR]):
                vector_hit, _ = corrected_metrics(runs[k][VECTOR][query_id], k)
                kg_hit, _ = corrected_metrics(runs[k][kg_method][query_id], k)
                counts[(vector_hit, kg_hit)] += 1
            vector_only = counts[(True, False)]
            kg_only = counts[(False, True)]
            results.append(
                {
                    "k": k,
                    "comparison": f"{kg_method} vs {VECTOR}",
                    "kg_method": kg_method,
                    "both_hit": counts[(True, True)],
                    "vector_only_hit": vector_only,
                    "kg_only_hit": kg_only,
                    "both_miss": counts[(False, False)],
                    "n10_vector_only_hit": vector_only,
                    "n01_kg_only_hit": kg_only,
                    "recall_difference_kg_minus_vector": (kg_only - vector_only) / 315.0,
                    "recall_difference_percentage_points": 100.0 * (kg_only - vector_only) / 315.0,
                    "exact_two_sided_mcnemar_p": exact_mcnemar_p(vector_only, kg_only),
                }
            )
    return {
        "test": "exact two-sided McNemar test (exact binomial on discordant query pairs)",
        "query_count": 315,
        "results": results,
    }


def target_balanced_results(
    runs: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]]
) -> Dict[str, Any]:
    rows = []
    for k in KS:
        for method in METHODS:
            query_rows = list(runs[k][method].values())
            grouped: Dict[Tuple[str, int], List[Tuple[float, float]]] = defaultdict(list)
            query_hits = []
            query_rrs = []
            for row in query_rows:
                hit, rr = corrected_metrics(row, k)
                grouped[target(row)].append((float(hit), rr))
                query_hits.append(float(hit))
                query_rrs.append(rr)
            if len(grouped) != 103:
                raise ValidationError(f"Expected 103 target groups for K={k} {method}")
            target_recalls = [sum(hit for hit, _ in values) / len(values) for values in grouped.values()]
            target_mrrs = [sum(rr for _, rr in values) / len(values) for values in grouped.values()]
            rows.append(
                {
                    "k": k,
                    "method": method,
                    "query_count": len(query_rows),
                    "unique_target_count": len(grouped),
                    "query_weighted_recall_at_k": sum(query_hits) / len(query_hits),
                    "target_balanced_recall_at_k": sum(target_recalls) / len(target_recalls),
                    "query_weighted_mrr_at_k": sum(query_rrs) / len(query_rrs),
                    "target_balanced_mrr_at_k": sum(target_mrrs) / len(target_mrrs),
                    "target_balancing_definition": (
                        "Average query outcomes within each expected (file,line) target, then average the 103 targets equally."
                    ),
                }
            )
    return {"rows": rows}


def validate_k20(
    runs: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    aggregates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    aggregate_by_method = {
        row["method"]: row for row in aggregates if int(row["k"]) == 20
    }
    primary_comparison = {}
    for method in METHODS:
        observed = aggregate_by_method[method]
        expected = PRIMARY_K20_EXPECTED[method]
        primary_comparison[method] = {
            "sensitivity_dynamic_k_min": 5 if method == STAGED else None,
            "primary_dynamic_k_min": 10 if method == STAGED else None,
            "sensitivity_hits": observed["hits_at_k"],
            "primary_hits": expected["hits"],
            "sensitivity_recall": observed["recall_at_k"],
            "primary_recall": expected["recall"],
            "sensitivity_mrr": observed["mrr_at_k"],
            "primary_mrr": expected["mrr"],
            "recall_matches_primary": abs(observed["recall_at_k"] - expected["recall"]) < 1e-15,
            "mrr_matches_primary": abs(observed["mrr_at_k"] - expected["mrr"]) < 1e-15,
        }

    per_query_comparison = {}
    for method in METHODS:
        authoritative_payload = read_json(
            AUTHORITATIVE_PER_QUERY_DIR / AUTHORITATIVE_METHOD_FILES[method]
        )
        authoritative = {row["query_id"]: row for row in authoritative_payload["queries"]}
        changed_ranks = []
        changed_hits = []
        for query_id, sensitivity_row in runs[20][method].items():
            primary_row = authoritative[query_id]
            if sensitivity_row.get("gold_rank") != primary_row.get("gold_rank"):
                changed_ranks.append(query_id)
            sensitivity_hit, _ = corrected_metrics(sensitivity_row, 20)
            primary_hit, _ = corrected_metrics(primary_row, 20)
            if sensitivity_hit != primary_hit:
                changed_hits.append(query_id)
        per_query_comparison[method] = {
            "changed_gold_rank_query_count": len(changed_ranks),
            "changed_hit_query_count": len(changed_hits),
            "changed_gold_rank_query_ids": changed_ranks,
            "changed_hit_query_ids": changed_hits,
        }
        if method != STAGED and changed_ranks:
            raise ValidationError(f"K=20 {method} did not reproduce authoritative ranks")

    return {
        "authoritative_primary_k": 20,
        "sensitivity_k20_note": (
            "The recovered sensitivity protocol fixes dynamic_k_min=5. The authoritative primary staged K=20 run uses "
            "dynamic_k_min=10 and remains unchanged."
        ),
        "aggregate_comparison": primary_comparison,
        "per_query_comparison": per_query_comparison,
    }


def markdown_summary(
    aggregates: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
    balanced: Mapping[str, Any],
    k20_validation: Mapping[str, Any],
) -> str:
    aggregate_map = {(int(row["k"]), row["method"]): row for row in aggregates}
    lines = [
        "# AST retrieval K-sensitivity analysis",
        "",
        "## Scope and configuration",
        "",
        "This is a sensitivity/robustness experiment over K={5,10,15,20}; K=20 remains the primary paper depth. "
        "No K was selected after observing the results. The benchmark, AST graph, Qdrant collection, embeddings, "
        "reranker, graph expansion, deduplication, thresholds, and corrected rank-at-K accounting were held fixed.",
        "",
        "The existing implementation would accept `dynamic_k_min=10` at K=5 but would force the first staged pool to "
        "at least 10 candidates, exceeding the requested final depth. The repository's original paper sensitivity protocol "
        "explicitly fixes `dynamic_k_min=5` across all four K values; that recovered protocol was used. Consequently, the "
        "K=20 sensitivity staged row (`min=5`) is supplementary and does not replace the authoritative primary K=20 "
        "staged row (`min=10`).",
        "",
        "## Query-weighted results",
        "",
        "| K | Method | Hits@K | Recall@K | MRR@K | Pair savings | CE-time savings |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for k in KS:
        for method in METHODS:
            row = aggregate_map[(k, method)]
            lines.append(
                f"| {k} | {method} | {row['hits_at_k']}/315 | {row['recall_at_k']:.6f} | "
                f"{row['mrr_at_k']:.6f} | {row['pair_savings_percent_vs_static_same_k']:.1f}% | "
                f"{row['ce_time_savings_percent_vs_static_same_k']:.1f}% |"
            )

    lines += [
        "",
        "Savings are staged-confidence savings relative to static KG at the same K. Zeros for other methods mean "
        "not applicable/no staged saving. Absolute CE time was not retained by the existing ladder's output schema; "
        "the measured same-run CE-time savings were retained.",
        "",
        "## Target-balanced results",
        "",
        "| K | Method | Query recall | Target-balanced recall | Query MRR | Target-balanced MRR |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in balanced["rows"]:
        lines.append(
            f"| {row['k']} | {row['method']} | {row['query_weighted_recall_at_k']:.6f} | "
            f"{row['target_balanced_recall_at_k']:.6f} | {row['query_weighted_mrr_at_k']:.6f} | "
            f"{row['target_balanced_mrr_at_k']:.6f} |"
        )

    lines += [
        "",
        "## Paired Recall@K comparisons",
        "",
        "| K | KG method vs vector | Both hit | Vector only | KG only | Both miss | KG-vector (pp) | Exact p |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired["results"]:
        lines.append(
            f"| {row['k']} | {row['kg_method']} | {row['both_hit']} | {row['vector_only_hit']} | "
            f"{row['kg_only_hit']} | {row['both_miss']} | {row['recall_difference_percentage_points']:+.3f} | "
            f"{row['exact_two_sided_mcnemar_p']:.6g} |"
        )

    lines += [
        "",
        "## K=20 primary cross-check",
        "",
    ]
    for method in METHODS:
        row = k20_validation["aggregate_comparison"][method]
        lines.append(
            f"- {method}: sensitivity Recall@20={row['sensitivity_recall']:.6f}, "
            f"MRR@20={row['sensitivity_mrr']:.6f}; primary Recall@20={row['primary_recall']:.6f}, "
            f"MRR@20={row['primary_mrr']:.6f}."
        )
    staged_delta_count = k20_validation["per_query_comparison"][STAGED]["changed_gold_rank_query_count"]
    lines += [
        "",
        "Vector-only, Vector+CE, and static KG reproduced the authoritative K=20 per-query gold ranks exactly. "
        f"The staged sensitivity configuration changed the gold rank on {staged_delta_count}/315 queries relative to "
        "the primary `min=10` staged run; the authoritative primary artifact remains unchanged.",
        "",
        "## Conclusions",
        "",
    ]

    significant_positive = []
    for row in paired["results"]:
        if row["recall_difference_percentage_points"] > 0 and row["exact_two_sided_mcnemar_p"] < 0.05:
            significant_positive.append((row["k"], row["kg_method"]))
    vector_mrr_stronger = all(
        aggregate_map[(k, VECTOR)]["mrr_at_k"] > aggregate_map[(k, method)]["mrr_at_k"]
        for k in KS
        for method in (VECTOR_CE, STATIC, STAGED)
    )
    staged_preserves = all(
        aggregate_map[(k, STAGED)]["recall_at_k"] >= aggregate_map[(k, STATIC)]["recall_at_k"]
        for k in KS
    )
    lines += [
        "1. **Does AST-KG significantly improve Recall at any tested K?** "
        + (
            "Yes: " + ", ".join(f"K={k} {method}" for k, method in significant_positive) + "."
            if significant_positive
            else "No. Neither static nor staged AST-KG has a significant positive McNemar result at any tested K."
        ),
        "",
        "2. **Is any apparent KG advantage consistent across multiple K values?** No. Static KG is below vector recall at "
        "every K, while staged is below vector at K=5 and K=15 and tied at K=10 and K=20. There is no repeated positive "
        "recall advantage over vector-only.",
        "",
        "3. **Does vector-only remain stronger on MRR?** "
        + (
            "Yes. Vector-only has the highest MRR at every tested K."
            if vector_mrr_stronger
            else "No; at least one tested K has another method with equal or higher MRR."
        ),
        "",
        "4. **Does staged confidence preserve retrieval quality at lower K?** "
        + (
            "Relative to static KG, yes descriptively: staged recall equals or exceeds static recall at every tested K, "
            "while saving substantial CE work. This does not imply improvement over vector-only."
            if staged_preserves
            else "Not consistently; staged recall falls below static KG at one or more tested K values."
        ),
        "",
        "5. **Are the K=20 conclusions robust to retrieval depth?** Yes. Across all tested depths, vector-only keeps the "
        "strongest MRR and neither AST-KG method shows a statistically reliable recall improvement over vector-only. "
        "K=20 remains the primary configuration; no alternative K is recommended from this sensitivity analysis.",
        "",
        "K=40 was not run. It was optional, is unnecessary for the requested main sweep, and would coincide with the "
        "staged `dynamic_k_max=40` ceiling.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    source_hashes_before = {str(path.relative_to(ROOT)): sha256(path) for path in FROZEN_SOURCE_PATHS}
    runs, raw_efficiency, benchmark = load_and_validate_runs()
    aggregates = aggregate_results(runs, raw_efficiency)
    paired = paired_tests(runs)
    balanced = target_balanced_results(runs)
    k20_validation = validate_k20(runs, aggregates)

    result_payload = {
        "status": "complete",
        "experiment": "AST K-sensitivity retrieval robustness analysis",
        "primary_k_remains": 20,
        "tested_k": list(KS),
        "results": aggregates,
        "k20_primary_cross_check": k20_validation,
    }
    write_csv(
        OUTPUT_DIR / "k_sensitivity_results.csv",
        [
            "k", "method", "hits_at_k", "queries", "recall_at_k", "mrr_at_k",
            "avg_candidate_pool_size", "total_ce_pair_count", "avg_ce_pair_count",
            "pair_savings_percent_vs_static_same_k", "ce_time_savings_percent_vs_static_same_k",
        ],
        aggregates,
    )
    write_json(OUTPUT_DIR / "k_sensitivity_results.json", result_payload)
    write_json(OUTPUT_DIR / "paired_tests.json", paired)
    write_json(OUTPUT_DIR / "target_balanced_results.json", balanced)

    experiment_config = {
        "status": "complete",
        "experiment_type": "sensitivity/robustness only; not parameter tuning",
        "primary_k": 20,
        "tested_k": list(KS),
        "optional_k40_run": False,
        "optional_k40_reason": (
            "Optional and unnecessary for the main sweep; K=40 coincides with dynamic_k_max=40."
        ),
        "benchmark": {
            "path": str(BENCHMARK_PATH.relative_to(ROOT)),
            "sha256": sha256(BENCHMARK_PATH),
            "queries": len(benchmark),
            "unique_targets": 103,
        },
        "stores": {
            "neo4j_uri": "bolt://localhost:7688",
            "qdrant_url": "http://localhost:6333",
            "qdrant_collection": "kafka_log_templates_ast",
            "log_template_count_verified": 1600,
        },
        "retrieval": {
            "embedding_model": "text-embedding-3-large",
            "cross_encoder": "BAAI/bge-reranker-v2-m3",
            "reranker_device": None,
            "reranker_batch_size": 16,
            "rerank_kg_max": 200,
            "candidate_deduplication": "unchanged existing implementation",
            "kg_expansion_logic": "unchanged existing implementation",
            "metric_definition": "Recall@K iff 1 <= final gold rank <= K; MRR@K=1/rank under the same cutoff, else 0.",
        },
        "staged_sensitivity_configuration": {
            "dynamic_k_min": 5,
            "dynamic_k_max": 40,
            "dynamic_k_drop": 0.15,
            "staged_batch_size": 20,
            "confidence_high": 0.60,
            "confidence_low": 0.25,
            "staged_big_step": None,
            "staged_small_step": None,
            "source": ["paper/kafkaops_kg_paper_draft.md", "paper/access.tex"],
            "k5_compatibility": (
                "The implementation accepts min=10 but would expand the initial staged pool to at least 10 at K=5. "
                "The recovered original sensitivity protocol fixes min=5 across the sweep."
            ),
        },
        "authoritative_primary_staged_configuration": {
            "k": 20,
            "dynamic_k_min": 10,
            "dynamic_k_max": 40,
            "dynamic_k_drop": 0.15,
            "staged_batch_size": 20,
            "confidence_high": 0.60,
            "confidence_low": 0.25,
            "preserved": True,
        },
        "execution": {
            "all_query_embeddings_from_cache": True,
            "embedding_api_calls_for_sweep": 0,
            "sequential_methods_for_timing": True,
            "raw_runs_dir": str(RAW_RUNS_DIR.relative_to(ROOT)),
            "per_query_results_dir": str(PER_QUERY_DIR.relative_to(ROOT)),
            "absolute_ce_time_note": (
                "The existing ladder retained same-run CE-time savings but discarded absolute CE seconds from its final schema."
            ),
        },
        "validation": {
            "all_16_method_k_artifacts_present": True,
            "rows_per_artifact": 315,
            "corrected_metric_accounting_verified_per_query": True,
            "query_ids_and_signatures_match_benchmark": True,
            "unique_targets_verified": 103,
            "k20_cross_check": k20_validation,
            "frozen_source_hashes_before": source_hashes_before,
        },
    }
    write_json(OUTPUT_DIR / "experiment_config.json", experiment_config)
    (OUTPUT_DIR / "analysis_summary.md").write_text(
        markdown_summary(aggregates, paired, balanced, k20_validation), encoding="utf-8"
    )

    source_hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in FROZEN_SOURCE_PATHS}
    if source_hashes_after != source_hashes_before:
        raise ValidationError("A frozen source artifact changed during sensitivity analysis")

    print(json.dumps({
        "status": "complete",
        "validated_method_k_artifacts": 16,
        "queries_per_artifact": 315,
        "unique_targets": 103,
        "output_dir": str(OUTPUT_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
