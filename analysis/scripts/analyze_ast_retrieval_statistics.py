#!/usr/bin/env python3
"""Paired and target-clustered analysis of frozen AST retrieval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METHOD_FILES = {
    "Vector-only": "vector_only.json",
    "Vector+CrossEncoder": "vector_crossencoder.json",
    "AST-KG+CrossEncoder(static-K)": "kg_crossencoder_static_k.json",
    "AST-KG+CrossEncoder+StagedConfidence": "kg_crossencoder_stagedconfidence.json",
}

COMPARISONS = [
    ("Vector-only", "Vector+CrossEncoder"),
    ("Vector-only", "AST-KG+CrossEncoder(static-K)"),
    ("Vector-only", "AST-KG+CrossEncoder+StagedConfidence"),
    ("Vector+CrossEncoder", "AST-KG+CrossEncoder(static-K)"),
    ("AST-KG+CrossEncoder(static-K)", "AST-KG+CrossEncoder+StagedConfidence"),
]

BOOTSTRAP_COMPARISONS = [
    ("Vector-only", "Vector+CrossEncoder"),
    ("Vector-only", "AST-KG+CrossEncoder(static-K)"),
    ("Vector-only", "AST-KG+CrossEncoder+StagedConfidence"),
    ("Vector+CrossEncoder", "AST-KG+CrossEncoder(static-K)"),
    ("AST-KG+CrossEncoder(static-K)", "AST-KG+CrossEncoder+StagedConfidence"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--jira", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--rank-examples", type=int, default=10)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target_of(row: dict) -> Tuple[str, int]:
    return row["expected_file"], int(row["expected_line"])


def signature(row: dict) -> Tuple[str, str, int]:
    expected = row["expected"]
    return row["question"], expected["file"], int(expected["line"])


def hit_at_k(row: dict, k: int) -> bool:
    rank = row.get("gold_rank")
    return rank is not None and int(rank) <= k


def reciprocal_rank_at_k(row: dict, k: int) -> float:
    rank = row.get("gold_rank")
    return 0.0 if rank is None or int(rank) > k else 1.0 / int(rank)


def exact_mcnemar_p(n10: int, n01: int) -> float:
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(n10, n01) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_methods(per_query_dir: Path) -> Dict[str, Dict[str, dict]]:
    methods: Dict[str, Dict[str, dict]] = {}
    reference_ids = None
    for method, filename in METHOD_FILES.items():
        payload = read_json(per_query_dir / filename)
        rows = payload["queries"]
        by_id = {row["query_id"]: row for row in rows}
        if len(rows) != len(by_id):
            raise SystemExit(f"Duplicate query IDs in {filename}")
        ids = set(by_id)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise SystemExit(f"Query ID mismatch in {filename}")
        methods[method] = by_id
    return methods


def subset_mapping(v2_path: Path, v3_path: Path, jira_path: Path) -> Dict[Tuple[str, str, int], str]:
    result: Dict[Tuple[str, str, int], str] = {}
    for subset, path in (("v2", v2_path), ("v3", v3_path), ("JIRA", jira_path)):
        for row in read_json(path):
            key = signature(row)
            existing = result.get(key)
            if existing is not None and existing != subset:
                raise SystemExit(f"Ambiguous benchmark provenance for {key}")
            result[key] = subset
    return result


def rank_change_analysis(
    methods: Dict[str, Dict[str, dict]],
    method_a: str,
    method_b: str,
    query_ids: Sequence[str],
    k: int,
    example_count: int,
) -> Tuple[dict, List[dict]]:
    changes = []
    paired_deltas = []
    improved = worsened = unchanged = 0
    for query_id in query_ids:
        row_a = methods[method_a][query_id]
        row_b = methods[method_b][query_id]
        rank_a = row_a.get("gold_rank")
        rank_b = row_b.get("gold_rank")
        effective_a = int(rank_a) if rank_a is not None and int(rank_a) <= k else k + 1
        effective_b = int(rank_b) if rank_b is not None and int(rank_b) <= k else k + 1
        delta = effective_b - effective_a
        if delta < 0:
            improved += 1
            direction = "improved"
        elif delta > 0:
            worsened += 1
            direction = "worsened"
        else:
            unchanged += 1
            direction = "unchanged"
        if hit_at_k(row_a, k) and hit_at_k(row_b, k):
            paired_deltas.append(int(rank_b) - int(rank_a))
        changes.append(
            {
                "comparison": f"{method_b} minus {method_a}",
                "direction": direction,
                "query_id": query_id,
                "question": row_a["question"],
                "expected_file": row_a["expected_file"],
                "expected_line": row_a["expected_line"],
                "rank_a": rank_a,
                "rank_b": rank_b,
                "effective_rank_change_b_minus_a": delta,
            }
        )

    examples = sorted((row for row in changes if row["direction"] == "improved"), key=lambda row: row["effective_rank_change_b_minus_a"])[
        :example_count
    ]
    examples += sorted((row for row in changes if row["direction"] == "worsened"), key=lambda row: row["effective_rank_change_b_minus_a"], reverse=True)[
        :example_count
    ]
    summary = {
        "method_a": method_a,
        "method_b": method_b,
        "rank_comparison_definition": f"Misses or ranks > {k} are assigned effective rank {k + 1}.",
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "both_hit_count_for_mean_median": len(paired_deltas),
        "mean_rank_change_b_minus_a_among_both_hits": statistics.fmean(paired_deltas) if paired_deltas else None,
        "median_rank_change_b_minus_a_among_both_hits": statistics.median(paired_deltas) if paired_deltas else None,
    }
    return summary, examples


def pool_diagnosis(row: dict, k: int) -> str:
    if not row.get("kg_candidate_pool_contains_gold_before_rerank"):
        return "graph expansion did not add the gold target"
    full_rank = row.get("kg_rank_after_rerank_full")
    if row.get("kg_pool_size", 0) > 200 and full_rank == row.get("kg_rank_before_rerank"):
        if full_rank is None or int(full_rank) > k:
            return "gold was in pool; reranking skipped by pool cap; raw pool rank was outside top-K"
    if full_rank is None or int(full_rank) > k:
        return "gold was in pool but reranking placed it outside top-K"
    return "gold was in expanded pool and retained in top-K"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    methods = load_methods(Path(args.per_query_dir))
    benchmark = read_json(Path(args.benchmark))
    query_ids = [row["id"] for row in benchmark]
    if set(query_ids) != set(next(iter(methods.values()))):
        raise SystemExit("Per-query output IDs do not match the revised benchmark")

    provenance = subset_mapping(Path(args.v2), Path(args.v3), Path(args.jira))
    query_subset = {}
    for row in benchmark:
        subset = provenance.get(signature(row))
        if subset is None:
            raise SystemExit(f"No v2/v3/JIRA provenance for {row['id']}")
        query_subset[row["id"]] = subset

    targets: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    reference = methods["Vector-only"]
    for query_id in query_ids:
        targets[target_of(reference[query_id])].append(query_id)
    if len(targets) != 103:
        raise SystemExit(f"Expected 103 target clusters, found {len(targets)}")

    long_rows = []
    query_metrics = {}
    for method, by_id in methods.items():
        hits = []
        rrs = []
        for query_id in query_ids:
            row = by_id[query_id]
            hit = hit_at_k(row, args.k)
            rr = reciprocal_rank_at_k(row, args.k)
            hits.append(float(hit))
            rrs.append(rr)
            long_rows.append(
                {
                    "method": method,
                    "query_id": query_id,
                    "subset": query_subset[query_id],
                    "question": row["question"],
                    "expected_file": row["expected_file"],
                    "expected_line": row["expected_line"],
                    "gold_hit_at_20": hit,
                    "gold_rank": row.get("gold_rank"),
                    "reciprocal_rank_at_20": rr,
                    "kg_candidate_pool_contains_gold_before_rerank": row.get(
                        "kg_candidate_pool_contains_gold_before_rerank"
                    ),
                    "kg_rank_before_rerank": row.get("kg_rank_before_rerank"),
                    "kg_rank_after_rerank_full": row.get("kg_rank_after_rerank_full"),
                    "kg_pool_size": row.get("kg_pool_size"),
                }
            )
        query_metrics[method] = {
            "query_count": len(query_ids),
            "hits_at_20": int(sum(hits)),
            "query_recall_at_20": statistics.fmean(hits),
            "query_mrr_at_20": statistics.fmean(rrs),
        }

    paired_rows = []
    for method_a, method_b in COMPARISONS:
        both_hit = a_only = b_only = both_miss = 0
        for query_id in query_ids:
            hit_a = hit_at_k(methods[method_a][query_id], args.k)
            hit_b = hit_at_k(methods[method_b][query_id], args.k)
            if hit_a and hit_b:
                both_hit += 1
            elif hit_a:
                a_only += 1
            elif hit_b:
                b_only += 1
            else:
                both_miss += 1
        paired_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "both_hit": both_hit,
                "n10_a_only": a_only,
                "n01_b_only": b_only,
                "both_miss": both_miss,
                "exact_mcnemar_two_sided_p": exact_mcnemar_p(a_only, b_only),
            }
        )

    target_rows = []
    target_balanced = {}
    for method, by_id in methods.items():
        target_hit_rates = []
        target_mean_rrs = []
        for (relative_file, line), associated_ids in sorted(targets.items()):
            target_hits = [float(hit_at_k(by_id[query_id], args.k)) for query_id in associated_ids]
            target_rrs = [reciprocal_rank_at_k(by_id[query_id], args.k) for query_id in associated_ids]
            hit_rate = statistics.fmean(target_hits)
            mean_rr = statistics.fmean(target_rrs)
            target_hit_rates.append(hit_rate)
            target_mean_rrs.append(mean_rr)
            target_rows.append(
                {
                    "method": method,
                    "expected_file": relative_file,
                    "expected_line": line,
                    "associated_query_count": len(associated_ids),
                    "query_hit_fraction_at_20": hit_rate,
                    "mean_reciprocal_rank_at_20": mean_rr,
                }
            )
        target_balanced[method] = {
            **query_metrics[method],
            "unique_target_count": len(targets),
            "target_balanced_recall_at_20": statistics.fmean(target_hit_rates),
            "target_balanced_mrr_at_20": statistics.fmean(target_mean_rrs),
        }

    rng = random.Random(args.bootstrap_seed)
    target_keys = sorted(targets)
    bootstrap_rows = []
    for method_a, method_b in BOOTSTRAP_COMPARISONS:
        recall_differences = []
        mrr_differences = []
        for _ in range(args.bootstrap_replicates):
            sampled_targets = [target_keys[rng.randrange(len(target_keys))] for _ in target_keys]
            sampled_ids = [query_id for target in sampled_targets for query_id in targets[target]]
            recall_diff = statistics.fmean(
                float(hit_at_k(methods[method_b][query_id], args.k))
                - float(hit_at_k(methods[method_a][query_id], args.k))
                for query_id in sampled_ids
            )
            mrr_diff = statistics.fmean(
                reciprocal_rank_at_k(methods[method_b][query_id], args.k)
                - reciprocal_rank_at_k(methods[method_a][query_id], args.k)
                for query_id in sampled_ids
            )
            recall_differences.append(recall_diff)
            mrr_differences.append(mrr_diff)
        for metric, values in (("Recall@20", recall_differences), ("MRR@20", mrr_differences)):
            point = (
                target_balanced[method_b]["query_recall_at_20"] - target_balanced[method_a]["query_recall_at_20"]
                if metric == "Recall@20"
                else target_balanced[method_b]["query_mrr_at_20"] - target_balanced[method_a]["query_mrr_at_20"]
            )
            bootstrap_rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "difference": f"{method_b} - {method_a}",
                    "metric": metric,
                    "point_estimate_query_weighted": point,
                    "ci_95_lower": percentile(values, 0.025),
                    "ci_95_upper": percentile(values, 0.975),
                    "bootstrap_seed": args.bootstrap_seed,
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "resampling_unit": "expected (file,line) target cluster",
                }
            )

    subset_rows = []
    for subset in ("v2", "v3", "JIRA"):
        subset_ids = [query_id for query_id in query_ids if query_subset[query_id] == subset]
        for method, by_id in methods.items():
            subset_rows.append(
                {
                    "subset": subset,
                    "method": method,
                    "query_count": len(subset_ids),
                    "hits_at_20": sum(hit_at_k(by_id[query_id], args.k) for query_id in subset_ids),
                    "recall_at_20": statistics.fmean(
                        float(hit_at_k(by_id[query_id], args.k)) for query_id in subset_ids
                    ),
                    "mrr_at_20": statistics.fmean(
                        reciprocal_rank_at_k(by_id[query_id], args.k) for query_id in subset_ids
                    ),
                }
            )

    wins_losses = []
    for kg_method in ("AST-KG+CrossEncoder(static-K)", "AST-KG+CrossEncoder+StagedConfidence"):
        for query_id in query_ids:
            vector_row = methods["Vector-only"][query_id]
            kg_row = methods[kg_method][query_id]
            vector_hit = hit_at_k(vector_row, args.k)
            kg_hit = hit_at_k(kg_row, args.k)
            if vector_hit == kg_hit:
                continue
            wins_losses.append(
                {
                    "comparison": f"{kg_method} vs Vector-only",
                    "outcome": "rescued_by_kg" if kg_hit else "lost_by_kg",
                    "query_id": query_id,
                    "question": vector_row["question"],
                    "expected_file": vector_row["expected_file"],
                    "expected_line": vector_row["expected_line"],
                    "vector_rank": vector_row.get("gold_rank"),
                    "static_kg_rank": methods["AST-KG+CrossEncoder(static-K)"][query_id].get("gold_rank"),
                    "staged_kg_rank": methods["AST-KG+CrossEncoder+StagedConfidence"][query_id].get("gold_rank"),
                    "kg_candidate_pool_contains_gold_before_rerank": kg_row.get(
                        "kg_candidate_pool_contains_gold_before_rerank"
                    ),
                    "kg_rank_before_rerank": kg_row.get("kg_rank_before_rerank"),
                    "kg_rank_after_rerank_full": kg_row.get("kg_rank_after_rerank_full"),
                    "kg_pool_size": kg_row.get("kg_pool_size"),
                    "diagnosis": pool_diagnosis(kg_row, args.k),
                }
            )

    rank_summaries = []
    rank_examples = []
    for method_b in ("Vector+CrossEncoder", "AST-KG+CrossEncoder(static-K)"):
        summary, examples = rank_change_analysis(
            methods,
            "Vector-only",
            method_b,
            query_ids,
            args.k,
            args.rank_examples,
        )
        rank_summaries.append(summary)
        rank_examples.extend(examples)

    static_rows = methods["AST-KG+CrossEncoder(static-K)"]
    cap_affected = [
        {
            "query_id": query_id,
            "question": static_rows[query_id]["question"],
            "expected": static_rows[query_id]["expected"],
            "raw_gold_rank": static_rows[query_id]["gold_rank"],
            "kg_pool_size": static_rows[query_id]["kg_pool_size"],
        }
        for query_id in query_ids
        if static_rows[query_id].get("gold_rank") is not None
        and int(static_rows[query_id]["gold_rank"]) > args.k
    ]
    accounting_diagnostic = {
        "runner_reported_static_hits": sum(row.get("kg_ok", False) for row in static_rows.values()),
        "literal_static_hits_at_20": query_metrics["AST-KG+CrossEncoder(static-K)"]["hits_at_20"],
        "runner_reported_static_mrr": statistics.fmean(row["reciprocal_rank"] for row in static_rows.values()),
        "literal_static_mrr_at_20": query_metrics["AST-KG+CrossEncoder(static-K)"]["query_mrr_at_20"],
        "cause": "When KG pool size exceeds rerank_kg_max=200, reranking/truncation is skipped and the evaluator counts any raw-pool gold rank as a hit.",
        "gold_ranks_beyond_20": cap_affected,
        "queries_with_pool_above_rerank_cap": sum(
            int(row.get("kg_pool_size", 0) > 200) for row in static_rows.values()
        ),
    }

    write_csv(output_dir / "per_query_results.csv", long_rows)
    write_csv(output_dir / "paired_mcnemar.csv", paired_rows)
    write_csv(output_dir / "target_level_metrics.csv", target_rows)
    target_summary_rows = [{"method": method, **metrics} for method, metrics in target_balanced.items()]
    write_csv(output_dir / "target_balanced_metrics.csv", target_summary_rows)
    write_csv(output_dir / "cluster_bootstrap_cis.csv", bootstrap_rows)
    write_csv(output_dir / "subset_metrics.csv", subset_rows)
    write_csv(output_dir / "kg_wins_losses.csv", wins_losses)
    write_csv(output_dir / "rank_change_summary.csv", rank_summaries)
    write_csv(output_dir / "rank_change_examples.csv", rank_examples)

    report = {
        "analysis_configuration": {
            "benchmark": str(Path(args.benchmark)),
            "query_count": len(query_ids),
            "unique_target_count": len(targets),
            "k": args.k,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_estimand": "query-weighted metric difference with target-cluster resampling",
        },
        "query_and_target_balanced_metrics": target_balanced,
        "paired_mcnemar": paired_rows,
        "cluster_bootstrap": bootstrap_rows,
        "kg_wins_losses": wins_losses,
        "rank_change_summary": rank_summaries,
        "rank_change_examples": rank_examples,
        "subset_metrics": subset_rows,
        "static_k_accounting_diagnostic": accounting_diagnostic,
    }
    write_json(output_dir / "statistical_analysis.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
