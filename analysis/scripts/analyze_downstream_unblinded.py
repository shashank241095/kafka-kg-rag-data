#!/usr/bin/env python3
"""Unblind and analyze the frozen 100-query downstream AST experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "results" / "ast_downstream_gpt41mini_v1"
DEFAULT_SCORE_PATH = EXPERIMENT_DIR / "scoring_blinded_completed.csv"
DEFAULT_MAPPING_PATH = EXPERIMENT_DIR / "blinding_map_private.json"
DEFAULT_RAW_GENERATIONS_PATH = EXPERIMENT_DIR / "raw_generations.json"
DEFAULT_RETRIEVAL_DIR = PROJECT_ROOT / "scripts" / "results_ast_315_paper_config" / "per_query"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "unblinded_analysis"

VECTOR = "Vector-only"
STATIC = "AST-KG + Cross-Encoder static-K"
STAGED = "AST-KG + Cross-Encoder + Staged Confidence"
METHODS = (VECTOR, STATIC, STAGED)

MAPPING_VALUE_TO_METHOD = {
    "Vector-only": VECTOR,
    "KG+CrossEncoder(static-K)": STATIC,
    "KG+CrossEncoder+StagedConfidence": STAGED,
}

RETRIEVAL_FILES = {
    VECTOR: "vector_only.json",
    STATIC: "kg_crossencoder_static_k.json",
    STAGED: "kg_crossencoder_stagedconfidence.json",
}

METRICS = {
    "root_cause_correctness": {
        "field": "judge_root_cause_correct",
        "label": "Root-cause correctness",
        "one_means": "correct",
        "higher_is_better": True,
    },
    "evidence_grounding": {
        "field": "judge_evidence_grounded",
        "label": "Evidence grounding",
        "one_means": "grounded",
        "higher_is_better": True,
    },
    "hallucination": {
        "field": "judge_hallucination",
        "label": "Hallucination",
        "one_means": "hallucinates",
        "higher_is_better": False,
    },
}

SCORE_FIELDS = tuple(metric["field"] for metric in METRICS.values())
PAIRWISE_COMPARISONS = (
    (STATIC, VECTOR),
    (STAGED, VECTOR),
    (STATIC, STAGED),
)
BOOTSTRAP_COMPARISONS = (
    (STATIC, VECTOR),
    (STAGED, VECTOR),
    (STAGED, STATIC),
)

LEGACY_RESULTS = {
    VECTOR: {
        "root_cause_correctness_percentage": 50.0,
        "evidence_grounding_percentage": 96.0,
        "hallucination_percentage": 4.0,
    },
    STATIC: {
        "legacy_label": "KG + CrossEncoder",
        "root_cause_correctness_percentage": 62.0,
        "evidence_grounding_percentage": 100.0,
        "hallucination_percentage": 0.0,
    },
    STAGED: {
        "legacy_label": "KG + StagedConfidence",
        "root_cause_correctness_percentage": 61.0,
        "evidence_grounding_percentage": 98.0,
        "hallucination_percentage": 2.0,
    },
}

LEGACY_CAVEATS = (
    "graph construction changed from regex to AST",
    "the indexed corpus changed",
    "the benchmark changed from 321 to an audited 315 queries",
    "the new 100-query sample is derived from the revised 103-target benchmark",
    "the new scoring is blinded",
)


class ValidationError(RuntimeError):
    """Raised when a frozen-artifact invariant fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=str(DEFAULT_SCORE_PATH))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--raw-generations", default=str(DEFAULT_RAW_GENERATIONS_PATH))
    parser.add_argument("--retrieval-dir", default=str(DEFAULT_RETRIEVAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValidationError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_fingerprint(rows: Sequence[Mapping[str, str]], method_field: str) -> str:
    values = [
        {
            "query_id": row["query_id"],
            "blinded_method": row[method_field],
            **{field: row[field] for field in SCORE_FIELDS},
        }
        for row in rows
    ]
    values.sort(key=lambda row: (row["query_id"], row["blinded_method"]))
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_mcnemar_p(n10: int, n01: int) -> float:
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, value) for value in range(min(n10, n01) + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci(
    differences: Sequence[int], seed: int, replicates: int
) -> Tuple[float, float]:
    if not differences:
        raise ValueError("Paired bootstrap requires observations")
    rng = random.Random(seed)
    count = len(differences)
    estimates = []
    for _ in range(replicates):
        estimates.append(sum(differences[rng.randrange(count)] for _ in range(count)) / count)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def format_percentage(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def format_p(value: float) -> str:
    return f"{value:.6g}"


def validate_and_unblind(
    score_path: Path, mapping_path: Path
) -> Tuple[List[str], List[Dict[str, str]], Dict[str, Any]]:
    fieldnames, scored_rows = read_csv(score_path)
    required = {"query_id", "method", "question", "expected_file", "expected_line", *SCORE_FIELDS}
    missing_columns = sorted(required - set(fieldnames))
    if missing_columns:
        raise ValidationError(f"Missing score columns: {missing_columns}")
    if len(scored_rows) != 300:
        raise ValidationError(f"Expected exactly 300 scored answers, found {len(scored_rows)}")

    by_query: MutableMapping[str, List[Dict[str, str]]] = defaultdict(list)
    for row in scored_rows:
        for field in SCORE_FIELDS:
            if row[field] not in {"0", "1"}:
                raise ValidationError(
                    f"Invalid {field}={row[field]!r} for {row['query_id']} {row['method']}"
                )
        by_query[row["query_id"]].append(row)
    if len(by_query) != 100:
        raise ValidationError(f"Expected exactly 100 unique query IDs, found {len(by_query)}")
    invalid_counts = {query_id: len(rows) for query_id, rows in by_query.items() if len(rows) != 3}
    if invalid_counts:
        raise ValidationError(f"Expected exactly three answers per query: {invalid_counts}")

    mapping_payload = read_json(mapping_path)
    mapping_rows = mapping_payload.get("mappings")
    if not isinstance(mapping_rows, list) or len(mapping_rows) != 100:
        raise ValidationError("Private mapping must contain exactly 100 mappings")
    mapping_by_query: Dict[str, Dict[str, str]] = {}
    for mapping_row in mapping_rows:
        query_id = mapping_row["query_id"]
        if query_id in mapping_by_query:
            raise ValidationError(f"Duplicate private mapping for {query_id}")
        blind_to_internal = mapping_row["mapping"]
        if set(blind_to_internal) != {"Method A", "Method B", "Method C"}:
            raise ValidationError(f"Incomplete blinded labels for {query_id}")
        if set(blind_to_internal.values()) != set(MAPPING_VALUE_TO_METHOD):
            raise ValidationError(f"Invalid or duplicate actual methods for {query_id}")
        mapping_by_query[query_id] = {
            blind: MAPPING_VALUE_TO_METHOD[internal]
            for blind, internal in blind_to_internal.items()
        }

    score_query_ids = set(by_query)
    mapping_query_ids = set(mapping_by_query)
    if score_query_ids != mapping_query_ids:
        missing_mapping = sorted(score_query_ids - mapping_query_ids)
        extra_mapping = sorted(mapping_query_ids - score_query_ids)
        raise ValidationError(
            f"Score/mapping query mismatch; missing mappings={missing_mapping}, extra mappings={extra_mapping}"
        )

    unblinded_rows: List[Dict[str, str]] = []
    for source_row in scored_rows:
        query_id = source_row["query_id"]
        blinded_method = source_row["method"]
        if blinded_method not in mapping_by_query[query_id]:
            raise ValidationError(f"Missing mapping for {query_id} {blinded_method}")
        row = dict(source_row)
        row["blinded_method"] = blinded_method
        row["method"] = mapping_by_query[query_id][blinded_method]
        unblinded_rows.append(row)

    for query_id, rows in by_query.items():
        actual = {
            mapping_by_query[query_id][row["method"]]
            for row in rows
        }
        if actual != set(METHODS):
            raise ValidationError(f"Query {query_id} does not map one-to-one to the three actual methods")

    source_fingerprint = score_fingerprint(scored_rows, "method")
    unblinded_fingerprint = score_fingerprint(unblinded_rows, "blinded_method")
    if source_fingerprint != unblinded_fingerprint:
        raise ValidationError("Score values changed during unblinding")

    unblinded_fieldnames = list(fieldnames)
    method_index = unblinded_fieldnames.index("method")
    unblinded_fieldnames.insert(method_index + 1, "blinded_method")
    validation = {
        "status": "passed",
        "total_answers": len(unblinded_rows),
        "unique_query_ids": len(by_query),
        "answers_per_query": 3,
        "method_assignments_per_query": list(METHODS),
        "missing_mappings": 0,
        "duplicate_method_assignments_within_query": 0,
        "invalid_or_missing_scores": 0,
        "score_values_unchanged": True,
        "score_value_fingerprint_sha256_before": source_fingerprint,
        "score_value_fingerprint_sha256_after": unblinded_fingerprint,
        "blinded_scoring_artifact_sha256": sha256_file(score_path),
        "private_mapping_artifact_sha256": sha256_file(mapping_path),
    }
    return unblinded_fieldnames, unblinded_rows, validation


def index_unblinded(rows: Sequence[Mapping[str, str]]) -> Dict[str, Dict[str, Mapping[str, str]]]:
    indexed: Dict[str, Dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        query_id = row["query_id"]
        method = row["method"]
        if method in indexed[query_id]:
            raise ValidationError(f"Duplicate unblinded assignment for {query_id} {method}")
        indexed[query_id][method] = row
    for query_id, methods in indexed.items():
        if set(methods) != set(METHODS):
            raise ValidationError(f"Incomplete unblinded methods for {query_id}")
        signatures = {
            (row["question"], row["expected_file"], row["expected_line"])
            for row in methods.values()
        }
        if len(signatures) != 1:
            raise ValidationError(f"Question/target mismatch across methods for {query_id}")
    return dict(indexed)


def aggregate_results(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    csv_rows = []
    json_rows: Dict[str, Any] = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        if len(subset) != 100:
            raise ValidationError(f"Expected 100 answers for {method}, found {len(subset)}")
        csv_row: Dict[str, Any] = {"method": method, "n": len(subset)}
        metric_results = {}
        for metric_name, configuration in METRICS.items():
            count = sum(int(row[configuration["field"]]) for row in subset)
            rate = count / len(subset)
            csv_row[f"{metric_name}_count"] = count
            csv_row[f"{metric_name}_denominator"] = len(subset)
            csv_row[f"{metric_name}_percentage"] = count * 100.0 / len(subset)
            metric_results[metric_name] = {
                "count": count,
                "denominator": len(subset),
                "rate": rate,
                "percentage": count * 100.0 / len(subset),
            }
        csv_rows.append(csv_row)
        json_rows[method] = {"n": len(subset), "metrics": metric_results}
    return csv_rows, json_rows


def calculate_confidence_intervals(
    aggregates: Mapping[str, Any], indexed: Mapping[str, Mapping[str, Mapping[str, str]]], seed: int, replicates: int
) -> Dict[str, Any]:
    method_intervals: Dict[str, Any] = {}
    for method in METHODS:
        method_intervals[method] = {}
        for metric_name in METRICS:
            metric = aggregates[method]["metrics"][metric_name]
            lower, upper = wilson_interval(metric["count"], metric["denominator"])
            method_intervals[method][metric_name] = {
                **metric,
                "interval": "Wilson score interval",
                "confidence_level": 0.95,
                "lower": lower,
                "upper": upper,
                "lower_percentage": 100.0 * lower,
                "upper_percentage": 100.0 * upper,
            }

    query_ids = sorted(indexed)
    bootstrap_results = []
    for metric_name, configuration in METRICS.items():
        field = configuration["field"]
        for method_a, method_b in BOOTSTRAP_COMPARISONS:
            differences = [
                int(indexed[query_id][method_a][field]) - int(indexed[query_id][method_b][field])
                for query_id in query_ids
            ]
            point = sum(differences) / len(differences)
            lower, upper = paired_bootstrap_ci(differences, seed, replicates)
            bootstrap_results.append(
                {
                    "metric": metric_name,
                    "difference": f"{method_a} - {method_b}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "point_estimate": point,
                    "point_estimate_percentage_points": 100.0 * point,
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "ci_95_lower_percentage_points": 100.0 * lower,
                    "ci_95_upper_percentage_points": 100.0 * upper,
                    "bootstrap_seed": seed,
                    "bootstrap_replicates": replicates,
                    "resampling_unit": "query",
                    "interval": "paired percentile bootstrap",
                    "interpretation_note": (
                        "Positive is better for correctness/grounding; positive is worse for hallucination."
                    ),
                }
            )
    return {
        "binomial_intervals": {
            "interval": "95% Wilson score interval",
            "methods": method_intervals,
        },
        "paired_bootstrap_intervals": bootstrap_results,
    }


def paired_tests(indexed: Mapping[str, Mapping[str, Mapping[str, str]]]) -> Dict[str, Any]:
    query_ids = sorted(indexed)
    by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for metric_name, configuration in METRICS.items():
        field = configuration["field"]
        comparisons = []
        for method_a, method_b in PAIRWISE_COMPARISONS:
            outcomes = Counter(
                (
                    int(indexed[query_id][method_a][field]),
                    int(indexed[query_id][method_b][field]),
                )
                for query_id in query_ids
            )
            n11 = outcomes[(1, 1)]
            n10 = outcomes[(1, 0)]
            n01 = outcomes[(0, 1)]
            n00 = outcomes[(0, 0)]
            method_a_count = n11 + n10
            method_b_count = n11 + n01
            rate_a = method_a_count / len(query_ids)
            rate_b = method_b_count / len(query_ids)
            difference = (method_a_count - method_b_count) / len(query_ids)
            difference_percentage_points = (
                (method_a_count - method_b_count) * 100.0 / len(query_ids)
            )
            result: Dict[str, Any] = {
                "method_a": method_a,
                "method_b": method_b,
                "both_1": n11,
                "n10_a1_b0": n10,
                "n01_a0_b1": n01,
                "both_0": n00,
                "method_a_count": method_a_count,
                "method_b_count": method_b_count,
                "method_a_rate": rate_a,
                "method_b_rate": rate_b,
                "difference_a_minus_b": difference,
                "difference_a_minus_b_percentage_points": difference_percentage_points,
                "absolute_difference_percentage_points": abs(difference_percentage_points),
                "discordant_pairs": n10 + n01,
                "exact_two_sided_mcnemar_p": exact_mcnemar_p(n10, n01),
            }
            if metric_name == "hallucination":
                result.update(
                    {
                        "both_hallucinate": n11,
                        "a_only_hallucinates": n10,
                        "b_only_hallucinates": n01,
                        "neither_hallucinates": n00,
                        "lower_rate_is_better": True,
                    }
                )
            else:
                result.update(
                    {
                        "both_positive": n11,
                        "a_positive_b_negative": n10,
                        "a_negative_b_positive": n01,
                        "both_negative": n00,
                    }
                )
            comparisons.append(result)
        by_metric[metric_name] = comparisons
    return {
        "test": "exact two-sided McNemar test (exact binomial on discordant pairs)",
        "sample_size": len(query_ids),
        "metrics": by_metric,
    }


def load_retrieval_rows(retrieval_dir: Path) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    result: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for method, filename in RETRIEVAL_FILES.items():
        payload = read_json(retrieval_dir / filename)
        rows = payload.get("queries")
        if not isinstance(rows, list):
            raise ValidationError(f"Invalid retrieval artifact: {filename}")
        by_id = {row["query_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValidationError(f"Duplicate query IDs in retrieval artifact: {filename}")
        result[method] = by_id
    return result


def validate_raw_generations(
    raw_path: Path,
    indexed: Mapping[str, Mapping[str, Mapping[str, str]]],
    retrieval: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    payload = read_json(raw_path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 300:
        raise ValidationError("Raw generations must contain exactly 300 rows")
    raw_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (row["query_id"], MAPPING_VALUE_TO_METHOD.get(row["retrieval_method"], row["retrieval_method"]))
        if key in raw_by_key:
            raise ValidationError(f"Duplicate raw generation: {key}")
        raw_by_key[key] = row

    for query_id, methods in indexed.items():
        for method, score_row in methods.items():
            key = (query_id, method)
            if key not in raw_by_key:
                raise ValidationError(f"Missing raw generation: {key}")
            raw_row = raw_by_key[key]
            retrieval_row = retrieval[method].get(query_id)
            if retrieval_row is None:
                raise ValidationError(f"Missing corrected retrieval row: {key}")
            expected = raw_row["expected"]
            signature = (score_row["question"], score_row["expected_file"], int(score_row["expected_line"]))
            raw_signature = (raw_row["question"], expected["file"], int(expected["line"]))
            retrieval_signature = (
                retrieval_row["question"],
                retrieval_row["expected_file"],
                int(retrieval_row["expected_line"]),
            )
            if signature != raw_signature or signature != retrieval_signature:
                raise ValidationError(f"Question/target mismatch in frozen joins: {key}")
            raw_rank = raw_row.get("final_gold_rank")
            retrieval_rank = retrieval_row.get("gold_rank")
            if raw_rank != retrieval_rank:
                raise ValidationError(
                    f"Frozen rank mismatch for {key}: generation={raw_rank}, retrieval={retrieval_rank}"
                )
            if bool(raw_row.get("gold_in_evidence_top5")) != (
                retrieval_rank is not None and int(retrieval_rank) <= 5
            ):
                raise ValidationError(f"Frozen top-5 flag mismatch for {key}")
    return raw_by_key


def query_level_rows(
    indexed: Mapping[str, Mapping[str, Mapping[str, str]]],
    retrieval: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    results = []
    rca_field = METRICS["root_cause_correctness"]["field"]
    for query_id in sorted(indexed):
        methods = indexed[query_id]
        vector_rca = int(methods[VECTOR][rca_field])
        static_rca = int(methods[STATIC][rca_field])
        staged_rca = int(methods[STAGED][rca_field])
        if vector_rca == static_rca and vector_rca == staged_rca:
            continue

        def outcome(kg_value: int) -> str:
            if vector_rca == 0 and kg_value == 1:
                return "rescue"
            if vector_rca == 1 and kg_value == 0:
                return "loss"
            return "unchanged"

        reference = methods[VECTOR]
        row: Dict[str, Any] = {
            "query_id": query_id,
            "query_text": reference["question"],
            "expected_file": reference["expected_file"],
            "expected_line": int(reference["expected_line"]),
            "expected_target": f"{reference['expected_file']}:{reference['expected_line']}",
            "static_vs_vector_outcome": outcome(static_rca),
            "staged_vs_vector_outcome": outcome(staged_rca),
        }
        for prefix, method in (("vector", VECTOR), ("static", STATIC), ("staged", STAGED)):
            score_row = methods[method]
            retrieval_row = retrieval[method][query_id]
            row[f"{prefix}_root_cause_correct"] = int(score_row["judge_root_cause_correct"])
            row[f"{prefix}_evidence_grounded"] = int(score_row["judge_evidence_grounded"])
            row[f"{prefix}_hallucination"] = int(score_row["judge_hallucination"])
            row[f"{prefix}_gold_rank"] = retrieval_row.get("gold_rank")
            row[f"{prefix}_gold_in_top5"] = (
                retrieval_row.get("gold_rank") is not None and int(retrieval_row["gold_rank"]) <= 5
            )
            row[f"{prefix}_gold_in_top20"] = (
                retrieval_row.get("gold_rank") is not None and int(retrieval_row["gold_rank"]) <= 20
            )
            row[f"{prefix}_top5_evidence"] = score_row["top5_evidence"]
        results.append(row)
    return results


def retrieval_vs_downstream(
    indexed: Mapping[str, Mapping[str, Mapping[str, str]]],
    retrieval: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    rca_field = METRICS["root_cause_correctness"]["field"]
    result: Dict[str, Any] = {
        "scope": "100 sampled downstream queries joined to authoritative corrected frozen retrieval rows",
        "causal_warning": "Descriptive contingency summaries only; no causal inference is claimed.",
        "methods": {},
    }
    for method in METHODS:
        top5 = Counter()
        top20 = Counter()
        rank_counts = Counter()
        for query_id, methods in indexed.items():
            rca = int(methods[method][rca_field])
            rank = retrieval[method][query_id].get("gold_rank")
            in_top5 = rank is not None and int(rank) <= 5
            in_top20 = rank is not None and int(rank) <= 20
            top5[(in_top5, rca)] += 1
            top20[(in_top20, rca)] += 1
            rank_counts["gold_in_top5"] += int(in_top5)
            rank_counts["gold_in_top20"] += int(in_top20)
            rank_counts["root_cause_correct"] += rca
        result["methods"][method] = {
            "n": len(indexed),
            "gold_in_top5_count": rank_counts["gold_in_top5"],
            "gold_in_top20_count": rank_counts["gold_in_top20"],
            "root_cause_correct_count": rank_counts["root_cause_correct"],
            "top5_contingency": {
                "gold_in_top5_and_rca_correct": top5[(True, 1)],
                "gold_in_top5_and_rca_incorrect": top5[(True, 0)],
                "gold_outside_top5_and_rca_correct": top5[(False, 1)],
                "gold_outside_top5_and_rca_incorrect": top5[(False, 0)],
            },
            "top20_contingency": {
                "gold_in_top20_and_rca_correct": top20[(True, 1)],
                "gold_in_top20_and_rca_incorrect": top20[(True, 0)],
                "gold_outside_top20_and_rca_correct": top20[(False, 1)],
                "gold_outside_top20_and_rca_incorrect": top20[(False, 0)],
            },
        }
    return result


def comparison_lookup(paired: Mapping[str, Any], metric: str, method_a: str, method_b: str) -> Mapping[str, Any]:
    for row in paired["metrics"][metric]:
        if row["method_a"] == method_a and row["method_b"] == method_b:
            return row
    raise KeyError((metric, method_a, method_b))


def bootstrap_lookup(intervals: Mapping[str, Any], metric: str, method_a: str, method_b: str) -> Mapping[str, Any]:
    for row in intervals["paired_bootstrap_intervals"]:
        if row["metric"] == metric and row["method_a"] == method_a and row["method_b"] == method_b:
            return row
    raise KeyError((metric, method_a, method_b))


def list_query_ids(query_rows: Sequence[Mapping[str, Any]], field: str, value: str) -> List[str]:
    return [row["query_id"] for row in query_rows if row[field] == value]


def conclusion_for_improvement(label: str, comparison: Mapping[str, Any], higher_is_better: bool = True) -> str:
    difference = comparison["difference_a_minus_b_percentage_points"]
    p_value = comparison["exact_two_sided_mcnemar_p"]
    favorable = difference > 0 if higher_is_better else difference < 0
    if p_value < 0.05 and favorable:
        return f"Yes. {label} changed the rate by {difference:+.1f} pp; exact McNemar p={format_p(p_value)}."
    if p_value < 0.05 and not favorable:
        return f"No. {label} changed the rate unfavorably by {difference:+.1f} pp; exact McNemar p={format_p(p_value)}."
    return f"No statistically significant difference. The observed change was {difference:+.1f} pp; exact McNemar p={format_p(p_value)}."


def build_markdown_summary(
    validation: Mapping[str, Any],
    aggregates: Mapping[str, Any],
    paired: Mapping[str, Any],
    intervals: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
    retrieval_relationship: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    lines = [
        "# Unblinded downstream analysis",
        "",
        "## Scope and provenance",
        "",
        "This report analyzes the frozen 100-query AST downstream experiment. Unblinding passed all structural checks: "
        f"{validation['total_answers']} scored answers, {validation['unique_query_ids']} queries, exactly three distinct "
        "retrieval conditions per query, no missing mappings, and an identical before/after score-value fingerprint.",
        "",
        "The final reported labels were manually assigned by the author under blinded Method A/B/C "
        "identities using the fixed rubric; see the 2026-09-11 correction at the top of `AI_SCORING_NOTICE.md`. "
        "No score, generation, retrieval result, benchmark row, private mapping, or legacy result was modified by this analysis.",
        "",
        "## New AST downstream results",
        "",
        "| Method | Root-cause correctness | Evidence grounding | Hallucination rate |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = aggregates[method]["metrics"]
        lines.append(
            f"| {method} | {metrics['root_cause_correctness']['count']}/100 "
            f"({metrics['root_cause_correctness']['percentage']:.1f}%) | "
            f"{metrics['evidence_grounding']['count']}/100 ({metrics['evidence_grounding']['percentage']:.1f}%) | "
            f"{metrics['hallucination']['count']}/100 ({metrics['hallucination']['percentage']:.1f}%) |"
        )

    lines += [
        "",
        "Hallucination = 1 means an unsupported causal claim was introduced; lower is better.",
        "",
        "### 95% Wilson confidence intervals",
        "",
        "| Method | Root-cause correctness | Evidence grounding | Hallucination |",
        "|---|---:|---:|---:|",
    ]
    method_intervals = intervals["binomial_intervals"]["methods"]
    for method in METHODS:
        values = []
        for metric_name in METRICS:
            result = method_intervals[method][metric_name]
            values.append(
                f"{result['percentage']:.1f}% [{result['lower_percentage']:.1f}%, {result['upper_percentage']:.1f}%]"
            )
        lines.append(f"| {method} | " + " | ".join(values) + " |")

    lines += [
        "",
        "## Orientation against legacy-parser results",
        "",
        "| Method | New AST RCA | Legacy RCA | New AST grounding | Legacy grounding | New AST hallucination | Legacy hallucination |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        new = aggregates[method]["metrics"]
        old = LEGACY_RESULTS[method]
        legacy_label = old.get("legacy_label", method)
        display = method if legacy_label == method else f"{method} (legacy: {legacy_label})"
        lines.append(
            f"| {display} | {new['root_cause_correctness']['percentage']:.1f}% | "
            f"{old['root_cause_correctness_percentage']:.1f}% | {new['evidence_grounding']['percentage']:.1f}% | "
            f"{old['evidence_grounding_percentage']:.1f}% | {new['hallucination']['percentage']:.1f}% | "
            f"{old['hallucination_percentage']:.1f}% |"
        )
    lines += [
        "",
        "This is not a strict apples-to-apples comparison and changes relative to the legacy values are not statistical evidence. "
        "The graph construction changed from regex to AST; the indexed corpus changed; the benchmark changed from 321 to an "
        "audited 315 queries; this 100-query sample comes from the revised 103-target benchmark; and the new scoring is blinded.",
    ]

    for metric_name, configuration in METRICS.items():
        lines += [
            "",
            f"## Paired {configuration['label']}",
            "",
        ]
        if metric_name == "hallucination":
            lines += [
                "| Comparison (A vs B) | Both hallucinate | A only | B only | Neither | A-B (pp) | Absolute difference (pp) | Exact p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        else:
            lines += [
                "| Comparison (A vs B) | Both 1 | n10 (A=1,B=0) | n01 (A=0,B=1) | Both 0 | A-B (pp) | Absolute difference (pp) | Exact p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        for result in paired["metrics"][metric_name]:
            lines.append(
                f"| {result['method_a']} vs {result['method_b']} | {result['both_1']} | "
                f"{result['n10_a1_b0']} | {result['n01_a0_b1']} | {result['both_0']} | "
                f"{result['difference_a_minus_b_percentage_points']:+.1f} | "
                f"{result['absolute_difference_percentage_points']:.1f} | "
                f"{format_p(result['exact_two_sided_mcnemar_p'])} |"
            )
        if metric_name == "hallucination":
            lines.append("")
            static_vector = comparison_lookup(paired, metric_name, STATIC, VECTOR)
            staged_vector = comparison_lookup(paired, metric_name, STAGED, VECTOR)
            static_staged = comparison_lookup(paired, metric_name, STATIC, STAGED)
            lines.append(
                "For hallucination, a negative A-B difference means A has fewer hallucinations and is better. "
                f"Static had {static_vector['difference_a_minus_b_percentage_points']:+.1f} pp versus vector; "
                f"staged had {staged_vector['difference_a_minus_b_percentage_points']:+.1f} pp versus vector; "
                f"and static had {static_staged['difference_a_minus_b_percentage_points']:+.1f} pp versus staged."
            )

    lines += [
        "",
        "## Paired bootstrap 95% confidence intervals",
        "",
        "Fixed seed 42, 10,000 query-level paired bootstrap replicates. Values are percentage-point differences.",
        "",
        "| Metric | Difference | Point estimate | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for result in intervals["paired_bootstrap_intervals"]:
        lines.append(
            f"| {METRICS[result['metric']]['label']} | {result['difference']} | "
            f"{result['point_estimate_percentage_points']:+.1f} | "
            f"[{result['ci_95_lower_percentage_points']:+.1f}, {result['ci_95_upper_percentage_points']:+.1f}] |"
        )

    static_rescues = list_query_ids(query_rows, "static_vs_vector_outcome", "rescue")
    static_losses = list_query_ids(query_rows, "static_vs_vector_outcome", "loss")
    staged_rescues = list_query_ids(query_rows, "staged_vs_vector_outcome", "rescue")
    staged_losses = list_query_ids(query_rows, "staged_vs_vector_outcome", "loss")
    lines += [
        "",
        "## Root-cause correctness query-level changes",
        "",
        f"- Static KG rescues ({len(static_rescues)}): " + (", ".join(static_rescues) or "none"),
        "",
        f"- Static KG losses ({len(static_losses)}): " + (", ".join(static_losses) or "none"),
        "",
        f"- Staged KG rescues ({len(staged_rescues)}): " + (", ".join(staged_rescues) or "none"),
        "",
        f"- Staged KG losses ({len(staged_losses)}): " + (", ".join(staged_losses) or "none"),
        "",
        "The companion `query_level_comparisons.csv` contains query text, expected target, all three correctness labels, "
        "retrieval ranks, top-5/top-20 flags, and all three top-5 evidence lists for every changed query.",
        "",
        "## Retrieval rank versus downstream correctness",
        "",
        "These are descriptive associations, not causal estimates.",
        "",
        "| Method | Gold top 5 | Gold top 20 | RCA correct | Top5+correct | Top5+incorrect | Outside top5+correct | Outside top5+incorrect |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        relationship = retrieval_relationship["methods"][method]
        contingency = relationship["top5_contingency"]
        lines.append(
            f"| {method} | {relationship['gold_in_top5_count']}/100 | {relationship['gold_in_top20_count']}/100 | "
            f"{relationship['root_cause_correct_count']}/100 | {contingency['gold_in_top5_and_rca_correct']} | "
            f"{contingency['gold_in_top5_and_rca_incorrect']} | "
            f"{contingency['gold_outside_top5_and_rca_correct']} | "
            f"{contingency['gold_outside_top5_and_rca_incorrect']} |"
        )

    lines += [
        "",
        "| Method | Top20+correct | Top20+incorrect | Outside top20+correct | Outside top20+incorrect |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        contingency = retrieval_relationship["methods"][method]["top20_contingency"]
        lines.append(
            f"| {method} | {contingency['gold_in_top20_and_rca_correct']} | "
            f"{contingency['gold_in_top20_and_rca_incorrect']} | "
            f"{contingency['gold_outside_top20_and_rca_correct']} | "
            f"{contingency['gold_outside_top20_and_rca_incorrect']} |"
        )

    static_rca = comparison_lookup(paired, "root_cause_correctness", STATIC, VECTOR)
    staged_rca = comparison_lookup(paired, "root_cause_correctness", STAGED, VECTOR)
    static_vs_staged_rca = comparison_lookup(paired, "root_cause_correctness", STATIC, STAGED)
    static_ground = comparison_lookup(paired, "evidence_grounding", STATIC, VECTOR)
    staged_ground = comparison_lookup(paired, "evidence_grounding", STAGED, VECTOR)
    static_hall = comparison_lookup(paired, "hallucination", STATIC, VECTOR)
    staged_hall = comparison_lookup(paired, "hallucination", STAGED, VECTOR)
    staged_static_rca_bootstrap = bootstrap_lookup(
        intervals, "root_cause_correctness", STAGED, STATIC
    )

    staged_minus_static = -static_vs_staged_rca["difference_a_minus_b_percentage_points"]
    lines += [
        "",
        "## Evidence-based conclusions",
        "",
        "1. **Does AST-KG static significantly improve root-cause correctness over vector-only?** "
        + conclusion_for_improvement("Static KG", static_rca),
        "",
        "2. **Does AST-KG staged significantly improve root-cause correctness over vector-only?** "
        + conclusion_for_improvement("Staged KG", staged_rca),
        "",
        "3. **Does staged confidence preserve downstream quality relative to static KG?** "
        f"Staged minus static correctness was {staged_minus_static:+.1f} pp; exact McNemar "
        f"p={format_p(static_vs_staged_rca['exact_two_sided_mcnemar_p'])}; the paired bootstrap 95% CI was "
        f"[{staged_static_rca_bootstrap['ci_95_lower_percentage_points']:+.1f}, "
        f"{staged_static_rca_bootstrap['ci_95_upper_percentage_points']:+.1f}] pp. "
        "The exact test did not cross 0.05, but the bootstrap interval excludes zero. The evidence is therefore borderline "
        "and method-dependent, and preservation cannot be claimed; this was not an equivalence or non-inferiority test.",
        "",
        "4. **Does AST-KG improve evidence grounding?** Static: "
        + conclusion_for_improvement("Static KG", static_ground)
        + " Staged: "
        + conclusion_for_improvement("Staged KG", staged_ground),
        "",
        "5. **Does AST-KG reduce hallucination?** Static: "
        + conclusion_for_improvement("Static KG", static_hall, higher_is_better=False)
        + " Staged: "
        + conclusion_for_improvement("Staged KG", staged_hall, higher_is_better=False),
        "",
    ]

    significant_rca = any(
        comparison["exact_two_sided_mcnemar_p"] < 0.05
        and comparison["difference_a_minus_b_percentage_points"] > 0
        for comparison in (static_rca, staged_rca)
    )
    if significant_rca:
        structural_conclusion = (
            "Yes, as an association: at least one AST-KG condition significantly improved downstream correctness despite "
            "no full-benchmark Recall@20 improvement. The contingency tables also show correct answers both with and without "
            "the exact target in the top five. This is consistent with useful structural context, but does not establish causality."
        )
    else:
        structural_conclusion = (
            "No statistically reliable evidence from root-cause correctness: neither AST-KG condition significantly improved "
            "downstream correctness. The contingency tables are descriptive and cannot establish a structural-context effect."
        )
    lines += [
        "6. **Given that retrieval Recall@20 did not improve, is there evidence that structural context improves downstream "
        "reasoning anyway?** " + structural_conclusion,
        "",
        "## Reproducibility",
        "",
        f"- Blinded score SHA-256: `{provenance['blinded_scoring_artifact_sha256']}`",
        "",
        f"- Private mapping SHA-256: `{provenance['private_mapping_artifact_sha256']}`",
        "",
        f"- Score-value fingerprint before/after: `{validation['score_value_fingerprint_sha256_before']}`",
        "",
        "- Exact McNemar tests use a two-sided exact binomial test over discordant pairs.",
        "",
        "- Binomial intervals are 95% Wilson intervals.",
        "",
        "- Paired bootstrap intervals use seed 42 and 10,000 query-level replicates.",
        "",
        "- The exact McNemar test is the primary binary paired test. Percentile-bootstrap intervals need not invert that "
        "discrete exact test, which explains the borderline staged-versus-static discrepancy.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    score_path = Path(args.scores).resolve()
    mapping_path = Path(args.mapping).resolve()
    raw_path = Path(args.raw_generations).resolve()
    retrieval_dir = Path(args.retrieval_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    source_hashes_before = {
        "blinded_scoring_artifact_sha256": sha256_file(score_path),
        "private_mapping_artifact_sha256": sha256_file(mapping_path),
        "raw_generations_artifact_sha256": sha256_file(raw_path),
        "retrieval_artifact_sha256": {
            method: sha256_file(retrieval_dir / filename)
            for method, filename in RETRIEVAL_FILES.items()
        },
    }

    unblinded_fieldnames, unblinded_rows, validation = validate_and_unblind(score_path, mapping_path)
    indexed = index_unblinded(unblinded_rows)
    retrieval = load_retrieval_rows(retrieval_dir)
    validate_raw_generations(raw_path, indexed, retrieval)

    aggregate_csv_rows, aggregates = aggregate_results(unblinded_rows)
    paired = paired_tests(indexed)
    intervals = calculate_confidence_intervals(
        aggregates, indexed, args.bootstrap_seed, args.bootstrap_replicates
    )
    query_rows = query_level_rows(indexed, retrieval)
    retrieval_relationship = retrieval_vs_downstream(indexed, retrieval)

    output_dir.mkdir(parents=True, exist_ok=True)
    unblinded_path = output_dir / "unblinded_scores.csv"
    aggregate_csv_path = output_dir / "aggregate_results.csv"
    aggregate_json_path = output_dir / "aggregate_results.json"
    paired_path = output_dir / "paired_tests.json"
    intervals_path = output_dir / "confidence_intervals.json"
    query_path = output_dir / "query_level_comparisons.csv"
    retrieval_path = output_dir / "retrieval_vs_downstream.json"
    summary_path = output_dir / "analysis_summary.md"

    write_csv(unblinded_path, unblinded_fieldnames, unblinded_rows)
    aggregate_fields = [
        "method", "n",
        "root_cause_correctness_count", "root_cause_correctness_denominator", "root_cause_correctness_percentage",
        "evidence_grounding_count", "evidence_grounding_denominator", "evidence_grounding_percentage",
        "hallucination_count", "hallucination_denominator", "hallucination_percentage",
    ]
    write_csv(aggregate_csv_path, aggregate_fields, aggregate_csv_rows)
    aggregate_payload = {
        "analysis": "unblinded downstream AST experiment",
        "validation": validation,
        "new_ast_results": aggregates,
        "legacy_results_for_orientation_only": LEGACY_RESULTS,
        "legacy_comparison_warning": {
            "strictly_comparable": False,
            "do_not_treat_changes_as_statistical_evidence": True,
            "reasons": list(LEGACY_CAVEATS),
        },
        "scoring_provenance": (
            "single-evaluator blinded manual adjudication; see the 2026-09-11 correction in AI_SCORING_NOTICE.md."
        ),
        "source_artifact_hashes": source_hashes_before,
    }
    write_json(aggregate_json_path, aggregate_payload)
    write_json(paired_path, paired)
    write_json(intervals_path, intervals)
    query_fieldnames = list(query_rows[0]) if query_rows else ["query_id"]
    write_csv(query_path, query_fieldnames, query_rows)
    write_json(retrieval_path, retrieval_relationship)
    summary_path.write_text(
        build_markdown_summary(
            validation,
            aggregates,
            paired,
            intervals,
            query_rows,
            retrieval_relationship,
            source_hashes_before,
        ),
        encoding="utf-8",
    )

    source_hashes_after = {
        "blinded_scoring_artifact_sha256": sha256_file(score_path),
        "private_mapping_artifact_sha256": sha256_file(mapping_path),
        "raw_generations_artifact_sha256": sha256_file(raw_path),
        "retrieval_artifact_sha256": {
            method: sha256_file(retrieval_dir / filename)
            for method, filename in RETRIEVAL_FILES.items()
        },
    }
    if source_hashes_after != source_hashes_before:
        raise ValidationError("A frozen source artifact changed during analysis")

    print(json.dumps({
        "status": "complete",
        "validation": validation,
        "output_dir": str(output_dir),
        "artifacts": [
            str(path)
            for path in (
                unblinded_path,
                aggregate_csv_path,
                aggregate_json_path,
                paired_path,
                intervals_path,
                query_path,
                retrieval_path,
                summary_path,
            )
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
