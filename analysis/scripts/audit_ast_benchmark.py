#!/usr/bin/env python3
"""Audit benchmark gold locations against AST Neo4j, Qdrant, and source.

The input benchmark is never modified. When requested, a proposed cleaned
benchmark removes only queries whose labels are confirmed legacy-regex false
positives. Valid logger calls that are outside the current indexing rules stay
in the proposed artifact for an explicit parser-vs-exclusion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_parsing import (  # noqa: E402
    AST_LOG_LEVELS,
    JAVA_CLASS_NODES,
    JAVA_METHOD_NODES,
    SCALA_CLASS_NODES,
    SCALA_METHOD_NODES,
    _captures,
    _first_argument,
    _last_identifier,
    _leading_literal,
    _looks_like_logger,
    _node_key,
    _node_text,
    _parser_for_language,
    parse_source_file_detailed,
)


Target = Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--neo4j-user", required=True)
    parser.add_argument("--neo4j-pass", required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--jira", required=True)
    parser.add_argument("--cleaned-out")
    parser.add_argument("--report-out")
    return parser.parse_args()


def read_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_target(query: dict) -> Target:
    expected = query["expected"]
    return expected["file"], int(expected["line"])


def neo4j_records(driver, repo_name: str) -> Dict[Target, dict]:
    query = """
    MATCH (c:Class)-[:DECLARES]->(m:Method)-[:EMITS]->(l:LogTemplate)
    WHERE l.repo = $repo
    RETURN l.file AS file, l.line AS line, l.level AS level,
           l.template AS template, l.logger AS logger,
           m.fqn AS method_fqn, c.fqn AS class_fqn
    """
    with driver.session() as session:
        rows = session.execute_read(lambda tx: list(tx.run(query, repo=repo_name)))
    return {(row["file"], int(row["line"])): dict(row) for row in rows}


def qdrant_records(client: QdrantClient, collection: str) -> Dict[Target, dict]:
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return {
        (point.payload["file"], int(point.payload["line"])): point.payload
        for point in points
    }


def call_details(node, source: bytes, language: str):
    if language == "java":
        if node.type != "method_invocation":
            return None
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        level = _node_text(name_node, source).lower()
        object_node = node.child_by_field_name("object")
        logger = "log" if object_node is None else _last_identifier(object_node, source)
        logger_like = object_node is None or bool(logger and _looks_like_logger(logger))
        return level, logger, logger_like, node.child_by_field_name("arguments")

    if node.type != "call_expression":
        return None
    function = node.child_by_field_name("function")
    if function is None:
        return None
    if function.type == "field_expression":
        level_node = function.child_by_field_name("field")
        value_node = function.child_by_field_name("value")
        if level_node is None:
            return None
        level = _node_text(level_node, source).lower()
        logger = _last_identifier(value_node, source)
        logger_like = bool(logger and _looks_like_logger(logger))
    elif function.type in {"identifier", "operator_identifier"}:
        level = _node_text(function, source).lower()
        logger = "log"
        logger_like = True
    else:
        return None
    arguments = node.child_by_field_name("arguments")
    if arguments is None:
        arguments = next(
            (child for child in node.named_children if child.type == "arguments"),
            None,
        )
    return level, logger, logger_like, arguments


def has_enclosing_method(node, language: str, anonymous_nodes: Iterable) -> bool:
    method_types = SCALA_METHOD_NODES if language == "scala" else JAVA_METHOD_NODES
    class_types = SCALA_CLASS_NODES if language == "scala" else JAVA_CLASS_NODES
    anonymous_keys = {_node_key(item) for item in anonymous_nodes}
    current = node.parent
    while current is not None:
        if current.type in method_types:
            return True
        if current.type in class_types or _node_key(current) in anonymous_keys:
            return False
        current = current.parent
    return False


class SourceAudit:
    def __init__(self, repo_root: str, relative_file: str):
        self.relative_file = relative_file
        self.path = os.path.join(repo_root, relative_file)
        self.exists = os.path.isfile(self.path)
        self.ast_lines = set()
        self.legacy_lines = set()
        self.ast_parser_error = False
        self.calls_by_line = defaultdict(list)
        self.language = "scala" if relative_file.endswith(".scala") else "java"
        self.source = b""
        self.tree = None
        self.anonymous_nodes = []
        if not self.exists:
            return

        ast = parse_source_file_detailed(repo_root, self.path, "audit", parser="ast")
        legacy = parse_source_file_detailed(repo_root, self.path, "audit", parser="legacy")
        self.ast_parser_error = ast.diagnostics.has_parser_errors
        self.ast_lines = {log.line for log in ast.logs}
        self.legacy_lines = {log.line for log in legacy.logs}
        with open(self.path, "rb") as source_file:
            self.source = source_file.read()
        self.tree = _parser_for_language(self.language).parse(self.source)
        captures = _captures(self.tree, self.language)
        self.anonymous_nodes = captures.get("anonymous_class", [])
        for call in captures.get("call", []):
            self.calls_by_line[call.start_point.row + 1].append(call)

    def classify_missing(self, line: int) -> Tuple[str, str]:
        if not self.exists:
            return "source/version mismatch", "Source file does not exist in the checked Kafka revision."

        relevant_calls = []
        for call in self.calls_by_line.get(line, []):
            details = call_details(call, self.source, self.language)
            if details is None:
                continue
            level, logger, logger_like, arguments = details
            if logger_like and level in AST_LOG_LEVELS:
                relevant_calls.append((call, level, logger, arguments))

        if relevant_calls:
            call, level, logger, arguments = relevant_calls[0]
            if not has_enclosing_method(call, self.language, self.anonymous_nodes):
                return "logger outside method/function", (
                    "Genuine supported logger call, but it has no enclosing method/function AST node."
                )
            literal = _leading_literal(_first_argument(arguments), self.source, self.language)
            if literal is None:
                return "valid logger call unsupported by AST/indexing rules", (
                    "Genuine supported logger call whose first argument is not an indexable string literal."
                )
            return "other", (
                "The source call satisfies AST indexing rules but is absent from the generated stores."
            )

        if line in self.legacy_lines:
            return "legacy regex false positive", (
                "Legacy regex produced a LogTemplate, but no logger invocation exists at this AST line."
            )
        return "source/version mismatch", (
            "No matching logger invocation or legacy extracted log exists at this source location."
        )


def query_signature(query: dict) -> Tuple[str, str, int]:
    relative_file, line = expected_target(query)
    return query["question"], relative_file, line


def source_part(query: dict, signature_to_parts: Dict[Tuple[str, str, int], set]) -> str:
    parts = signature_to_parts.get(query_signature(query), set())
    return next(iter(parts)) if len(parts) == 1 else "unknown"


def composition(queries: Sequence[dict], signature_to_parts: Dict[Tuple[str, str, int], set]) -> dict:
    counts = Counter(source_part(query, signature_to_parts) for query in queries)
    return {
        "query_count": len(queries),
        "unique_target_count": len({expected_target(query) for query in queries}),
        "v2_count": counts["v2"],
        "v3_count": counts["v3"],
        "jira_seeded_count": counts["jira"],
        "unknown_count": counts["unknown"],
    }


def write_json(path: str, value) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def main() -> None:
    args = parse_args()
    benchmark = read_json(args.benchmark)
    original_checksum = sha256(args.benchmark)

    parts = {
        "v2": read_json(args.v2),
        "v3": read_json(args.v3),
        "jira": read_json(args.jira),
    }
    signature_to_parts = defaultdict(set)
    for part, queries in parts.items():
        for query in queries:
            signature_to_parts[query_signature(query)].add(part)

    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_pass),
    )
    neo4j = neo4j_records(driver, args.repo_name)
    driver.close()
    qdrant = qdrant_records(QdrantClient(url=args.qdrant_url), args.collection_name)

    targets = sorted({expected_target(query) for query in benchmark})
    source_audits = {
        relative_file: SourceAudit(args.repo_root, relative_file)
        for relative_file in sorted({target[0] for target in targets})
    }

    target_results = {}
    for target in targets:
        relative_file, line = target
        source = source_audits[relative_file]
        exists_neo4j = target in neo4j
        exists_qdrant = target in qdrant
        exists_source_ast = line in source.ast_lines
        present = exists_neo4j and exists_qdrant and exists_source_ast
        classification = None
        detail = None
        if not present:
            classification, detail = source.classify_missing(line)
        target_results[target] = {
            "file": relative_file,
            "line": line,
            "exists_neo4j": exists_neo4j,
            "exists_qdrant": exists_qdrant,
            "exists_source_ast": exists_source_ast,
            "present_in_all_ast_checks": present,
            "classification": classification,
            "detail": detail,
        }

    missing_queries = []
    for query in benchmark:
        result = target_results[expected_target(query)]
        if not result["present_in_all_ast_checks"]:
            missing_queries.append(
                {
                    "id": query["id"],
                    "question": query["question"],
                    "expected": query["expected"],
                    "source_part": source_part(query, signature_to_parts),
                    "classification": result["classification"],
                    "detail": result["detail"],
                    "exists_neo4j": result["exists_neo4j"],
                    "exists_qdrant": result["exists_qdrant"],
                    "exists_source_ast": result["exists_source_ast"],
                }
            )

    false_positive_ids = {
        item["id"]
        for item in missing_queries
        if item["classification"] == "legacy regex false positive"
    }
    cleaned = [query for query in benchmark if query["id"] not in false_positive_ids]
    missing_targets = {
        expected_target(query)
        for query in benchmark
        if not target_results[expected_target(query)]["present_in_all_ast_checks"]
    }
    present_queries = len(benchmark) - len(missing_queries)
    present_targets = len(targets) - len(missing_targets)

    report = {
        "benchmark": os.path.abspath(args.benchmark),
        "benchmark_sha256_before": original_checksum,
        "benchmark_sha256_after": sha256(args.benchmark),
        "neo4j_uri": args.neo4j_uri,
        "qdrant_collection": args.collection_name,
        "summary": {
            "total_queries": len(benchmark),
            "total_unique_gold_targets": len(targets),
            "queries_target_exists_in_neo4j": sum(
                target_results[expected_target(query)]["exists_neo4j"] for query in benchmark
            ),
            "unique_targets_existing_in_neo4j": sum(
                target_results[target]["exists_neo4j"] for target in targets
            ),
            "queries_target_exists_in_qdrant": sum(
                target_results[expected_target(query)]["exists_qdrant"] for query in benchmark
            ),
            "unique_targets_existing_in_qdrant": sum(
                target_results[target]["exists_qdrant"] for target in targets
            ),
            "queries_verified_by_source_ast": sum(
                target_results[expected_target(query)]["exists_source_ast"] for query in benchmark
            ),
            "unique_targets_verified_by_source_ast": sum(
                target_results[target]["exists_source_ast"] for target in targets
            ),
            "queries_target_exists_in_ast": present_queries,
            "unique_targets_existing_in_ast": present_targets,
            "queries_target_missing_in_ast": len(missing_queries),
            "unique_targets_missing_in_ast": len(missing_targets),
            "missing_classifications_by_query": dict(
                Counter(item["classification"] for item in missing_queries)
            ),
            "missing_classifications_by_unique_target": dict(
                Counter(target_results[target]["classification"] for target in missing_targets)
            ),
        },
        "composition_before": composition(benchmark, signature_to_parts),
        "composition_after_proposed_false_positive_cleaning": composition(cleaned, signature_to_parts),
        "proposed_cleaning": {
            "policy": "Remove only confirmed legacy-regex false-positive labels.",
            "removed_query_ids": sorted(false_positive_ids),
            "removed_query_count": len(false_positive_ids),
            "retained_valid_but_unsupported_query_ids": sorted(
                item["id"]
                for item in missing_queries
                if item["classification"] != "legacy regex false positive"
            ),
        },
        "missing_queries": missing_queries,
        "missing_unique_targets": [target_results[target] for target in sorted(missing_targets)],
        "store_consistency": {
            "gold_neo4j_only": [list(target) for target in sorted((set(neo4j) - set(qdrant)) & set(targets))],
            "gold_qdrant_only": [list(target) for target in sorted((set(qdrant) - set(neo4j)) & set(targets))],
        },
        "errors": {
            "source_files_with_ast_parser_errors": sorted(
                source.relative_file
                for source in source_audits.values()
                if source.ast_parser_error
            ),
            "indexing_errors": [],
            "neo4j_errors": [],
            "qdrant_errors": [],
        },
    }

    if false_positive_ids and args.cleaned_out:
        write_json(args.cleaned_out, cleaned)
    if args.report_out:
        write_json(args.report_out, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if false_positive_ids and args.cleaned_out:
        print(f"\nProposed cleaned benchmark: {args.cleaned_out}")
    if args.report_out:
        print(f"Audit report: {args.report_out}")


if __name__ == "__main__":
    main()
