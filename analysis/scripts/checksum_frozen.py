import csv, hashlib, json, os, sys

ROOT = "/Users/shashankmishra/kafka-kg-rag"

ARTIFACTS = [
    # (artifact_name, path)
    ("benchmark_315_final", "scripts/benchMarkScript/benchmark_incident_v2_plus_v3_plus_jira_flat_ast_cleaned_proposed.json"),
    ("benchmark_ast_audit_report", "scripts/benchMarkScript/benchmark_ast_revised_audit_report.json"),
    ("corpus_generator_embed_script", "scripts/embed_logs_to_qdrant.py"),
    ("corpus_serialization_eval_script", "scripts/eval_kg_vs_vector.py"),
    ("primary_per_query_vector_only", "scripts/results_ast_315_paper_config/per_query/vector_only.json"),
    ("primary_per_query_vector_crossencoder", "scripts/results_ast_315_paper_config/per_query/vector_crossencoder.json"),
    ("primary_per_query_kg_static_k", "scripts/results_ast_315_paper_config/per_query/kg_crossencoder_static_k.json"),
    ("primary_per_query_kg_staged", "scripts/results_ast_315_paper_config/per_query/kg_crossencoder_stagedconfidence.json"),
    ("primary_corrected_results_table", "scripts/results_ast_315_paper_config/corrected_metrics/results_table.csv"),
    ("primary_corrected_results_summary", "scripts/results_ast_315_paper_config/corrected_metrics/results_summary.json"),
    ("stats_per_query_results_csv", "scripts/results_ast_315_paper_config/statistical_analysis/per_query_results.csv"),
    ("stats_paired_mcnemar", "scripts/results_ast_315_paper_config/statistical_analysis/paired_mcnemar.csv"),
    ("stats_cluster_bootstrap_cis", "scripts/results_ast_315_paper_config/statistical_analysis/cluster_bootstrap_cis.csv"),
    ("stats_target_balanced_metrics", "scripts/results_ast_315_paper_config/statistical_analysis/target_balanced_metrics.csv"),
    ("stats_target_level_metrics", "scripts/results_ast_315_paper_config/statistical_analysis/target_level_metrics.csv"),
    ("stats_statistical_analysis_json", "scripts/results_ast_315_paper_config/statistical_analysis/statistical_analysis.json"),
    ("k_sensitivity_experiment_config", "results/ast_k_sensitivity_v1/experiment_config.json"),
    ("k_sensitivity_results_csv", "results/ast_k_sensitivity_v1/k_sensitivity_results.csv"),
    ("k_sensitivity_results_json", "results/ast_k_sensitivity_v1/k_sensitivity_results.json"),
    ("k_sensitivity_paired_tests", "results/ast_k_sensitivity_v1/paired_tests.json"),
    ("k_sensitivity_target_balanced", "results/ast_k_sensitivity_v1/target_balanced_results.json"),
    ("eval_code_mcnemar_bootstrap_primary", "scripts/analyze_ast_retrieval_statistics.py"),
    ("eval_code_mcnemar_bootstrap_ksens", "scripts/analyze_ast_k_sensitivity.py"),
    ("eval_code_recompute_corrected_metrics", "scripts/recompute_corrected_ast_metrics.py"),
    ("paper_access_tex", "paper/access.tex"),
    ("paper_results_tables_tex", "paper/results_tables.tex"),
    ("paper_draft_md", "paper/kafkaops_kg_paper_draft.md"),
]
for k in (5, 10, 15, 20):
    for m in ("vector_only", "vector_crossencoder", "kg_crossencoder_static_k", "kg_crossencoder_stagedconfidence"):
        ARTIFACTS.append((f"ksens_per_query_k{k}_{m}",
                          f"results/ast_k_sensitivity_v1/per_query_results/k_{k}/{m}.json"))


def rowcount(path):
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    if path.endswith(".json"):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            return ""
        if isinstance(d, list):
            return len(d)
        if isinstance(d, dict):
            for key in ("queries", "rows"):
                if key in d and isinstance(d[key], list):
                    return len(d[key])
        return ""
    return ""


out = os.path.join(ROOT, "artifacts/hipporag/frozen_checksums_before.csv")
missing = []
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["artifact_name", "path", "sha256", "size_bytes", "row_count_if_applicable"])
    for name, rel in ARTIFACTS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            missing.append(rel)
            w.writerow([name, rel, "MISSING", "", ""])
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        w.writerow([name, rel, h, os.path.getsize(p), rowcount(p)])

print("wrote", out)
print("entries:", len(ARTIFACTS), "missing:", len(missing))
for m in missing:
    print("  MISSING:", m)
