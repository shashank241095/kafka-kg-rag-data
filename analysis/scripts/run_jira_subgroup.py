#!/usr/bin/env python3
"""Exploratory JIRA-seeded subgroup robustness analysis (n=21).

Computes retrieval metrics on the 21 issue-seeded queries ONLY, entirely from frozen
per-query artifacts. No retrieval, embedding, OpenIE or LLM call is executed; no primary
artifact is modified. Metric definitions are identical to the main paper:
  Recall@K = 1 iff 1 <= gold_rank <= K ;  MRR@K = 1/gold_rank under the same cutoff, else 0.

The JIRA membership is derived two independent ways and the two are required to agree.
"""
import csv, hashlib, importlib.util, json, math, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/jira_subgroup_analysis_v1")
KS = (5, 10, 15, 20)
BENCH = os.path.join(ROOT, "scripts/benchMarkScript/"
                           "benchmark_incident_v2_plus_v3_plus_jira_flat_ast_cleaned_proposed.json")
JIRA_SRC = os.path.join(ROOT, "scripts/benchMarkScript/benchmark_incident_jira_seed_flat.json")
STATS_CSV = os.path.join(ROOT, "scripts/results_ast_315_paper_config/"
                               "statistical_analysis/per_query_results.csv")

_spec = importlib.util.spec_from_file_location(
    "fs", os.path.join(ROOT, "scripts/analyze_ast_retrieval_statistics.py"))
fs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fs)
exact_mcnemar_p, percentile = fs.exact_mcnemar_p, fs.percentile


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sig(row):
    e = row.get("expected") or {}
    return (row.get("question"), e.get("file"), e.get("line"))


# ---------------- STEP 1: identify the JIRA subset, two independent ways ----------------
bench = json.load(open(BENCH, encoding="utf-8"))
jira_src = json.load(open(JIRA_SRC, encoding="utf-8"))
by_sig = {sig(r): r["id"] for r in bench}
derived_a = {by_sig[sig(r)] for r in jira_src if sig(r) in by_sig}          # via source file
derived_b = {r["query_id"] for r in csv.DictReader(open(STATS_CSV))
             if r["subset"] == "JIRA" and r["method"] == "Vector-only"}     # via frozen label
assert derived_a == derived_b, (f"JIRA derivations disagree: only-A={derived_a-derived_b} "
                                f"only-B={derived_b-derived_a}")
JIRA = sorted(derived_a)
assert len(JIRA) == 21, f"expected 21 JIRA queries, got {len(JIRA)}"

gold = {r["id"]: (r["expected"]["file"], r["expected"]["line"]) for r in bench}
jira_gold = {q: gold[q] for q in JIRA}
targets = {}
for q in JIRA:
    targets.setdefault(f"{jira_gold[q][0]}:{jira_gold[q][1]}", []).append(q)

# ---------------- STEP 2: locate frozen per-query results ----------------
INTERNAL = {"Vector-only": "vector_only.json",
            "Vector + CE": "vector_crossencoder.json",
            "AST-KG static": "kg_crossencoder_static_k.json",
            "AST-KG staged": "kg_crossencoder_stagedconfidence.json"}
EXTERNAL = {"ColBERTv2-only": "artifacts/hipporag/retriever_only_per_query.jsonl",
            "HippoRAG (ensemble off)": "artifacts/hipporag/full_per_query.jsonl",
            "HippoRAG + doc ensemble [POST-HOC]": "artifacts/hipporag/doc_ensemble_true/per_query.jsonl"}

sources, ranks = {}, {}   # ranks[method][k][qid] = gold rank or None
for m, fn in INTERNAL.items():
    ranks[m] = {}
    for k in KS:
        p = os.path.join(ROOT, f"results/ast_k_sensitivity_v1/per_query_results/k_{k}/{fn}")
        sources[f"{m} @K={k}"] = {"path": os.path.relpath(p, ROOT), "sha256": sha(p)}
        rows = json.load(open(p, encoding="utf-8"))["queries"]
        assert len(rows) == 315
        ranks[m][k] = {r["query_id"]: r.get("gold_rank") for r in rows}
for m, rel in EXTERNAL.items():
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        continue
    sources[m] = {"path": rel, "sha256": sha(p)}
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert len(rows) == 315
    full = {r["query_id"]: r.get("gold_rank") for r in rows}   # rank over the whole corpus
    ranks[m] = {k: full for k in KS}                            # thresholded at scoring time

for m in ranks:
    missing = [q for q in JIRA if q not in ranks[m][20]]
    assert not missing, f"{m}: missing JIRA queries {missing}"

# gold labels identical in every external artifact
for m, rel in EXTERNAL.items():
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        continue
    for r in (json.loads(l) for l in open(p, encoding="utf-8")):
        if r["query_id"] in jira_gold:
            assert (r["gold_file"], r["gold_line"]) == jira_gold[r["query_id"]], \
                f"{m}: gold label drift at {r['query_id']}"


def hit(m, k, q):
    r = ranks[m][k].get(q)
    return 1.0 if (r is not None and int(r) <= k) else 0.0


def rr(m, k, q):
    r = ranks[m][k].get(q)
    return (1.0 / int(r)) if (r is not None and int(r) <= k) else 0.0


METHODS = [m for m in list(INTERNAL) + list(EXTERNAL) if m in ranks]

# ---------------- STEP 8 (first): reconstruct FULL-benchmark metrics ----------------
FROZEN = {"Vector-only": (301, 0.955556, 0.751194), "Vector + CE": (301, 0.955556, 0.697488),
          "AST-KG static": (298, 0.946032, 0.704563), "AST-KG staged": (301, 0.955556, 0.697188),
          "ColBERTv2-only": (253, 0.803175, 0.551997),
          "HippoRAG (ensemble off)": (97, 0.307937, 0.109862),
          "HippoRAG + doc ensemble [POST-HOC]": (204, 0.647619, 0.219012)}
allq = [r["id"] for r in bench]
recon, recon_ok = {}, True
for m in METHODS:
    h = sum(hit(m, 20, q) for q in allq)
    mr = sum(rr(m, 20, q) for q in allq) / len(allq)
    fh, frc, fmr = FROZEN[m]
    ok = (int(h) == fh and abs(h/len(allq) - frc) < 5e-7 and abs(mr - fmr) < 5e-7)
    recon_ok &= ok
    recon[m] = {"hits": int(h), "recall": h/len(allq), "mrr": mr,
                "frozen_hits": fh, "frozen_recall": frc, "frozen_mrr": fmr, "matches": ok}
if not recon_ok:
    for m, v in recon.items():
        if not v["matches"]:
            print(f"MISMATCH {m}: {v}")
    raise SystemExit("STOP: full-benchmark reconstruction does not match frozen values.")

# ---------------- STEP 3: JIRA-only metrics ----------------
n = len(JIRA)
metric_rows = []
for m in METHODS:
    for k in KS:
        h = sum(hit(m, k, q) for q in JIRA)
        mr = sum(rr(m, k, q) for q in JIRA) / n
        tb_r = statistics.fmean([statistics.fmean(hit(m, k, q) for q in qs) for qs in targets.values()])
        tb_m = statistics.fmean([statistics.fmean(rr(m, k, q) for q in qs) for qs in targets.values()])
        metric_rows.append({"method": m, "k": k, "queries": n, "hits": int(h),
                            "recall": h/n, "mrr": mr,
                            "target_balanced_recall": tb_r, "target_balanced_mrr": tb_m})

# ---------------- STEP 4/5: paired comparisons (exploratory) ----------------
PAIRS = [("Vector-only", "AST-KG static"), ("Vector-only", "AST-KG staged"),
         ("AST-KG static", "AST-KG staged")]
if "ColBERTv2-only" in ranks and "HippoRAG (ensemble off)" in ranks:
    PAIRS.append(("ColBERTv2-only", "HippoRAG (ensemble off)"))
if "ColBERTv2-only" in ranks and "HippoRAG + doc ensemble [POST-HOC]" in ranks:
    PAIRS.append(("ColBERTv2-only", "HippoRAG + doc ensemble [POST-HOC]"))

pair_rows = []
for a, b in PAIRS:
    both = n10 = n01 = neither = 0
    for q in JIRA:
        ha, hb = hit(a, 20, q) > 0, hit(b, 20, q) > 0
        both += ha and hb; n10 += ha and not hb; n01 += hb and not ha
        neither += (not ha) and (not hb)
    disc = n10 + n01
    p = exact_mcnemar_p(n10, n01) if disc else None
    dr = sum(hit(b, 20, q) - hit(a, 20, q) for q in JIRA) / n
    dm = sum(rr(b, 20, q) - rr(a, 20, q) for q in JIRA) / n
    pair_rows.append({"comparison": f"{b} - {a}", "method_a": a, "method_b": b, "n": n,
                      "recall_diff": dr, "mrr_diff": dm,
                      "a_only_wins": n10, "b_only_wins": n01, "both_hit": both,
                      "both_miss": neither, "discordant_pairs": disc,
                      "exact_two_sided_p": p if p is not None else "",
                      "note": ("no discordant pairs; no test performed" if not disc
                               else "EXPLORATORY, severely underpowered at n=21")})
# Vector + CE vs Vector-only: MRR only (Recall identical by construction)
pair_rows.append({"comparison": "Vector + CE - Vector-only (MRR only)",
                  "method_a": "Vector-only", "method_b": "Vector + CE", "n": n,
                  "recall_diff": "N/A (identical candidate set by construction)",
                  "mrr_diff": sum(rr("Vector + CE", 20, q) - rr("Vector-only", 20, q) for q in JIRA)/n,
                  "a_only_wins": "", "b_only_wins": "", "both_hit": "", "both_miss": "",
                  "discordant_pairs": "", "exact_two_sided_p": "",
                  "note": "Recall@K identical by construction; no Recall test reported"})

# ---------------- STEP 7: per-query table ----------------
def cell(m, q):
    r = ranks[m][20].get(q)
    return int(r) if (r is not None and int(r) <= 20) else "MISS"

q_rows = []
for q in JIRA:
    row = {"query_id": q, "gold_file": jira_gold[q][0], "gold_line": jira_gold[q][1]}
    for m in METHODS:
        row[m] = cell(m, q)
    q_rows.append(row)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "jira_subgroup_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys())); w.writeheader(); w.writerows(metric_rows)
with open(os.path.join(OUT, "jira_query_level_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(q_rows[0].keys())); w.writeheader(); w.writerows(q_rows)
with open(os.path.join(OUT, "jira_pairwise_comparisons.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys())); w.writeheader(); w.writerows(pair_rows)
json.dump({"analysis": "exploratory JIRA-seeded subgroup robustness check",
           "n_queries": n, "n_unique_targets": len(targets),
           "jira_query_ids": JIRA, "jira_gold": {k: list(v) for k, v in jira_gold.items()},
           "queries_per_target": {t: len(v) for t, v in targets.items()},
           "membership_derivation": "intersection of (a) benchmark_incident_jira_seed_flat.json "
                                    "matched on (question, file, line) and (b) the frozen "
                                    "statistical_analysis subset=JIRA label; both agreed exactly",
           "sources": sources,
           "full_benchmark_reconstruction": recon,
           "retrieval_reexecuted": False, "api_calls": 0,
           "metric_definition": "Recall@K=1 iff 1<=gold_rank<=K; MRR@K=1/gold_rank else 0"},
          open(os.path.join(OUT, "jira_subgroup_metadata.json"), "w"), indent=2)

print(f"JIRA queries: {n}   unique targets: {len(targets)}")
print(f"full-benchmark reconstruction matches frozen values: {recon_ok}")
print(f"\n{'Method':<36}{'Hits@20':>9}{'Recall@20':>11}{'MRR@20':>10}")
print("-" * 66)
for r in metric_rows:
    if r["k"] == 20:
        print(f"{r['method']:<36}{str(r['hits'])+'/'+str(n):>9}{r['recall']:>11.4f}{r['mrr']:>10.4f}")
print("\nPaired (K=20, exploratory):")
for r in pair_rows:
    print(f"  {r['comparison']:<52} dRecall={r['recall_diff'] if isinstance(r['recall_diff'],str) else format(r['recall_diff'],'+.4f')}"
          f"  dMRR={r['mrr_diff']:+.4f}  disc={r['discordant_pairs']}  p={r['exact_two_sided_p']}")
print(f"\nwrote {OUT}/")
