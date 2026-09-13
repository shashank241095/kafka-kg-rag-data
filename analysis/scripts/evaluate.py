#!/usr/bin/env python3
"""Evaluate the new external-baseline conditions and compare against frozen methods.

Reuses the FROZEN metric and statistics definitions by importing them directly from
scripts/analyze_ast_retrieval_statistics.py -- exact_mcnemar_p, percentile, hit_at_k and
reciprocal_rank_at_k are not reimplemented here, so the new comparisons use byte-identical
definitions to the ones already in the manuscript.

Gold matching is exact canonical (file,line). No semantic, method, class, substring or
approximate-line matching anywhere.

Reads only committed artifacts; never touches Qdrant, Neo4j, or the live HippoRAG index.
"""
import argparse, csv, importlib.util, json, os, random, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "artifacts/hipporag")
KS = (5, 10, 15, 20)

# --- import the frozen statistics implementations -------------------------------------
_spec = importlib.util.spec_from_file_location(
    "frozen_stats", os.path.join(ROOT, "scripts/analyze_ast_retrieval_statistics.py"))
frozen_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen_stats)
exact_mcnemar_p = frozen_stats.exact_mcnemar_p
percentile = frozen_stats.percentile

FROZEN_PER_QUERY = {
    "Vector-only": "vector_only.json",
    "Vector + Cross-Encoder": "vector_crossencoder.json",
    "AST-KG static": "kg_crossencoder_static_k.json",
    "AST-KG staged": "kg_crossencoder_stagedconfidence.json",
}


def hit(row, k):
    r = row.get("gold_rank")
    return 0.0 if r is None or int(r) > k else 1.0


def rr(row, k):
    r = row.get("gold_rank")
    return 0.0 if r is None or int(r) > k else 1.0 / int(r)


def load_frozen(k):
    """Per-query rows for the four frozen methods at cutoff k, keyed by query_id."""
    d = os.path.join(ROOT, f"results/ast_k_sensitivity_v1/per_query_results/k_{k}")
    out = {}
    for label, fn in FROZEN_PER_QUERY.items():
        rows = json.load(open(os.path.join(d, fn), encoding="utf-8"))["queries"]
        out[label] = {r["query_id"]: r for r in rows}
    return out


def load_new(name):
    p = os.path.join(A, f"{name}_per_query.jsonl")
    if not os.path.exists(p):
        return None
    return {r["query_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}


def aggregate(rows, k):
    n = len(rows)
    hits = sum(hit(r, k) for r in rows.values())
    mrr = sum(rr(r, k) for r in rows.values()) / n
    return {"queries": n, "hits": int(hits), "recall": hits / n, "mrr": mrr}


def target_balanced(rows, targets, k):
    """Average within each expected (file,line) target, then average the 103 targets equally."""
    rec = [statistics.fmean(hit(rows[q], k) for q in qs) for qs in targets.values()]
    mrr = [statistics.fmean(rr(rows[q], k) for q in qs) for qs in targets.values()]
    return {"target_balanced_recall": statistics.fmean(rec),
            "target_balanced_mrr": statistics.fmean(mrr),
            "unique_target_count": len(targets)}


def mcnemar(a, b, k):
    """b vs a. n10 = a-only hits, n01 = b-only hits."""
    both = n10 = n01 = neither = 0
    for q in a:
        ha, hb = hit(a[q], k) > 0, hit(b[q], k) > 0
        if ha and hb: both += 1
        elif ha and not hb: n10 += 1
        elif hb and not ha: n01 += 1
        else: neither += 1
    return {"both_hit": both, "n10_a_only": n10, "n01_b_only": n01, "both_miss": neither,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(n10, n01)}


def cluster_bootstrap(a, b, targets, k, seed=42, reps=10000):
    """Resample the 103 targets with replacement, keep all queries in each drawn target."""
    rng = random.Random(seed)
    keys = sorted(targets)
    rd, md = [], []
    for _ in range(reps):
        drawn = [keys[rng.randrange(len(keys))] for _ in keys]
        ids = [q for t in drawn for q in targets[t]]
        rd.append(statistics.fmean(hit(b[q], k) - hit(a[q], k) for q in ids))
        md.append(statistics.fmean(rr(b[q], k) - rr(a[q], k) for q in ids))
    n = len(a)
    return {
        "Recall": {"difference": sum(hit(b[q], k) - hit(a[q], k) for q in a) / n,
                   "ci_95_lower": percentile(rd, 0.025), "ci_95_upper": percentile(rd, 0.975)},
        "MRR": {"difference": sum(rr(b[q], k) - rr(a[q], k) for q in a) / n,
                "ci_95_lower": percentile(md, 0.025), "ci_95_upper": percentile(md, 0.975)},
        "bootstrap_seed": seed, "bootstrap_replicates": reps,
        "resampling_unit": "expected (file,line) target cluster",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["retriever_only"])
    args = ap.parse_args()

    queries = json.load(open(os.path.join(A, "queries.json"), encoding="utf-8"))
    targets = {}
    for q in queries:
        targets.setdefault(q["canonical_target"], []).append(q["query_id"])
    assert len(queries) == 315 and len(targets) == 103

    NEW_LABEL = {"retriever_only": "ColBERTv2-only", "full": "HippoRAG (ColBERTv2)"}
    new = {}
    for c in args.conditions:
        rows = load_new(c)
        if rows is None:
            print(f"[skip] {c}: no per-query file yet")
            continue
        assert len(rows) == 315, f"{c}: {len(rows)} rows"
        assert set(rows) == {q["query_id"] for q in queries}, f"{c}: query_id mismatch"
        new[NEW_LABEL[c]] = rows

    report = {"k_values": list(KS), "methods": {}, "comparisons": {}}

    # ---- per-K aggregates for frozen + new -------------------------------------------
    ktab = []
    for k in KS:
        frozen = load_frozen(k)
        for label, rows in list(frozen.items()) + list(new.items()):
            agg = aggregate(rows, k)
            tb = target_balanced(rows, targets, k)
            report["methods"].setdefault(label, {})[f"k_{k}"] = {**agg, **tb}
            ktab.append({"method": label, "k": k, **agg, **tb})

    with open(os.path.join(A, "table_k_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ktab[0].keys()))
        w.writeheader(); w.writerows(ktab)

    # ---- paired comparisons at K=20 ---------------------------------------------------
    K = 20
    frozen20 = load_frozen(K)
    pairs = []
    if "ColBERTv2-only" in new:
        pairs.append(("Vector-only", "ColBERTv2-only"))
    if "HippoRAG (ColBERTv2)" in new:
        pairs.append(("ColBERTv2-only", "HippoRAG (ColBERTv2)"))   # PRIMARY control
        pairs += [(m, "HippoRAG (ColBERTv2)")
                  for m in ("Vector-only", "AST-KG static", "AST-KG staged")]

    allrows = {**frozen20, **new}
    paired_csv = []
    for a_lbl, b_lbl in pairs:
        if a_lbl not in allrows or b_lbl not in allrows:
            continue
        a, b = allrows[a_lbl], allrows[b_lbl]
        mc = mcnemar(a, b, K)
        bs = cluster_bootstrap(a, b, targets, K)
        key = f"{b_lbl} vs {a_lbl}"
        report["comparisons"][key] = {"method_a": a_lbl, "method_b": b_lbl, "k": K,
                                      "mcnemar": mc, "bootstrap": bs}
        paired_csv.append({
            "method_a": a_lbl, "method_b": b_lbl, "difference": f"{b_lbl} - {a_lbl}", "k": K,
            "recall_difference": bs["Recall"]["difference"],
            "recall_ci_95_lower": bs["Recall"]["ci_95_lower"],
            "recall_ci_95_upper": bs["Recall"]["ci_95_upper"],
            "mrr_difference": bs["MRR"]["difference"],
            "mrr_ci_95_lower": bs["MRR"]["ci_95_lower"],
            "mrr_ci_95_upper": bs["MRR"]["ci_95_upper"],
            "both_hit": mc["both_hit"], "n10_a_only": mc["n10_a_only"],
            "n01_b_only": mc["n01_b_only"], "both_miss": mc["both_miss"],
            "exact_mcnemar_two_sided_p": mc["exact_mcnemar_two_sided_p"],
            "bootstrap_seed": 42, "bootstrap_replicates": 10000,
            "resampling_unit": "expected (file,line) target cluster",
        })
    if paired_csv:
        with open(os.path.join(A, "table_paired.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(paired_csv[0].keys()))
            w.writeheader(); w.writerows(paired_csv)

    # ---- main table (K=20) ------------------------------------------------------------
    ORDER = ["Vector-only", "Vector + Cross-Encoder", "ColBERTv2-only",
             "HippoRAG (ColBERTv2)", "AST-KG static", "AST-KG staged"]
    main_rows = []
    for m in ORDER:
        if m not in report["methods"]:
            continue
        d = report["methods"][m][f"k_{K}"]
        main_rows.append({"Method": m, "Hits@20": f"{d['hits']}/{d['queries']}",
                          "Recall@20": f"{d['recall']:.6f}", "MRR@20": f"{d['mrr']:.6f}",
                          "TargetBalancedRecall@20": f"{d['target_balanced_recall']:.6f}",
                          "TargetBalancedMRR@20": f"{d['target_balanced_mrr']:.6f}"})
    with open(os.path.join(A, "table_main.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
        w.writeheader(); w.writerows(main_rows)

    json.dump(report, open(os.path.join(A, "evaluation.json"), "w"), indent=2)

    # ---- console ----------------------------------------------------------------------
    print(f"\n{'Method':<24}{'Hits@20':>10}{'Recall@20':>12}{'MRR@20':>11}{'tbRec@20':>11}{'tbMRR@20':>11}")
    print("-" * 79)
    for r in main_rows:
        print(f"{r['Method']:<24}{r['Hits@20']:>10}{r['Recall@20']:>12}"
              f"{r['MRR@20']:>11}{r['TargetBalancedRecall@20']:>11}{r['TargetBalancedMRR@20']:>11}")

    print(f"\nK sensitivity (new conditions):")
    print(f"{'Method':<24}{'K':>4}{'Hits':>9}{'Recall':>11}{'MRR':>11}")
    print("-" * 59)
    for r in ktab:
        if r["method"] in new:
            print(f"{r['method']:<24}{r['k']:>4}{str(r['hits'])+'/'+str(r['queries']):>9}"
                  f"{r['recall']:>11.6f}{r['mrr']:>11.6f}")

    for key, c in report["comparisons"].items():
        mc, bs = c["mcnemar"], c["bootstrap"]
        print(f"\n{key}  (K=20)")
        print(f"  Recall diff {bs['Recall']['difference']:+.6f}   "
              f"95% CI [{bs['Recall']['ci_95_lower']:+.6f}, {bs['Recall']['ci_95_upper']:+.6f}]")
        print(f"  MRR    diff {bs['MRR']['difference']:+.6f}   "
              f"95% CI [{bs['MRR']['ci_95_lower']:+.6f}, {bs['MRR']['ci_95_upper']:+.6f}]")
        print(f"  discordant: {c['method_a']}-only={mc['n10_a_only']}  "
              f"{c['method_b']}-only={mc['n01_b_only']}  "
              f"both={mc['both_hit']}  neither={mc['both_miss']}")
        print(f"  exact two-sided McNemar p = {mc['exact_mcnemar_two_sided_p']:.6g}")

    print(f"\nwrote table_main.csv, table_k_sensitivity.csv, table_paired.csv, evaluation.json")


if __name__ == "__main__":
    main()
