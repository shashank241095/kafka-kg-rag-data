#!/usr/bin/env python3
"""Post-hoc sensitivity comparisons for HippoRAG doc_ensemble=true.

Reuses the frozen statistical definitions (exact McNemar, percentile, target-cluster
bootstrap with seed 42 / 10,000 replicates) imported from the manuscript's own analysis
module, so these numbers are computed identically to every other paired result reported.
"""
import csv, importlib.util, json, os, random, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "artifacts/hipporag")
OUT = os.path.join(A, "doc_ensemble_true")
K = 20
KS = (5, 10, 15, 20)

_spec = importlib.util.spec_from_file_location(
    "fs", os.path.join(ROOT, "scripts/analyze_ast_retrieval_statistics.py"))
fs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fs)
exact_mcnemar_p, percentile = fs.exact_mcnemar_p, fs.percentile

FROZEN = {"Vector-only": "vector_only.json", "Vector + Cross-Encoder": "vector_crossencoder.json",
          "AST-KG static": "kg_crossencoder_static_k.json",
          "AST-KG staged": "kg_crossencoder_stagedconfidence.json"}


def hit(r, k=K):
    g = r.get("gold_rank")
    return 0.0 if g is None or int(g) > k else 1.0


def rr(r, k=K):
    g = r.get("gold_rank")
    return 0.0 if g is None or int(g) > k else 1.0 / int(g)


def jl(p):
    return {r["query_id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}


def frozen(fn, k=K):
    p = os.path.join(ROOT, f"results/ast_k_sensitivity_v1/per_query_results/k_{k}/{fn}")
    return {r["query_id"]: r for r in json.load(open(p, encoding="utf-8"))["queries"]}


def agg(rows, k=K):
    n = len(rows)
    h = sum(hit(r, k) for r in rows.values())
    return {"queries": n, "hits": int(h), "recall": h / n,
            "mrr": sum(rr(r, k) for r in rows.values()) / n}


def tb(rows, targets, k=K):
    return {"tb_recall": statistics.fmean(
                [statistics.fmean(hit(rows[q], k) for q in qs) for qs in targets.values()]),
            "tb_mrr": statistics.fmean(
                [statistics.fmean(rr(rows[q], k) for q in qs) for qs in targets.values()])}


def paired(a, b, targets, seed=42, reps=10000):
    both = n10 = n01 = neither = 0
    for q in a:
        ha, hb = hit(a[q]) > 0, hit(b[q]) > 0
        both += ha and hb; n10 += ha and not hb; n01 += hb and not ha
        neither += (not ha) and (not hb)
    rng = random.Random(seed); keys = sorted(targets); rd, md = [], []
    for _ in range(reps):
        drawn = [keys[rng.randrange(len(keys))] for _ in keys]
        ids = [q for t in drawn for q in targets[t]]
        rd.append(statistics.fmean(hit(b[q]) - hit(a[q]) for q in ids))
        md.append(statistics.fmean(rr(b[q]) - rr(a[q]) for q in ids))
    n = len(a)
    return {"n10_a_only": n10, "n01_b_only": n01, "both_hit": both, "both_miss": neither,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(n10, n01),
            "recall_diff": sum(hit(b[q]) - hit(a[q]) for q in a) / n,
            "recall_ci_lo": percentile(rd, 0.025), "recall_ci_hi": percentile(rd, 0.975),
            "mrr_diff": sum(rr(b[q]) - rr(a[q]) for q in a) / n,
            "mrr_ci_lo": percentile(md, 0.025), "mrr_ci_hi": percentile(md, 0.975)}


def concentration(rows, label):
    import collections
    t1 = collections.Counter(r["ranked_doc_idx"][0] for r in rows.values())
    t20 = collections.Counter(d for r in rows.values() for d in r["ranked_doc_idx"])
    return {"method": label, "distinct_top1_docs": len(t1),
            "max_top1_repeat": t1.most_common(1)[0][1],
            "distinct_docs_in_top20": len(t20),
            "max_top20_frequency": t20.most_common(1)[0][1],
            "top20_coverage_pct": 100.0 * len(t20) / 1600}


def main():
    queries = json.load(open(os.path.join(A, "queries.json"), encoding="utf-8"))
    targets = {}
    for q in queries:
        targets.setdefault(q["canonical_target"], []).append(q["query_id"])
    assert len(queries) == 315 and len(targets) == 103

    M = {"ColBERTv2-only": jl(os.path.join(A, "retriever_only_per_query.jsonl")),
         "HippoRAG doc_ensemble=false": jl(os.path.join(A, "full_per_query.jsonl")),
         "HippoRAG doc_ensemble=true": jl(os.path.join(OUT, "per_query.jsonl"))}
    for lbl, fn in FROZEN.items():
        M[lbl] = frozen(fn)
    for lbl, rows in M.items():
        assert len(rows) == 315, f"{lbl}: {len(rows)}"

    # ---- aggregates ----
    rows_out = []
    for lbl in ("ColBERTv2-only", "HippoRAG doc_ensemble=false", "HippoRAG doc_ensemble=true",
                "Vector-only", "Vector + Cross-Encoder", "AST-KG static", "AST-KG staged"):
        a = agg(M[lbl]); t = tb(M[lbl], targets)
        rows_out.append({"method": lbl, **a, **t})
    with open(os.path.join(OUT, "aggregate_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print(f"\n{'Method':<30}{'Hits@20':>10}{'Recall@20':>12}{'MRR@20':>11}{'tbRec':>10}{'tbMRR':>10}")
    print("-" * 83)
    for r in rows_out:
        print(f"{r['method']:<30}{str(r['hits'])+'/315':>10}{r['recall']:>12.6f}"
              f"{r['mrr']:>11.6f}{r['tb_recall']:>10.6f}{r['tb_mrr']:>10.6f}")

    # ---- paired comparisons ----
    T = "HippoRAG doc_ensemble=true"
    comps = [("HippoRAG doc_ensemble=false", T), ("ColBERTv2-only", T),
             ("Vector-only", T), ("AST-KG static", T), ("AST-KG staged", T)]
    pr = []
    for a_lbl, b_lbl in comps:
        st = paired(M[a_lbl], M[b_lbl], targets)
        pr.append({"comparison": f"{b_lbl} - {a_lbl}", "method_a": a_lbl, "method_b": b_lbl, **st})
        print(f"\n{b_lbl} - {a_lbl}  (K=20)")
        print(f"  Recall {st['recall_diff']:+.6f}  CI [{st['recall_ci_lo']:+.6f}, {st['recall_ci_hi']:+.6f}]")
        print(f"  MRR    {st['mrr_diff']:+.6f}  CI [{st['mrr_ci_lo']:+.6f}, {st['mrr_ci_hi']:+.6f}]")
        print(f"  discordant {b_lbl}-only={st['n01_b_only']} / {a_lbl}-only={st['n10_a_only']}"
              f"   both={st['both_hit']} neither={st['both_miss']}")
        print(f"  exact McNemar p = {st['exact_mcnemar_two_sided_p']:.6g}")
    with open(os.path.join(OUT, "paired_statistics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pr[0].keys())); w.writeheader(); w.writerows(pr)

    # ---- K sensitivity across the three external conditions ----
    ks = []
    for lbl in ("ColBERTv2-only", "HippoRAG doc_ensemble=false", "HippoRAG doc_ensemble=true"):
        for k in KS:
            a = agg(M[lbl], k)
            ks.append({"method": lbl, "k": k, **a})
    with open(os.path.join(OUT, "k_sensitivity_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ks[0].keys())); w.writeheader(); w.writerows(ks)
    print(f"\n{'Method':<30}{'K':>4}{'Hits':>10}{'Recall':>11}{'MRR':>11}")
    print("-" * 66)
    for r in ks:
        print(f"{r['method']:<30}{r['k']:>4}{str(r['hits'])+'/315':>10}{r['recall']:>11.6f}{r['mrr']:>11.6f}")

    # ---- concentration diagnostics ----
    cc = [concentration(M[l], l) for l in
          ("ColBERTv2-only", "HippoRAG doc_ensemble=false", "HippoRAG doc_ensemble=true")]
    with open(os.path.join(OUT, "concentration_diagnostics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cc[0].keys())); w.writeheader(); w.writerows(cc)
    print(f"\n{'Method':<30}{'distinct top20':>15}{'max doc freq':>14}{'distinct top1':>15}{'max top1':>10}")
    print("-" * 84)
    for c in cc:
        print(f"{c['method']:<30}{c['distinct_docs_in_top20']:>15}{c['max_top20_frequency']:>14}"
              f"{c['distinct_top1_docs']:>15}{c['max_top1_repeat']:>10}")

    json.dump({"aggregates": rows_out, "paired": pr, "k_sensitivity": ks,
               "concentration": cc,
               "note": "POST-HOC sensitivity analysis; not preregistered."},
              open(os.path.join(OUT, "comparison.json"), "w"), indent=2)
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
