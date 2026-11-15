import json, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1])
for d, t in [("qmd-eval-v1-fiqa-test","fiqa"),("qmd-distill-public-v0-scifact-train","scifact"),("qmd-eval-v1-cqadup-android","android")]:
    base = R/d
    load = lambda p: {r["qid"]: r for r in map(json.loads, open(p))}
    raw = load(base/"runs/results/raw.jsonl")
    exp = load(next((base/"runs/results").glob(f"g0d-{t}-*.jsonl")))
    ex = load(base/f"expansions/g0d-{t}.jsonl")
    q = [k for k in exp if k in raw and exp[k].get("metrics") and raw[k].get("metrics")]
    for m in ("ndcg_at_10","recall_at_30"):
        dl = [exp[k]["metrics"][m]-raw[k]["metrics"][m] for k in q]
        pos = sum(x>1e-9 for x in dl)/len(dl); neg = sum(x<-1e-9 for x in dl)/len(dl)
        print(f"{t:8} {m:13} raw={st.mean(raw[k]['metrics'][m] for k in q):.4f} exp={st.mean(exp[k]['metrics'][m] for k in q):.4f} "
              f"delta mean={st.mean(dl):+.4f} sd={st.pstdev(dl):.4f} +{pos:.0%} -{neg:.0%} ={1-pos-neg:.0%}")
    rawz = sum(raw[k]["metrics"]["recall_at_30"]==0 for k in q)/len(q)
    rawone = sum(raw[k]["metrics"]["ndcg_at_10"]>=0.999 for k in q)/len(q)
    print(f"{t:8} raw R@30=0: {rawz:.0%}   raw nDCG@10=1: {rawone:.0%}")
    ok = [ex[k] for k in q if ex[k]["status"]=="ok"]
    print(f"{t:8} status ok {len(ok)}/{len(q)}")
    chars = sorted(len(r["raw_output"]) for r in ok)
    hyde = sorted(len(r["raw_output"].split("\n")[0]) for r in ok)
    nvec = [r["raw_output"].count("\nvec:") for r in ok]
    pc = lambda a,p: a[int(p*(len(a)-1))]
    print(f"{t:8} chars p10/50/90/99={pc(chars,.1)}/{pc(chars,.5)}/{pc(chars,.9)}/{pc(chars,.99)}  hyde p50/90={pc(hyde,.5)}/{pc(hyde,.9)}  vec=1:{nvec.count(1)} vec=2:{nvec.count(2)}")
    g = [r.get("generation") or {} for r in ex.values()]
    print(f"{t:8} gen keys={sorted(g[0].keys()) if g and g[0] else None} stop={dict((s, sum(x.get('stop_reason')==s for x in g)) for s in set(x.get('stop_reason') for x in g))}")
    print()
