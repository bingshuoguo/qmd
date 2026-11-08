#!/usr/bin/env python3
"""
analyze.py — gates and paired statistics for the FTS scope-isolation experiment.

Reads (under artifacts/fts-scope-isolation/):
  bench/{S,I}/<dataset>/runs/results/*.jsonl   R3 bench per-query rankings
  probe/<variant>/<dataset>/{bm25,vec,strong,gates}.jsonl   R1/R2/R4/G3/G4

Metrics are recomputed from rankings + qrels.tsv, as in merge-eval's
bootstrap-analysis.py. Bootstrap: paired, stratified by dataset, macro =
mean of per-dataset means, 20,000 resamples, seed 42.

Writes a JSON summary to artifacts/fts-scope-isolation/summary.json and prints
a text report.
"""
import glob, json, math, os, random, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
X = os.path.join(ROOT, 'artifacts/fts-scope-isolation')
SETS = [('fiqa', 'qmd-eval-v1-fiqa-test'), ('scifact', 'qmd-expansion-scifact-v1'), ('android', 'qmd-eval-v1-cqadup-android')]
METRICS = ('ndcg10', 'r30', 'mrr10', 'r10')
N_BOOT, SEED = 20000, 42
MARGIN = {'ndcg10': -0.005, 'r30': -0.01}


def qrels_of(bench):
    rels = {}
    for i, line in enumerate(open(f'{bench}/qrels.tsv')):
        if i == 0 and line.startswith('query-id'):
            continue
        p = line.split()
        if len(p) == 3:
            rels.setdefault(p[0], {})[p[1]] = int(p[2])
    return rels


def metrics_of(ranked, rel):
    dcg = sum((1 if rel.get(d, 0) > 0 else 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    n_rel = sum(1 for v in rel.values() if v > 0)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, 10)))
    rr = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if rel.get(d, 0) > 0), 0.0)
    rec = lambda k: (sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n_rel) if n_rel else None
    return {'ndcg10': dcg / idcg if idcg else 0.0, 'mrr10': rr, 'r10': rec(10), 'r30': rec(30)}


def load_rows(path):
    return {r['qid']: r for r in map(json.loads, open(path))}


def ranked_ids(row):
    return [x['doc_id'] for x in (row.get('ranking') or [])]


def bench_results(group, dataset, run_glob):
    files = glob.glob(f'{X}/bench/{group}/{dataset}/runs/results/{run_glob}.jsonl')
    return load_rows(files[0]) if files else None


def probe(variant, dataset, mode):
    path = f'{X}/probe/{variant}/{dataset}/{mode}.jsonl'
    return load_rows(path) if os.path.exists(path) else None


def per_query(rows, rels):
    out = {}
    for q, r in rows.items():
        if r.get('retrieval_status') != 'ok':
            out[q] = None
            continue
        out[q] = metrics_of(ranked_ids(r), rels.get(q, {}))
    return out


def paired(arms_by_ds, metric):
    """arms_by_ds: {tag: (pq_a, pq_b)} -> delta a-b macro with stratified bootstrap."""
    per_ds = {}
    for tag, (a, b) in arms_by_ds.items():
        d = [a[q][metric] - b[q][metric] for q in a
             if q in b and a[q] and b[q] and a[q][metric] is not None and b[q][metric] is not None]
        per_ds[tag] = d
    obs = sum(sum(d) / len(d) for d in per_ds.values()) / len(per_ds)
    rng = random.Random(SEED)
    boot = []
    lists = list(per_ds.values())
    for _ in range(N_BOOT):
        vals = []
        for d in lists:
            n = len(d)
            vals.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
        boot.append(sum(vals) / len(vals))
    boot.sort()
    lo, hi = boot[int(0.025 * N_BOOT)], boot[int(0.975 * N_BOOT) - 1]
    return {'delta': obs, 'lo': lo, 'hi': hi, 'significant': lo > 0 or hi < 0,
            'per_dataset': {t: sum(d) / len(d) for t, d in per_ds.items()},
            'n': {t: len(d) for t, d in per_ds.items()}}


def mean_metric(pq, metric):
    v = [m[metric] for m in pq.values() if m and m[metric] is not None]
    return sum(v) / len(v)


def changed_share(rows_a, rows_b, k):
    qs = [q for q in rows_a if q in rows_b]
    return sum(1 for q in qs if ranked_ids(rows_a[q])[:k] != ranked_ids(rows_b[q])[:k]) / len(qs)


summary = {'gates': {}, 'r1': {}, 'r2': {}, 'r3': {}, 'r4': {}}
rels = {tag: qrels_of(f'{X}/bench/S/{d}') for tag, d in SETS}

# ---------- gates ----------
print('== Gates')
for tag, d in SETS:
    p = f'{X}/probe/I-{tag}/{d}/gates.jsonl'
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    for r in rows:
        ok3 = r['fts_rows'] == r['target_docs'] == r['active_docs']
        ok4 = r['g4_order_mismatch'] == 0 and r['g4_max_abs_diff'] < 1e-9
        summary['gates'][f'I-{tag}'] = {**r, 'G3': ok3, 'G4': ok4}
        print(f"  I-{tag:8s} G3 fts_rows={r['fts_rows']} target={r['target_docs']} active={r['active_docs']} -> {'PASS' if ok3 else 'FAIL'};"
              f" G4 {r['g4_queries']} queries, order_mismatch={r['g4_order_mismatch']}, max|Δbm25|={r['g4_max_abs_diff']:.2e} -> {'PASS' if ok4 else 'FAIL'}")
for v in ('S', 'D-fiqa', 'D-scifact'):
    for tag, d in SETS:
        p = f'{X}/probe/{v}/{d}/gates.jsonl'
        if os.path.exists(p):
            r = json.loads(open(p).readline())
            ok4 = r['g4_order_mismatch'] == 0 and r['g4_max_abs_diff'] < 1e-9
            summary['gates'][f'{v}/{tag}'] = {**r, 'G4': ok4}
            print(f"  {v}/{tag:8s} fts_rows={r['fts_rows']} G4 order_mismatch={r['g4_order_mismatch']} max|Δ|={r['g4_max_abs_diff']:.2e} -> {'PASS' if ok4 else 'FAIL'}")

# ---------- R2: vec identity ----------
print('\n== R2 vector-only identity (S vs I)')
for tag, d in SETS:
    s, i = probe('S', d, 'vec'), probe(f'I-{tag}', d, 'vec')
    if not (s and i):
        print(f'  {tag}: missing'); continue
    diff = [q for q in s if ranked_ids(s[q]) != ranked_ids(i.get(q, {}))]
    summary['r2'][tag] = {'queries': len(s), 'ranking_differs': len(diff)}
    print(f"  {tag:8s} {len(s)} queries, rankings differ: {len(diff)} -> {'PASS' if not diff else 'FAIL'}")

# ---------- R1: BM25-only ----------
print('\n== R1 BM25-only (original query, top-30)')
bm = {}
for tag, d in SETS:
    for v in ('S', f'D-{tag}', f'I-{tag}'):
        rows = probe(v, d, 'bm25')
        if rows:
            bm[(tag, v.split('-')[0])] = (rows, per_query(rows, rels[tag]))
    # Android has no duplicate corpus in S, so its dedup variant is S itself.
    if (tag, 'D') not in bm and (tag, 'S') in bm:
        bm[(tag, 'D')] = bm[(tag, 'S')]
for tag, d in SETS:
    line = [f'  {tag:8s}']
    for v in ('S', 'D', 'I'):
        if (tag, v) in bm:
            pq = bm[(tag, v)][1]
            line.append(f"{v}: nDCG@10 {mean_metric(pq, 'ndcg10'):.4f} R@30 {mean_metric(pq, 'r30'):.4f}")
    print('  |  '.join(line))
for name, a, b in (('I-S', 'I', 'S'), ('D-S (dedup only)', 'D', 'S'), ('I-D (cross-domain)', 'I', 'D')):
    arms = {tag: (bm[(tag, a)][1], bm[(tag, b)][1]) for tag, _ in SETS if (tag, a) in bm and (tag, b) in bm}
    if len(arms) < 3:
        continue
    summary['r1'][name] = {}
    for metric in METRICS:
        r = paired(arms, metric)
        summary['r1'][name][metric] = r
        per = ' '.join(f'{t}={v:+.4f}' for t, v in r['per_dataset'].items())
        print(f"  {name:20s} {metric:6s} {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {'SIG' if r['significant'] else 'n.s.'}  ({per})")
for tag, _ in SETS:
    if (tag, 'S') in bm and (tag, 'I') in bm:
        summary['r1'].setdefault('changed_top10_share', {})[tag] = changed_share(bm[(tag, 'S')][0], bm[(tag, 'I')][0], 10)
print('  share of queries whose BM25 top-10 changed S->I:', {t: round(v, 3) for t, v in summary['r1'].get('changed_top10_share', {}).items()})

# ---------- R3: hybrid + rerank ----------
print('\n== R3 hybrid + rerank (I - S)')
r3 = {}
for tag, d in SETS:
    for g in ('S', 'I'):
        for arm, pat in (('raw', 'raw'), ('q8', f'g0d-{tag}*')):
            rows = bench_results(g, d, pat)
            if rows:
                r3[(tag, g, arm)] = (rows, per_query(rows, rels[tag]))
for tag, _ in SETS:
    for arm in ('raw', 'q8'):
        cells = []
        for g in ('S', 'I'):
            if (tag, g, arm) in r3:
                pq = r3[(tag, g, arm)][1]
                cells.append(f"{g}: nDCG@10 {mean_metric(pq, 'ndcg10'):.4f} R@30 {mean_metric(pq, 'r30'):.4f}")
                errs = sum(1 for m in pq.values() if m is None)
                if errs:
                    cells.append(f'{g} retrieval errors: {errs}')
        print(f'  {tag:8s} {arm:4s} ' + '  |  '.join(cells))
for arm in ('raw', 'q8'):
    arms = {tag: (r3[(tag, 'I', arm)][1], r3[(tag, 'S', arm)][1]) for tag, _ in SETS if (tag, 'I', arm) in r3 and (tag, 'S', arm) in r3}
    if len(arms) < 3:
        print(f'  {arm}: incomplete ({len(arms)}/3 datasets)'); continue
    summary['r3'][arm] = {}
    for metric in METRICS:
        r = paired(arms, metric)
        summary['r3'][arm][metric] = r
        per = ' '.join(f'{t}={v:+.4f}' for t, v in r['per_dataset'].items())
        print(f"  {arm:4s} {metric:6s} {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {'SIG' if r['significant'] else 'n.s.'}  ({per})")
    summary['r3'][arm]['changed_top10_share'] = {t: changed_share(r3[(t, 'S', arm)][0], r3[(t, 'I', arm)][0], 10) for t in arms}
if 'q8' in summary['r3']:
    q = summary['r3']['q8']
    non_inf = q['ndcg10']['lo'] > MARGIN['ndcg10'] and q['r30']['lo'] > MARGIN['r30']
    summary['decision'] = {'non_inferior': non_inf,
                           'ndcg10_lo': q['ndcg10']['lo'], 'r30_lo': q['r30']['lo'], 'margins': MARGIN}
    print(f"\n  DECISION (q8, macro): nDCG@10 lo={q['ndcg10']['lo']:+.4f} (> {MARGIN['ndcg10']}), R@30 lo={q['r30']['lo']:+.4f} (> {MARGIN['r30']})"
          f" -> {'NON-INFERIOR' if non_inf else 'NOT NON-INFERIOR'}")

# ---------- R4: strong-signal ----------
print('\n== R4 strong-signal')
prod = {}
for tag, d in SETS:
    for g, v in (('S', 'S'), ('I', f'I-{tag}')):
        rows = probe(v, d, 'strong')
        if not rows or (tag, g, 'q8') not in r3:
            continue
        fired = [q for q, r in rows.items() if r['strong']]
        q8rows = r3[(tag, g, 'q8')][0]
        mixed = {q: (rows[q] if rows[q]['strong'] else q8rows[q]) for q in q8rows}
        prod[(tag, g)] = per_query(mixed, rels[tag])
        tops = sorted(r['top'] for r in rows.values())
        summary['r4'].setdefault(tag, {})[g] = {'fired': len(fired), 'queries': len(rows),
                                                 'top_score_median': tops[len(tops) // 2]}
        print(f"  {tag:8s} {g}: fired {len(fired)}/{len(rows)}  median top BM25 score {tops[len(tops)//2]:.3f}")
arms = {tag: (prod[(tag, 'I')], prod[(tag, 'S')]) for tag, _ in SETS if (tag, 'I') in prod and (tag, 'S') in prod}
if len(arms) == 3:
    summary['r4']['prod_like_I_minus_S'] = {}
    for metric in METRICS:
        r = paired(arms, metric)
        summary['r4']['prod_like_I_minus_S'][metric] = r
        print(f"  prod-like I-S {metric:6s} {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {'SIG' if r['significant'] else 'n.s.'}")
    for g in ('S', 'I'):
        arms_g = {tag: (prod[(tag, g)], r3[(tag, g, 'q8')][1]) for tag, _ in SETS}
        summary['r4'][f'shortcut_effect_{g}'] = {}
        for metric in ('ndcg10', 'r30'):
            r = paired(arms_g, metric)
            summary['r4'][f'shortcut_effect_{g}'][metric] = r
            print(f"  shortcut effect within {g} (prod-like - q8) {metric:6s} {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {'SIG' if r['significant'] else 'n.s.'}")

json.dump(summary, open(f'{X}/summary.json', 'w'), indent=2)
print(f'\nwrote {X}/summary.json')
