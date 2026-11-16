#!/usr/bin/env node
/**
 * Deep-rank probe for R@30=0 qids: where does the relevant doc actually rank
 * in each retrieval list (BM25 AND on the raw query, vector on the raw query),
 * and what sits at the top instead?
 *
 *   tsx r30zero-probe.ts --db <index.sqlite> --bench <prepared dir> --qids <tsv> --source <id> --out <jsonl> [--depth 1000]
 *
 * Read-only: opens the index through createStore, never writes it.
 */
import { appendFileSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { createStore } from "../../../../src/store.js";
import { applyLlamaEnvMitigation } from "../../../benchmarks/lib/cli.js";

const { values } = parseArgs({
  options: {
    db: { type: "string" }, bench: { type: "string" }, qids: { type: "string" },
    source: { type: "string" }, out: { type: "string" }, depth: { type: "string", default: "1000" },
  },
});
const depth = Number(values.depth);
const bench = resolve(values.bench!);
const profile = readFileSync(join(bench, "retrieval-profile.yaml"), "utf8");
const collection = /collection_name:\s*(\S+)/.exec(profile)![1]!;
const embedModel = /embedding_model:\s*(\S+)/.exec(profile)![1]!;

const queries = new Map<string, string>(readFileSync(join(bench, "queries.jsonl"), "utf8").trim().split("\n")
  .map(l => JSON.parse(l)).map(q => [q.qid, q.query]));
const docByPath = new Map<string, string>(readFileSync(join(bench, "documents.jsonl"), "utf8").trim().split("\n")
  .map(l => JSON.parse(l)).map(d => [d.path, d.doc_id]));
const rels = new Map<string, Set<string>>();
for (const line of readFileSync(join(bench, "qrels.tsv"), "utf8").split("\n")) {
  const [q, d, s] = line.split("\t");
  if (s?.trim() === "1") { if (!rels.has(q!)) rels.set(q!, new Set()); rels.get(q!)!.add(d!); }
}
const qids = readFileSync(values.qids!, "utf8").trim().split("\n").map(l => l.split("\t"))
  .filter(([s]) => s === values.source).map(([, q]) => q!);

applyLlamaEnvMitigation();
const { LlamaCpp } = await import("../../../../src/llm.js");
const store = createStore(resolve(values.db!));
store.llm = new LlamaCpp({ embedModel, disposeModelsOnInactivity: false });

const prefix = `qmd://${collection}/`;
const toDoc = (file: string) => docByPath.get(file.startsWith(prefix) ? file.slice(prefix.length) : file);

for (const qid of qids) {
  const query = queries.get(qid)!;
  const rel = rels.get(qid)!;
  const fts = store.searchFTS(query, depth, collection);
  const vec = await store.searchVec(query, embedModel, depth, collection);
  const rankIn = (list: { filepath: string }[]) =>
    [...rel].map(d => { const i = list.findIndex(r => toDoc(r.filepath) === d); return i === -1 ? null : i + 1; });
  const top = (list: { filepath: string; title: string; score: number }[], n: number) =>
    list.slice(0, n).map(r => ({ doc_id: toDoc(r.filepath), title: r.title, score: Number(r.score.toFixed(4)) }));
  const relScore = [...rel].map(d => vec.find(r => toDoc(r.filepath) === d)?.score ?? null);
  appendFileSync(values.out!, JSON.stringify({
    source_id: values.source, qid, query, rel: [...rel],
    fts_hits: fts.length, fts_rank: rankIn(fts), vec_rank: rankIn(vec),
    vec_rel_score: relScore, vec_top1_score: vec[0]?.score ?? null,
    fts_top: top(fts, 3), vec_top: top(vec, 3),
  }) + "\n");
  process.stderr.write(".");
}
process.stderr.write("\n");
await store.llm.dispose();
store.close();
