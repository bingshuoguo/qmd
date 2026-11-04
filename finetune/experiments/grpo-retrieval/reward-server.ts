#!/usr/bin/env node
/**
 * Retrieval reward server for GRPO.
 *
 * Scores a model completion by running the real qmd retrieval path
 * (`retrieveForBenchmark`, path A: hybrid RRF, rerank forced off) against
 * benchmark qrels. Reward shaping is left to the Python client; this server
 * only returns metrics for the expansion and for the raw query.
 *
 * Usage:
 *   node_modules/.bin/tsx finetune/experiments/grpo-retrieval/reward-server.ts \
 *     --db <index.sqlite> --source fiqa=<benchmark dir> [--source ...] [--port 8787]
 *
 * POST /score  {"items":[{"source_id","qid","completion","truncated"?}]}
 * GET  /health
 */

import { createServer } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { inspectBenchmarkIndex } from "../../../src/bench/bench.js";
import { loadBenchmarkV2, loadRetrievalProfile } from "../../../src/bench/qrels.js";
import { scoreCanonicalRanking, type CanonicalMetrics } from "../../../src/bench/score.js";
import { parseVhExpansionResponse } from "../../../src/query-expansion-parser.js";
import { createStore, retrieveForBenchmark } from "../../../src/store.js";
import { applyLlamaEnvMitigation } from "../../benchmarks/lib/cli.js";

const { values } = parseArgs({
  options: {
    db: { type: "string" },
    source: { type: "string", multiple: true },
    port: { type: "string", default: "8787" },
    "skip-index-verify": { type: "boolean", default: false },
  },
});
if (!values.db || !existsSync(values.db)) throw new Error("--db must point to an existing index");
if (!values.source?.length) throw new Error("At least one --source id=<benchmark dir> is required");

const CUTOFFS = [1, 3, 5, 10, 20, 30];

const sources = new Map(values.source.map(spec => {
  const eq = spec.indexOf("=");
  if (eq <= 0) throw new Error(`Invalid --source ${spec}; expected id=<dir>`);
  const id = spec.slice(0, eq);
  const root = resolve(spec.slice(eq + 1));
  const benchmark = loadBenchmarkV2(root);
  const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
  const relevant = new Map<string, Set<string>>();
  for (const [qid, relevance] of benchmark.relevanceByQuery) {
    const docs = new Set([...relevance].filter(([, value]) => value === 1).map(([docId]) => docId));
    if (docs.size > 0) relevant.set(qid, docs);
  }
  return [id, {
    root,
    benchmark,
    profile,
    relevant,
    queryByQid: new Map(benchmark.queries.map(query => [query.qid, query.query])),
    docIdByPath: new Map(benchmark.documents.map(document => [document.path, document.doc_id])),
  }] as const;
}));

// Path A is the target: every source must share the same limits, rerank is forced off.
const settings = (id: string) => {
  const p = sources.get(id)!.profile;
  return { result_limit: p.result_limit, per_list_limit: p.per_list_limit, candidate_limit: p.candidate_limit };
};
const sourceIds = [...sources.keys()];
const shared = settings(sourceIds[0]!);
for (const id of sourceIds) {
  if (JSON.stringify(settings(id)) !== JSON.stringify(shared)) {
    throw new Error(`${id}: retrieval limits differ from ${sourceIds[0]}`);
  }
}
const embeddingModels = new Set(sourceIds.map(id => sources.get(id)!.profile.embedding_model));
if (embeddingModels.size !== 1) throw new Error("All sources must use the same embedding model");

applyLlamaEnvMitigation();
const { LlamaCpp } = await import("../../../src/llm.js");
const store = createStore(resolve(values.db));
const llm = new LlamaCpp({
  embedModel: [...embeddingModels][0],
  // Long-lived server: never dispose models on idle (a sleep/wake clock jump
  // disposed them mid-run in the G0 rerank round).
  disposeModelsOnInactivity: false,
});
store.llm = llm;

if (!values["skip-index-verify"]) {
  for (const [id, source] of sources) {
    const frozen = JSON.parse(readFileSync(join(source.root, "index-manifest.json"), "utf8"));
    const actual = inspectBenchmarkIndex(store, source.root, source.benchmark, source.profile);
    if (JSON.stringify(actual) !== JSON.stringify(frozen)) {
      throw new Error(`${id}: index does not match frozen index-manifest.json`);
    }
    if (actual.pending_embedding_count !== 0) throw new Error(`${id}: embeddings are incomplete`);
    process.stderr.write(`[reward] verified index for ${id}\n`);
  }
}

type ScoreItem = { source_id: string; qid: string; completion: string; truncated?: boolean };
type ScoreResult = {
  source_id: string;
  qid: string;
  status: "ok" | "format_error";
  error: string | null;
  metrics: CanonicalMetrics;
  raw_metrics: CanonicalMetrics;
  ms: number;
};

async function retrieve(sourceId: string, qid: string, expansions: [string, string][]): Promise<CanonicalMetrics> {
  const source = sources.get(sourceId)!;
  const results = await retrieveForBenchmark(store, {
    originalQuery: source.queryByQid.get(qid)!,
    expansions: expansions.map(([type, query]) => ({ type: type as "vec" | "hyde", query })),
    collection: source.profile.collection_name,
    resultLimit: shared.result_limit,
    perListLimit: shared.per_list_limit,
    candidateLimit: shared.candidate_limit,
    rerank: false,
  });
  const prefix = `qmd://${source.profile.collection_name}/`;
  const docIds: string[] = [];
  const seen = new Set<string>();
  for (const result of results) {
    if (!result.file.startsWith(prefix)) throw new Error(`Cross-collection result: ${result.file}`);
    const docId = source.docIdByPath.get(result.file.slice(prefix.length));
    if (!docId) throw new Error(`Unknown result path: ${result.file}`);
    if (!seen.has(docId)) { seen.add(docId); docIds.push(docId); }
  }
  return scoreCanonicalRanking(docIds, source.relevant.get(qid)!, CUTOFFS);
}

const rawCache = new Map<string, CanonicalMetrics>();

async function scoreItem(item: ScoreItem): Promise<ScoreResult> {
  const start = performance.now();
  const source = sources.get(item.source_id);
  if (!source) throw new Error(`Unknown source_id: ${item.source_id}`);
  if (!source.queryByQid.has(item.qid)) throw new Error(`${item.source_id}: unknown qid ${item.qid}`);
  if (!source.relevant.has(item.qid)) throw new Error(`${item.source_id}: qid ${item.qid} has no relevant documents`);

  const key = `${item.source_id}\0${item.qid}`;
  let raw = rawCache.get(key);
  if (!raw) {
    raw = await retrieve(item.source_id, item.qid, []);
    rawCache.set(key, raw);
  }

  // Same acceptance rule as serving (parseVhExpansion): a truncated response is
  // a format error even if its tail parses; format errors fall back to raw.
  const parsed = parseVhExpansionResponse(item.completion);
  const error = item.truncated
    ? "truncated"
    : parsed.diagnostics.length > 0
      ? `${parsed.diagnostics[0]!.code} at line ${parsed.diagnostics[0]!.line}`
      : null;
  const metrics = error ? raw : await retrieve(item.source_id, item.qid, parsed.output as [string, string][]);
  return {
    source_id: item.source_id,
    qid: item.qid,
    status: error ? "format_error" : "ok",
    error,
    metrics,
    raw_metrics: raw,
    ms: Math.round(performance.now() - start),
  };
}

// node-llama-cpp embedding is not safe to run concurrently: serialize all work.
let queue: Promise<unknown> = Promise.resolve();
function serialized<T>(task: () => Promise<T>): Promise<T> {
  const next = queue.then(task, task);
  queue = next.catch(() => undefined);
  return next;
}

const server = createServer((request, response) => {
  const reply = (status: number, body: unknown) => {
    response.writeHead(status, { "content-type": "application/json" });
    response.end(JSON.stringify(body));
  };
  if (request.method === "GET" && request.url === "/health") {
    reply(200, {
      ok: true,
      sources: Object.fromEntries(sourceIds.map(id => [id, {
        collection: sources.get(id)!.profile.collection_name,
        scorable_queries: sources.get(id)!.relevant.size,
      }])),
      retrieval: { ...shared, rerank: false },
      raw_cache: rawCache.size,
    });
    return;
  }
  if (request.method !== "POST" || request.url !== "/score") {
    reply(404, { error: "not found" });
    return;
  }
  const chunks: Buffer[] = [];
  request.on("data", chunk => chunks.push(chunk));
  request.on("end", () => {
    serialized(async () => {
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as { items: ScoreItem[] };
      const results: ScoreResult[] = [];
      for (const item of body.items) results.push(await scoreItem(item));
      return results;
    }).then(
      results => reply(200, { results }),
      error => reply(400, { error: error instanceof Error ? error.message : String(error) }),
    );
  });
});

const shutdown = async () => {
  server.close();
  await llm.dispose();
  store.close();
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

server.listen(Number(values.port), "127.0.0.1", () => {
  process.stderr.write(`[reward] listening on 127.0.0.1:${values.port} sources=${sourceIds.join(",")}\n`);
});
