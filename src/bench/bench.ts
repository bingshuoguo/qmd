/**
 * QMD Benchmark Harness
 *
 * Runs queries from a fixture file against multiple search backends
 * and measures precision@k, recall, MRR, F1, and latency.
 *
 * Usage:
 *   qmd bench <fixture.json> [--json] [--collection <name>]
 *
 * Backends tested:
 *   - bm25: BM25 keyword search (searchLex)
 *   - vector: Vector similarity search (searchVector)
 *   - hybrid: BM25 + vector RRF fusion without reranking
 *   - full: Full hybrid pipeline with LLM reranking
 */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { platform, arch } from "node:os";
import {
  createStore,
  getDefaultDbPath,
  type QMDStore,
  type SearchResult,
  type HybridQueryResult,
  type ExpandedQuery,
} from "../index.js";
import {
  DEFAULT_MODEL_CACHE_DIR,
  inspectGgufFile,
  LlamaCpp,
} from "../llm.js";
import {
  createStore as createInternalStore,
  getEmbeddingFingerprint,
  retrieveForBenchmark,
  type Store,
} from "../store.js";
import {
  averageCanonicalMetrics,
  scoreCanonicalRanking,
  scoreResults,
  type CanonicalMetrics,
} from "./score.js";
import {
  loadBenchmarkV2,
  loadRetrievalProfile,
  parseExpansionsJsonl,
} from "./qrels.js";
import { validateBenchmarkRunName } from "./run-name.js";
import type {
  BenchmarkFixture,
  BenchmarkQuery,
  BackendResult,
  QueryResult,
  BenchmarkResult,
  BenchmarkExpansion,
  BenchmarkExpansionRecord,
  BenchmarkRunV2,
  BenchmarkRunMetrics,
  BenchmarkVariant,
  CanonicalQueryResult,
  IndexManifestV2,
  LoadedBenchmarkV2,
  RetrievalProfileV2,
} from "./types.js";

type Backend = {
  name: string;
  run: (store: QMDStore, query: BenchmarkQuery, limit: number, collection?: string) => Promise<string[]>;
};

type ParsedStructuredQuery = {
  searches: ExpandedQuery[];
  intent?: string;
};

function parseStructuredQuery(query: string): ParsedStructuredQuery | undefined {
  const lines = query.split("\n").map((line, idx) => ({
    trimmed: line.trim(),
    number: idx + 1,
  })).filter(line => line.trimmed.length > 0);

  if (lines.length === 0) return undefined;

  const prefixRe = /^(lex|vec|hyde):\s*/i;
  const intentRe = /^intent:\s*/i;
  const searches: ExpandedQuery[] = [];
  let intent: string | undefined;

  for (const line of lines) {
    if (intentRe.test(line.trimmed)) {
      if (intent !== undefined) {
        throw new Error(`Line ${line.number}: only one intent: line is allowed per benchmark query.`);
      }
      intent = line.trimmed.replace(intentRe, "").trim();
      if (!intent) {
        throw new Error(`Line ${line.number}: intent: must include text.`);
      }
      continue;
    }

    const match = line.trimmed.match(prefixRe);
    if (match) {
      const type = match[1]!.toLowerCase() as "lex" | "vec" | "hyde";
      const text = line.trimmed.slice(match[0].length).trim();
      if (!text) {
        throw new Error(`Line ${line.number} (${type}:) must include text.`);
      }
      searches.push({ type, query: text, line: line.number });
      continue;
    }

    if (lines.length === 1) {
      return undefined;
    }

    throw new Error(`Line ${line.number} is missing a lex:/vec:/hyde:/intent: prefix.`);
  }

  if (intent && searches.length === 0) {
    throw new Error("intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line.");
  }

  return searches.length > 0 ? { searches, intent } : undefined;
}

function uniqueFiles(files: string[], limit: number): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const file of files) {
    if (seen.has(file)) continue;
    seen.add(file);
    out.push(file);
    if (out.length >= limit) break;
  }
  return out;
}

const BACKENDS: Backend[] = [
  {
    name: "bm25",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const lexQueries = structured?.searches.filter(q => q.type === "lex");
      if (structured) {
        const files: string[] = [];
        for (const lex of lexQueries ?? []) {
          const results = await store.searchLex(lex.query, { limit, collection });
          files.push(...results.map((r: SearchResult) => r.filepath));
        }
        return uniqueFiles(files, limit);
      }

      const results = await store.searchLex(query.query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    },
  },
  {
    name: "vector",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const vectorQueries = structured?.searches.filter(q => q.type === "vec" || q.type === "hyde");
      if (structured) {
        const files: string[] = [];
        for (const vectorQuery of vectorQueries ?? []) {
          const results = await store.searchVector(vectorQuery.query, { limit, collection });
          files.push(...results.map((r: SearchResult) => r.filepath));
        }
        return uniqueFiles(files, limit);
      }

      const results = await store.searchVector(query.query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    },
  },
  {
    name: "hybrid",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const results = structured
        ? await store.search({ queries: structured.searches, intent: structured.intent, limit, collection, rerank: false })
        : await store.search({ query: query.query, limit, collection, rerank: false });
      return results.map((r: HybridQueryResult) => r.file);
    },
  },
  {
    name: "full",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const results = structured
        ? await store.search({ queries: structured.searches, intent: structured.intent, limit, collection, rerank: true })
        : await store.search({ query: query.query, limit, collection, rerank: true });
      return results.map((r: HybridQueryResult) => r.file);
    },
  },
];

async function runQuery(
  store: QMDStore,
  backend: Backend,
  query: BenchmarkQuery,
  collection?: string,
): Promise<BackendResult> {
  const limit = Math.max(query.expected_in_top_k, 10);
  const start = Date.now();

  let resultFiles: string[];
  try {
    resultFiles = await backend.run(store, query, limit, collection);
  } catch {
    // Backend may not be available (e.g., no embeddings for vector search)
    return {
      precision_at_k: 0,
      recall: 0,
      recall_at_1: 0,
      recall_at_3: 0,
      recall_at_5: 0,
      mrr: 0,
      f1: 0,
      hits_at_k: 0,
      total_expected: query.expected_files.length,
      latency_ms: Date.now() - start,
      top_files: [],
      matched_files: [],
      unmatched_expected_files: query.expected_files,
    };
  }

  const latency_ms = Date.now() - start;
  const scores = scoreResults(resultFiles, query.expected_files, query.expected_in_top_k);

  return {
    ...scores,
    total_expected: query.expected_files.length,
    latency_ms,
    top_files: resultFiles.slice(0, 10),
  };
}

function formatTable(results: QueryResult[]): string {
  const lines: string[] = [];
  const pad = (s: string, n: number) => s.slice(0, n).padEnd(n);
  const num = (n: number) => n.toFixed(2).padStart(5);

  lines.push(
    `${pad("Query", 25)} ${pad("Backend", 8)} ${pad("P@k", 6)} ${pad("R@1", 6)} ${pad("R@3", 6)} ${pad("R@5", 6)} ${pad("MRR", 6)} ${pad("F1", 6)} ${pad("ms", 8)}`
  );
  lines.push("-".repeat(88));

  for (const r of results) {
    for (const [backend, br] of Object.entries(r.backends)) {
      lines.push(
        `${pad(r.id, 25)} ${pad(backend, 8)} ${num(br.precision_at_k)} ${num(br.recall_at_1)} ${num(br.recall_at_3)} ${num(br.recall_at_5)} ${num(br.mrr)} ${num(br.f1)} ${String(Math.round(br.latency_ms)).padStart(7)}ms`
      );
    }
    lines.push("");
  }

  return lines.join("\n");
}

function computeSummary(results: QueryResult[]): BenchmarkResult["summary"] {
  const summary: BenchmarkResult["summary"] = {};

  // Collect all backend names
  const backendNames = new Set<string>();
  for (const r of results) {
    for (const name of Object.keys(r.backends)) {
      backendNames.add(name);
    }
  }

  for (const name of Array.from(backendNames)) {
    let totalP = 0, totalR = 0, totalR1 = 0, totalR3 = 0, totalR5 = 0, totalMrr = 0, totalF1 = 0, totalLat = 0, count = 0;
    for (const r of results) {
      const br = r.backends[name];
      if (!br) continue;
      totalP += br.precision_at_k;
      totalR += br.recall;
      totalR1 += br.recall_at_1;
      totalR3 += br.recall_at_3;
      totalR5 += br.recall_at_5;
      totalMrr += br.mrr;
      totalF1 += br.f1;
      totalLat += br.latency_ms;
      count++;
    }
    if (count > 0) {
      summary[name] = {
        avg_precision: totalP / count,
        avg_recall: totalR / count,
        avg_recall_at_1: totalR1 / count,
        avg_recall_at_3: totalR3 / count,
        avg_recall_at_5: totalR5 / count,
        avg_mrr: totalMrr / count,
        avg_f1: totalF1 / count,
        avg_latency_ms: totalLat / count,
      };
    }
  }

  return summary;
}

export async function runBenchmark(
  fixturePath: string,
  options: { json?: boolean; collection?: string; backends?: string[]; dbPath?: string; configPath?: string } = {},
): Promise<BenchmarkResult> {
  // Load fixture
  const raw = readFileSync(resolve(fixturePath), "utf-8");
  const fixture: BenchmarkFixture = JSON.parse(raw);

  if (!fixture.queries || !Array.isArray(fixture.queries)) {
    throw new Error("Invalid fixture: missing 'queries' array");
  }

  // Open store
  const store = await createStore({
    dbPath: options.dbPath ?? getDefaultDbPath(),
    ...(options.configPath ? { configPath: options.configPath } : {}),
  });

  // Filter backends if requested
  const activeBackends = options.backends
    ? BACKENDS.filter(b => options.backends!.includes(b.name))
    : BACKENDS;

  const collection = options.collection ?? fixture.collection;

  // Run queries
  const results: QueryResult[] = [];
  for (const query of fixture.queries) {
    const backends: Record<string, BackendResult> = {};

    for (const backend of activeBackends) {
      if (!options.json) {
        process.stderr.write(`  ${query.id} / ${backend.name}...`);
      }
      backends[backend.name] = await runQuery(store, backend, query, collection);
      if (!options.json) {
        process.stderr.write(` ${Math.round(backends[backend.name]!.latency_ms)}ms\n`);
      }
    }

    results.push({
      id: query.id,
      query: query.query,
      type: query.type,
      backends,
    });
  }

  await store.close();

  const summary = computeSummary(results);
  const timestamp = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);

  const benchResult: BenchmarkResult = {
    timestamp,
    fixture: fixturePath,
    results,
    summary,
  };

  // Output
  if (options.json) {
    console.log(JSON.stringify(benchResult, null, 2));
  } else {
    console.log("\n" + formatTable(results));
    console.log("Summary:");
    console.log("-".repeat(70));
    const pad = (s: string, n: number) => s.slice(0, n).padEnd(n);
    const num = (n: number) => n.toFixed(3).padStart(6);
    for (const [name, s] of Object.entries(summary)) {
      console.log(
        `  ${pad(name, 8)} P@k=${num(s.avg_precision)} R@1=${num(s.avg_recall_at_1)} R@3=${num(s.avg_recall_at_3)} R@5=${num(s.avg_recall_at_5)} MRR=${num(s.avg_mrr)} F1=${num(s.avg_f1)} Avg=${Math.round(s.avg_latency_ms)}ms`
      );
    }
  }

  return benchResult;
}

// ---------------------------------------------------------------------------
// Canonical qrels benchmark (v2)
// ---------------------------------------------------------------------------

export type BenchmarkV2Options = {
  run: BenchmarkVariant;
  model?: string;
  only?: "lex" | "vec" | "hyde";
  json?: boolean;
  dbPath?: string;
  configPath?: string;
};

type RunMetadata = {
  qmd_commit: string;
  qmd_dirty: boolean;
  qmd_diff_sha256: string | null;
  qmd_config_sha256: string;
  embedding_artifact_sha256: string;
  reranker_artifact_sha256: string | null;
  runtime: BenchmarkRunV2["runtime"];
};

export type BenchmarkV2Dependencies = {
  openStore?: (
    dbPath: string,
    configPath?: string,
    profile?: RetrievalProfileV2,
  ) => Promise<QMDStore>;
  verifyIndex?: (
    store: QMDStore,
    benchmarkDir: string,
    benchmark: LoadedBenchmarkV2,
    profile: RetrievalProfileV2,
  ) => IndexManifestV2;
  metadata?: (
    store: QMDStore,
    profile: RetrievalProfileV2,
    configPath?: string,
  ) => RunMetadata;
  retrieve?: typeof retrieveForBenchmark;
  onProgress?: (progress: BenchmarkV2Progress) => void;
};

export type BenchmarkV2Progress = {
  completed: number;
  total: number;
  elapsed_ms: number;
  eta_ms: number | null;
  last_query_ms: number | null;
  qid: string | null;
  error_count: number;
};

function formatProgressDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function renderBenchmarkProgress(progress: BenchmarkV2Progress): void {
  if (!process.stderr.isTTY) return;
  const width = 24;
  const ratio = progress.total === 0 ? 1 : progress.completed / progress.total;
  const filled = Math.round(ratio * width);
  const bar = `${"=".repeat(filled)}${"-".repeat(width - filled)}`;
  const percent = Math.round(ratio * 100).toString().padStart(3);
  const eta = progress.eta_ms === null ? "--" : formatProgressDuration(progress.eta_ms);
  const last = progress.last_query_ms === null
    ? "--"
    : formatProgressDuration(progress.last_query_ms);
  const errors = progress.error_count > 0 ? ` errors=${progress.error_count}` : "";
  process.stderr.write(
    `\x1b[2K\rBenchmark [${bar}] ${percent}% ${progress.completed}/${progress.total}`
    + ` elapsed=${formatProgressDuration(progress.elapsed_ms)} eta=${eta} last=${last}${errors}`
    + (progress.completed === progress.total ? "\n" : ""),
  );
}

function reportBenchmarkProgress(
  reporter: (progress: BenchmarkV2Progress) => void,
  progress: BenchmarkV2Progress,
): void {
  try {
    reporter(progress);
  } catch {
    // Progress reporting must never fail a benchmark run.
  }
}

async function openBenchmarkStore(
  dbPath: string,
  _configPath?: string,
  profile?: RetrievalProfileV2,
): Promise<QMDStore> {
  if (!existsSync(dbPath)) {
    throw new Error(
      `Benchmark index does not exist: ${dbPath}. `
      + "Create the collection and embeddings manually before running the benchmark.",
    );
  }
  const internal = createInternalStore(dbPath);
  const llm = new LlamaCpp({
    embedModel: profile?.embedding_model,
    rerankModel: profile?.reranker_model ?? undefined,
    inactivityTimeoutMs: 5 * 60 * 1000,
    disposeModelsOnInactivity: true,
  });
  internal.llm = llm;
  return {
    internal,
    dbPath,
    close: async () => {
      await llm.dispose();
      internal.close();
    },
  } as QMDStore;
}

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function stableJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) throw new Error("Cannot canonicalize undefined JSON value");
    return encoded;
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0);
  return `{${entries.map(([key, entry]) =>
    `${JSON.stringify(key)}:${canonicalJson(entry)}`
  ).join(",")}}`;
}

const BENCHMARK_ENV_KEYS = [
  "QMD_EMBED_PARALLELISM",
  "QMD_EMBED_CONTEXT_SIZE",
  "QMD_RERANK_CONTEXT_SIZE",
  "QMD_LLAMA_GPU",
  "QMD_FORCE_CPU",
  "QMD_METAL_KEEP_RESIDENCY",
  "GGML_METAL_NO_RESIDENCY",
] as const;

export function computeBenchmarkConfigSha256(
  profile: RetrievalProfileV2,
  environment: Readonly<Record<string, string | undefined>> = process.env,
): string {
  const relevantEnvironment = Object.fromEntries(
    BENCHMARK_ENV_KEYS.map(key => [key, environment[key] ?? null]),
  );
  return sha256(canonicalJson({
    models: {
      embed: profile.embedding_model,
      rerank: profile.reranker_model,
    },
    retrieval: {
      collection_name: profile.collection_name,
      result_limit: profile.result_limit,
      per_list_limit: profile.per_list_limit,
      candidate_limit: profile.candidate_limit,
      rerank: profile.rerank,
      auto_expand: profile.auto_expand,
      strong_signal_bypass: profile.strong_signal_bypass,
    },
    environment: relevantEnvironment,
  }));
}

export function computeGitDiffSha256(
  trackedDiff: string,
  untrackedFiles: readonly { path: string; bytes: Uint8Array }[],
): string {
  const hash = createHash("sha256");
  hash.update("tracked\0", "utf8");
  hash.update(trackedDiff, "utf8");
  for (const file of [...untrackedFiles].sort((left, right) =>
    left.path < right.path ? -1 : left.path > right.path ? 1 : 0
  )) {
    hash.update("\0untracked\0", "utf8");
    hash.update(file.path, "utf8");
    hash.update("\0", "utf8");
    hash.update(file.bytes);
  }
  return hash.digest("hex");
}

function inspectGitState(): Pick<RunMetadata, "qmd_commit" | "qmd_dirty" | "qmd_diff_sha256"> {
  const git = (args: string[]): string =>
    execFileSync("git", args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  const repoRoot = git(["rev-parse", "--show-toplevel"]).trim();
  const qmdCommit = git(["rev-parse", "HEAD"]).trim();
  const status = git(["status", "--porcelain=v1", "-z"]);
  if (status.length === 0) {
    return { qmd_commit: qmdCommit, qmd_dirty: false, qmd_diff_sha256: null };
  }
  const trackedDiff = git(["diff", "--binary", "HEAD", "--"]);
  const untrackedPaths = git(["ls-files", "--others", "--exclude-standard", "-z"])
    .split("\0")
    .filter(path => path.length > 0);
  const untrackedFiles = untrackedPaths.map(path => ({
    path,
    bytes: readFileSync(join(repoRoot, path)),
  }));
  return {
    qmd_commit: qmdCommit,
    qmd_dirty: true,
    qmd_diff_sha256: computeGitDiffSha256(trackedDiff, untrackedFiles),
  };
}

function modelArtifactSha256(model: string): string {
  let artifactPath: string | undefined;
  if (model.startsWith("hf:")) {
    const filename = model.split("/").pop();
    if (filename && existsSync(DEFAULT_MODEL_CACHE_DIR)) {
      artifactPath = readdirSync(DEFAULT_MODEL_CACHE_DIR)
        .filter(name => !name.endsWith(".etag") && name.includes(filename))
        .sort()
        .map(name => join(DEFAULT_MODEL_CACHE_DIR, name))
        .find(path => inspectGgufFile(path).valid);
    }
  } else {
    const candidate = resolve(model);
    if (existsSync(candidate) && statSync(candidate).isFile()) artifactPath = candidate;
  }
  if (!artifactPath) {
    throw new Error(`Model artifact is not available locally or is invalid: ${model}`);
  }
  return sha256(readFileSync(artifactPath));
}

function defaultRunMetadata(
  store: QMDStore,
  profile: RetrievalProfileV2,
  _configPath?: string,
): RunMetadata {
  const packageJson = JSON.parse(
    readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
  ) as { version: string };
  const db = store.internal.db;
  const sqlite = db.prepare("SELECT sqlite_version() AS version")
    .get() as { version: string };
  const sqliteVec = db.prepare("SELECT vec_version() AS version")
    .get() as { version: string };
  const gitState = inspectGitState();
  return {
    ...gitState,
    qmd_config_sha256: computeBenchmarkConfigSha256(profile),
    embedding_artifact_sha256: modelArtifactSha256(profile.embedding_model),
    reranker_artifact_sha256: profile.rerank && profile.reranker_model
      ? modelArtifactSha256(profile.reranker_model)
      : null,
    runtime: {
      qmd: packageJson.version,
      bun_or_node: "Bun" in globalThis
        ? `bun ${(globalThis as { Bun?: { version?: string } }).Bun?.version ?? "unknown"}`
        : `node ${process.version}`,
      sqlite: sqlite.version,
      sqlite_vec: sqliteVec.version,
      platform: `${platform()}-${arch()}`,
    },
  };
}

function vectorBytes(value: unknown): Buffer {
  if (value instanceof Float32Array) {
    const bytes = Buffer.alloc(value.length * Float32Array.BYTES_PER_ELEMENT);
    for (let index = 0; index < value.length; index++) {
      bytes.writeFloatLE(value[index]!, index * Float32Array.BYTES_PER_ELEMENT);
    }
    return bytes;
  }
  const source = Buffer.isBuffer(value)
    ? value
    : value instanceof Uint8Array
      ? Buffer.from(value.buffer, value.byteOffset, value.byteLength)
      : undefined;
  if (!source || source.length % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Index validation could not read float32 vector bytes");
  }
  const bytes = Buffer.alloc(source.length);
  for (let offset = 0; offset < source.length; offset += Float32Array.BYTES_PER_ELEMENT) {
    bytes.writeFloatLE(source.readFloatLE(offset), offset);
  }
  return bytes;
}

export function inspectBenchmarkIndex(
  store: Store,
  benchmarkDir: string,
  benchmark: LoadedBenchmarkV2,
  profile: RetrievalProfileV2,
): IndexManifestV2 {
  const collection = store.db.prepare(
    "SELECT path FROM store_collections WHERE name = ?",
  ).get(profile.collection_name) as { path: string } | undefined;
  if (!collection) {
    throw new Error(`Collection not found in index: ${profile.collection_name}`);
  }
  const collectionRoot = resolve(collection.path);
  const expectedRoot = resolve(benchmarkDir, profile.collection_root);
  if (collectionRoot !== expectedRoot) {
    throw new Error(
      `Collection root mismatch: profile resolves to ${expectedRoot}, index uses ${collectionRoot}`,
    );
  }

  const rows = store.db.prepare(`
    SELECT d.path, d.hash
    FROM documents d
    WHERE d.collection = ? AND d.active = 1
  `).all(profile.collection_name) as { path: string; hash: string }[];
  const expectedPaths = new Set(benchmark.documents.map(document => document.path));
  const actualPaths = new Set(rows.map(row => row.path));
  if (
    expectedPaths.size !== actualPaths.size
    || [...expectedPaths].some(path => !actualPaths.has(path))
  ) {
    throw new Error(
      `Index document mapping mismatch: expected ${expectedPaths.size}, got ${actualPaths.size}`,
    );
  }

  const documentByPath = new Map(
    benchmark.documents.map(document => [document.path, document]),
  );
  const indexedDocuments = rows.map(row => ({
    ...row,
    doc_id: documentByPath.get(row.path)!.doc_id,
  })).sort((left, right) => Buffer.compare(
    Buffer.from(left.doc_id, "utf8"),
    Buffer.from(right.doc_id, "utf8"),
  ));
  const fingerprint = createHash("sha256");
  const fingerprints = new Set<string>();
  const hasVectorTable = !!store.db.prepare(
    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vectors_vec'",
  ).get();
  let vectorDocumentCount = 0;
  let vectorChunkCount = 0;
  let pendingEmbeddingCount = 0;
  const vectorRecords: {
    doc_id: string;
    seq: number;
    pos: number;
    total_chunks: number;
    model: string;
    embed_fingerprint: string;
    embedding_sha256: string;
  }[] = [];
  for (const row of indexedDocuments) {
    fingerprint.update(`D\0${row.doc_id}\0${row.path}\0${row.hash}\n`, "utf8");
    const chunks = store.db.prepare(`
      SELECT seq, pos, total_chunks, embed_fingerprint
      FROM content_vectors
      WHERE hash = ? AND model = ?
      ORDER BY seq
    `).all(row.hash, profile.embedding_model) as {
      seq: number;
      pos: number;
      total_chunks: number;
      embed_fingerprint: string;
    }[];
    let complete = chunks.length > 0;
    const expectedChunkCount = chunks[0]?.total_chunks ?? 0;
    if (chunks.length !== expectedChunkCount) complete = false;
    for (const chunk of chunks) {
      fingerprints.add(chunk.embed_fingerprint);
      const hashSeq = `${row.hash}_${chunk.seq}`;
      const vector = hasVectorTable
        ? store.db.prepare(
          "SELECT embedding FROM vectors_vec WHERE hash_seq = ?",
        ).get(hashSeq) as { embedding: unknown } | undefined
        : undefined;
      if (!vector) complete = false;
      if (vector) {
        vectorRecords.push({
          doc_id: row.doc_id,
          seq: chunk.seq,
          pos: chunk.pos,
          total_chunks: chunk.total_chunks,
          model: profile.embedding_model,
          embed_fingerprint: chunk.embed_fingerprint,
          embedding_sha256: sha256(vectorBytes(vector.embedding)),
        });
      }
    }
    if (complete) {
      vectorDocumentCount++;
      vectorChunkCount += chunks.length;
    } else {
      pendingEmbeddingCount++;
    }
  }
  vectorRecords.sort((left, right) => {
    const docOrder = Buffer.compare(
      Buffer.from(left.doc_id, "utf8"),
      Buffer.from(right.doc_id, "utf8"),
    );
    return docOrder || left.seq - right.seq;
  });
  for (const vector of vectorRecords) {
    fingerprint.update(
      `V\0${vector.doc_id}\0${vector.seq}\0${vector.pos}\0${vector.total_chunks}`
      + `\0${vector.model}\0${vector.embed_fingerprint}\0${vector.embedding_sha256}\n`,
      "utf8",
    );
  }
  if (fingerprints.size > 1) {
    throw new Error(
      `Index contains multiple embedding fingerprints for ${profile.embedding_model}`,
    );
  }
  const embeddingFingerprint = [...fingerprints][0]
    ?? getEmbeddingFingerprint(profile.embedding_model);
  return {
    collection_name: profile.collection_name,
    collection_root: collectionRoot,
    documents_sha256: sha256(
      readFileSync(join(resolve(benchmarkDir), "documents.jsonl")),
    ),
    embedding_model: profile.embedding_model,
    embedding_fingerprint: embeddingFingerprint,
    document_count: rows.length,
    vector_document_count: vectorDocumentCount,
    vector_chunk_count: vectorChunkCount,
    pending_embedding_count: pendingEmbeddingCount,
    index_fingerprint: fingerprint.digest("hex"),
  };
}

function indexManifestsMatch(left: IndexManifestV2, right: IndexManifestV2): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function defaultVerifyIndex(
  store: QMDStore,
  benchmarkDir: string,
  benchmark: LoadedBenchmarkV2,
  profile: RetrievalProfileV2,
): IndexManifestV2 {
  const path = join(resolve(benchmarkDir), "index-manifest.json");
  if (!existsSync(path)) {
    throw new Error(
      `Missing index-manifest.json. Run qmd bench ${benchmarkDir} --check-index first.`,
    );
  }
  const frozen = JSON.parse(readFileSync(path, "utf8")) as IndexManifestV2;
  const actual = inspectBenchmarkIndex(
    store.internal,
    benchmarkDir,
    benchmark,
    profile,
  );
  if (!indexManifestsMatch(frozen, actual)) {
    throw new Error("Current collection/index does not match index-manifest.json");
  }
  if (actual.pending_embedding_count !== 0) {
    throw new Error(
      `Index has ${actual.pending_embedding_count} documents pending embeddings`,
    );
  }
  return actual;
}

export async function writeBenchmarkIndexManifest(
  benchmarkDir: string,
  options: { dbPath?: string; configPath?: string } = {},
  dependencies: BenchmarkV2Dependencies = {},
): Promise<IndexManifestV2> {
  const root = resolve(benchmarkDir);
  const benchmark = loadBenchmarkV2(root);
  const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
  const openStore = dependencies.openStore ?? openBenchmarkStore;
  const store = await openStore(
    options.dbPath ?? getDefaultDbPath(),
    options.configPath,
    profile,
  );
  try {
    const manifest = inspectBenchmarkIndex(store.internal, root, benchmark, profile);
    writeFileSync(join(root, "index-manifest.json"), stableJson(manifest), "utf8");
    return manifest;
  } finally {
    await store.close();
  }
}

function expansionFile(
  root: string,
  variant: BenchmarkVariant,
  benchmark: LoadedBenchmarkV2,
): { records: BenchmarkExpansionRecord[]; sha: string } | null {
  if (variant === "raw") return null;
  const path = join(root, "expansions", `${variant}.jsonl`);
  const bytes = readFileSync(path);
  return {
    records: parseExpansionsJsonl(bytes.toString("utf8"), benchmark.queries),
    sha: sha256(bytes),
  };
}

function runId(variant: BenchmarkVariant, model: string | undefined, only?: string): string {
  if (variant === "raw") return "raw";
  const slug = model!
    .replace(/^hf:/, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
  return `${variant}-${slug}${only ? `-${only}-only` : ""}`;
}

function resultDocId(
  file: string,
  collection: string,
  documentIdByPath: ReadonlyMap<string, string>,
): string {
  const prefix = `qmd://${collection}/`;
  if (!file.startsWith(prefix)) {
    throw new Error(`Result filepath is outside collection "${collection}": ${file}`);
  }
  const path = file.slice(prefix.length);
  const docId = documentIdByPath.get(path);
  if (!docId) throw new Error(`Result filepath is not in documents.jsonl: ${path}`);
  return docId;
}

export async function runBenchmarkV2(
  benchmarkDir: string,
  options: BenchmarkV2Options,
  dependencies: BenchmarkV2Dependencies = {},
): Promise<BenchmarkRunV2> {
  const root = resolve(benchmarkDir);
  validateBenchmarkRunName(options.run);
  const benchmarkManifestBytes = readFileSync(join(root, "benchmark.yaml"));
  const benchmark = loadBenchmarkV2(root);
  const profileBytes = readFileSync(join(root, "retrieval-profile.yaml"));
  const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
  if (profile.collection_name !== benchmark.manifest.benchmark_id) {
    throw new Error("retrieval-profile.yaml collection_name must equal benchmark_id");
  }
  if (options.run === "raw" && (options.model || options.only)) {
    throw new Error("raw run does not accept --model or --only");
  }
  if (options.run !== "raw" && !options.model) {
    throw new Error(`${options.run} run requires --model`);
  }
  const expansions = expansionFile(root, options.run, benchmark);
  const recordsByQid = new Map(
    expansions?.records.map(record => [record.qid, record]) ?? [],
  );
  const openStore = dependencies.openStore ?? openBenchmarkStore;
  const store = await openStore(
    options.dbPath ?? getDefaultDbPath(),
    options.configPath,
    profile,
  );
  const retrieve = dependencies.retrieve ?? retrieveForBenchmark;
  const verifyIndex = dependencies.verifyIndex ?? defaultVerifyIndex;
  const metadata = dependencies.metadata ?? defaultRunMetadata;
  const onProgress = dependencies.onProgress ?? renderBenchmarkProgress;
  try {
    const indexManifest = verifyIndex(store, root, benchmark, profile);
    const environment = metadata(store, profile, options.configPath);
    const documentIdByPath = new Map(
      benchmark.documents.map(document => [document.path, document.doc_id]),
    );
    const id = runId(options.run, options.model, options.only);
    const resultsRelative = `runs/results/${id}.jsonl`;
    const resultsPath = join(root, ...resultsRelative.split("/"));
    const partialPath = `${resultsPath}.partial`;
    mkdirSync(dirname(resultsPath), { recursive: true });
    if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");
    const partialLines = readFileSync(partialPath, "utf8").split("\n");
    if (partialLines.at(-1) === "") {
      partialLines.pop();
    } else {
      try {
        JSON.parse(partialLines.at(-1)!);
      } catch {
        partialLines.pop();
      }
      writeFileSync(partialPath, `${partialLines.join("\n")}${partialLines.length ? "\n" : ""}`, "utf8");
    }
    const queryResults = partialLines
      .filter(line => line.trim())
      .map(line => JSON.parse(line) as CanonicalQueryResult);
    for (let index = 0; index < queryResults.length; index++) {
      if (queryResults[index]!.qid !== benchmark.queries[index]?.qid) {
        throw new Error(`Partial benchmark results are not a valid qid prefix: ${partialPath}`);
      }
    }
    if (queryResults.length > benchmark.queries.length) {
      throw new Error(`Partial benchmark results exceed query count: ${partialPath}`);
    }
    let formatErrorCount = queryResults.filter(r => r.expansion_status === "format_error").length;
    let generationErrorCount = queryResults.filter(r => r.expansion_status === "generation_error").length;
    let fallbackCount = queryResults.filter(r => r.fallback_used).length;
    let expansionPassCount = queryResults.filter(r => r.expansion_status === "ok").length;
    let retrievalErrorCount = queryResults.filter(r => r.retrieval_status === "error").length;
    const total = benchmark.queries.length;
    const runStartedAt = performance.now();

    reportBenchmarkProgress(onProgress, {
      completed: queryResults.length,
      total,
      elapsed_ms: 0,
      eta_ms: null,
      last_query_ms: null,
      qid: null,
      error_count: 0,
    });

    for (let queryIndex = queryResults.length; queryIndex < total; queryIndex++) {
      const query = benchmark.queries[queryIndex]!;
      const expansionRecord = recordsByQid.get(query.qid);
      if (expansionRecord?.status === "ok") expansionPassCount++;
      if (expansionRecord?.status === "format_error") formatErrorCount++;
      if (expansionRecord?.status === "generation_error") generationErrorCount++;
      if (expansionRecord?.fallback_used) fallbackCount++;
      const selected = (expansionRecord?.output ?? [])
        .filter(([type]) => !options.only || type === options.only);
      const start = performance.now();
      try {
        const results = await retrieve(store.internal, {
          originalQuery: query.query,
          expansions: selected.map(([type, text]) => ({ type, query: text })),
          collection: profile.collection_name,
          resultLimit: profile.result_limit,
          perListLimit: profile.per_list_limit,
          candidateLimit: profile.candidate_limit,
          rerank: profile.rerank,
        });
        const docIds: string[] = [];
        const seen = new Set<string>();
        for (const result of results) {
          const docId = resultDocId(result.file, profile.collection_name, documentIdByPath);
          if (seen.has(docId)) continue;
          seen.add(docId);
          docIds.push(docId);
        }
        const relevance = benchmark.relevanceByQuery.get(query.qid)!;
        const relevant = new Set(
          [...relevance.entries()]
            .filter(([, value]) => value === 1)
            .map(([docId]) => docId),
        );
        const metrics = scoreCanonicalRanking(
          docIds,
          relevant,
          benchmark.manifest.cutoffs,
        );
        const diagnostics = benchmark.documents.length >= profile.result_limit
          && docIds.length < profile.result_limit
          ? [{
            code: "ranking_below_result_limit" as const,
            expected: profile.result_limit,
            actual: docIds.length,
          }]
          : [];
        queryResults.push({
          qid: query.qid,
          variant: options.run,
          retrieval_status: "ok",
          expansion_status: expansionRecord?.status ?? null,
          expansions: selected,
          ranking: docIds.map((docId, index) => ({
            rank: index + 1,
            doc_id: docId,
            relevance: relevance.get(docId) ?? null,
          })),
          latency_ms: performance.now() - start,
          metrics,
          fallback_used: expansionRecord?.fallback_used ?? null,
          expansion_error: expansionRecord?.error ?? null,
          retrieval_error: null,
          diagnostics,
        });
      } catch (error) {
        retrievalErrorCount++;
        queryResults.push({
          qid: query.qid,
          variant: options.run,
          retrieval_status: "error",
          expansion_status: expansionRecord?.status ?? null,
          expansions: selected,
          ranking: [],
          latency_ms: performance.now() - start,
          metrics: null,
          fallback_used: expansionRecord?.fallback_used ?? null,
          expansion_error: expansionRecord?.error ?? null,
          retrieval_error: error instanceof Error ? error.message : String(error),
          diagnostics: benchmark.documents.length >= profile.result_limit
            ? [{
              code: "ranking_below_result_limit",
              expected: profile.result_limit,
              actual: 0,
            }]
            : [],
        });
      }
      appendFileSync(
        partialPath,
        `${JSON.stringify(queryResults[queryResults.length - 1])}\n`,
        "utf8",
      );
      const completed = queryIndex + 1;
      const elapsedMs = performance.now() - runStartedAt;
      reportBenchmarkProgress(onProgress, {
        completed,
        total,
        elapsed_ms: elapsedMs,
        eta_ms: (elapsedMs / completed) * (total - completed),
        last_query_ms: queryResults[queryResults.length - 1]!.latency_ms,
        qid: query.qid,
        error_count: retrievalErrorCount,
      });
    }

    const failed = queryResults.some(result => result.retrieval_status === "error");
    const successfulMetrics = queryResults
      .map(result => result.metrics)
      .filter((value): value is CanonicalMetrics => value !== null);
    const rawRun = options.run === "raw";
    const expansionRates = {
      expansion_pass_rate: rawRun ? null : expansionPassCount / total,
      format_error_rate: rawRun ? null : formatErrorCount / total,
      generation_error_rate: rawRun ? null : generationErrorCount / total,
      fallback_rate: rawRun || expansionPassCount === 0
        ? null
        : fallbackCount / expansionPassCount,
    };
    const metrics = failed
      ? null
      : {
        ...averageCanonicalMetrics(successfulMetrics, benchmark.manifest.cutoffs),
        ...expansionRates,
      } satisfies BenchmarkRunMetrics;
    renameSync(partialPath, resultsPath);
    const indexManifestBytes = readFileSync(join(root, "index-manifest.json"));
    const run: BenchmarkRunV2 = {
      run_id: id,
      benchmark_id: benchmark.manifest.benchmark_id,
      benchmark_manifest_sha256: sha256(benchmarkManifestBytes),
      retrieval_profile: profile.profile_id,
      retrieval_profile_sha256: sha256(profileBytes),
      qmd_commit: environment.qmd_commit,
      qmd_dirty: environment.qmd_dirty,
      qmd_diff_sha256: environment.qmd_diff_sha256,
      qmd_config_sha256: environment.qmd_config_sha256,
      collection_name: profile.collection_name,
      collection_root: resolve(root, profile.collection_root),
      index_manifest_sha256: sha256(indexManifestBytes),
      index_fingerprint: indexManifest.index_fingerprint,
      embedding_artifact_sha256: environment.embedding_artifact_sha256,
      reranker_artifact_sha256: environment.reranker_artifact_sha256,
      variant: options.run,
      expansion_model: options.run === "raw" ? null : options.model!,
      expansions_sha256: expansions?.sha ?? null,
      retrieval: {
        result_limit: profile.result_limit,
        per_list_limit: profile.per_list_limit,
        candidate_limit: profile.candidate_limit,
      },
      command: process.argv,
      runtime: environment.runtime,
      status: failed ? "failed" : "completed",
      results: resultsRelative,
      metrics,
      expansion_failures: {
        expansion_pass_count: expansionPassCount,
        format_error_count: formatErrorCount,
        generation_error_count: generationErrorCount,
        fallback_count: fallbackCount,
        ...expansionRates,
      },
    };
    mkdirSync(join(root, "runs"), { recursive: true });
    writeFileSync(join(root, "runs", `${id}.json`), stableJson(run), "utf8");
    if (options.json) console.log(JSON.stringify(run, null, 2));
    return run;
  } finally {
    await store.close();
  }
}

export async function runBenchmarkCommand(
  inputPath: string,
  options: {
    json?: boolean;
    collection?: string;
    backends?: string[];
    dbPath?: string;
    configPath?: string;
    run?: BenchmarkVariant;
    model?: string;
    only?: "lex" | "vec" | "hyde";
  } = {},
): Promise<BenchmarkResult | BenchmarkRunV2> {
  const resolved = resolve(inputPath);
  if (statSync(resolved).isDirectory()) {
    if (!options.run) throw new Error("v2 benchmark directory requires --run");
    return runBenchmarkV2(resolved, {
      run: options.run,
      model: options.model,
      only: options.only,
      json: options.json,
      dbPath: options.dbPath,
      configPath: options.configPath,
    });
  }
  return runBenchmark(resolved, options);
}
