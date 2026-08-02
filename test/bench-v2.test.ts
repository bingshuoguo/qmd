import { afterEach, describe, expect, test, vi } from "vitest";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  createStore,
  hybridQuery,
  retrieveForBenchmark,
  type SearchResult,
  type Store,
} from "../src/store.js";
import {
  inspectBenchmarkIndex,
  computeBenchmarkConfigSha256,
  computeGitDiffSha256,
  runBenchmarkCommand,
  runBenchmarkV2,
  type BenchmarkV2Dependencies,
} from "../src/bench/bench.js";
import type { QMDStore } from "../src/index.js";
import { loadBenchmarkV2, loadRetrievalProfile } from "../src/bench/qrels.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

function result(name: string, source: "fts" | "vec"): SearchResult {
  return {
    filepath: `qmd://bench/${name}.md`,
    displayPath: `${name}.md`,
    title: name,
    context: null,
    hash: `${name}-hash`,
    docid: name,
    collectionName: "bench",
    modifiedAt: "2026-01-01T00:00:00.000Z",
    bodyLength: 20,
    body: `# ${name}\n\nbenchmark text`,
    score: 1,
    source,
  };
}

function controlledStore(): {
  store: Store;
  fts: ReturnType<typeof vi.fn>;
  vector: ReturnType<typeof vi.fn>;
  expand: ReturnType<typeof vi.fn>;
  rerank: ReturnType<typeof vi.fn>;
} {
  const directory = mkdtempSync(join(tmpdir(), "qmd-bench-v2-"));
  temporaryDirectories.push(directory);
  const store = createStore(join(directory, "index.sqlite"));
  store.db.exec("CREATE TABLE vectors_vec (hash_seq TEXT PRIMARY KEY, embedding BLOB)");

  const fts = vi.fn((query: string, limit?: number, collection?: string) => {
    const name = query === "original" ? "original-fts" : `lex-${query}`;
    return [result(name, "fts")];
  });
  const vector = vi.fn(async (query: string) => [result(`vec-${query}`, "vec")]);
  const expand = vi.fn(async () => {
    throw new Error("benchmark must not call online expansion");
  });
  const rerank = vi.fn(async (
    _query: string,
    documents: { file: string; text: string }[],
  ) => documents.map((document, index) => ({ file: document.file, score: 1 - index / 100 })));
  store.searchFTS = fts as Store["searchFTS"];
  store.searchVec = vector as Store["searchVec"];
  store.expandQuery = expand as Store["expandQuery"];
  store.rerank = rerank as Store["rerank"];
  store.llm = {
    embedModelName: "test-embedding",
    embedBatch: vi.fn(async (texts: string[]) =>
      texts.map(() => ({ embedding: [1, 2, 3], model: "test-embedding" }))
    ),
  } as never;
  return { store, fts, vector, expand, rerank };
}

describe("controlled benchmark retrieval seam", () => {
  test("raw still executes original BM25 and vector retrieval", async () => {
    const { store, fts, vector, expand } = controlledStore();
    try {
      const results = await retrieveForBenchmark(store, {
        originalQuery: "original",
        expansions: [],
        collection: "bench",
        resultLimit: 30,
        perListLimit: 30,
        candidateLimit: 40,
        rerank: false,
      });

      expect(fts).toHaveBeenCalledTimes(1);
      expect(fts).toHaveBeenCalledWith("original", 30, "bench");
      expect(vector).toHaveBeenCalledTimes(1);
      expect(vector.mock.calls[0]?.[0]).toBe("original");
      expect(vector.mock.calls[0]?.[2]).toBe(30);
      expect(vector.mock.calls[0]?.[3]).toBe("bench");
      expect(results.map(item => item.file)).toEqual(expect.arrayContaining([
        "qmd://bench/original-fts.md",
        "qmd://bench/vec-original.md",
      ]));
      expect(expand).not.toHaveBeenCalled();
    } finally {
      store.close();
    }
  });

  test("keeps original retrieval and routes lex, vec, and hyde expansions", async () => {
    const { store, fts, vector, expand } = controlledStore();
    try {
      const results = await retrieveForBenchmark(store, {
        originalQuery: "original",
        expansions: [
          { type: "lex", query: "keywords" },
          { type: "vec", query: "semantic" },
          { type: "hyde", query: "hypothetical" },
        ],
        collection: "bench",
        resultLimit: 30,
        perListLimit: 17,
        candidateLimit: 40,
        rerank: false,
      });

      expect(fts.mock.calls.map(call => call.slice(0, 3))).toEqual([
        ["original", 17, "bench"],
        ["keywords", 17, "bench"],
      ]);
      expect(vector.mock.calls.map(call => [call[0], call[2], call[3]])).toEqual([
        ["original", 17, "bench"],
        ["semantic", 17, "bench"],
        ["hypothetical", 17, "bench"],
      ]);
      expect(results.map(item => item.file)).toEqual(expect.arrayContaining([
        "qmd://bench/original-fts.md",
        "qmd://bench/vec-original.md",
        "qmd://bench/lex-keywords.md",
        "qmd://bench/vec-semantic.md",
        "qmd://bench/vec-hypothetical.md",
      ]));
      expect(expand).not.toHaveBeenCalled();
    } finally {
      store.close();
    }
  });

  test("rerank switch changes only the rerank stage", async () => {
    const first = controlledStore();
    try {
      await retrieveForBenchmark(first.store, {
        originalQuery: "original",
        expansions: [],
        collection: "bench",
        resultLimit: 30,
        perListLimit: 30,
        candidateLimit: 40,
        rerank: false,
      });
      expect(first.rerank).not.toHaveBeenCalled();
    } finally {
      first.store.close();
    }

    const second = controlledStore();
    try {
      await retrieveForBenchmark(second.store, {
        originalQuery: "original",
        expansions: [],
        collection: "bench",
        resultLimit: 30,
        perListLimit: 30,
        candidateLimit: 40,
        rerank: true,
      });
      expect(second.rerank).toHaveBeenCalledTimes(1);
    } finally {
      second.store.close();
    }
  });

  test("production hybridQuery retains online expansion behavior", async () => {
    const { store, expand } = controlledStore();
    expand.mockReset();
    expand.mockResolvedValue([]);
    store.searchFTS = vi.fn(() => []) as Store["searchFTS"];
    try {
      await hybridQuery(store, "production", { skipRerank: true });
      expect(expand).toHaveBeenCalledTimes(1);
      expect(expand).toHaveBeenCalledWith("production", undefined, undefined);
    } finally {
      store.close();
    }
  });
});

function writeV2Fixture(documentCount = 2): string {
  const parent = mkdtempSync(join(tmpdir(), "qmd-bench-v2-runner-"));
  temporaryDirectories.push(parent);
  const root = join(parent, "qmd-expansion-scifact-v1");
  mkdirSync(root);
  mkdirSync(join(root, "expansions"));
  const sourceQrels = [
    "query-id\tcorpus-id\tscore",
    "q1\td1\t1",
    "q2\td2\t1",
    "",
  ].join("\n");
  const excludedQids = "[]\n";
  const leakageReport = "{}\n";
  const hash = (value: string): string =>
    createHash("sha256").update(value).digest("hex");
  writeFileSync(join(root, "benchmark.yaml"), [
    "benchmark_id: qmd-expansion-scifact-v1",
    "source:",
    "  url: https://example.test/scifact.zip",
    `  archive_md5: "${"0".repeat(32)}"`,
    "  split: test",
    `source_qrels_sha256: ${hash(sourceQrels)}`,
    `excluded_qids_sha256: ${hash(excludedQids)}`,
    `leakage_report_sha256: ${hash(leakageReport)}`,
    `converted_data_sha256: "${"4".repeat(64)}"`,
    "qrels:",
    "  relevant_threshold: 1",
    "  unjudged: nonrelevant",
    "  graded: false",
    "cutoffs: [1, 3, 5, 10, 20, 30]",
    "metrics: [recall_at_cutoffs, mrr_at_10, ndcg_at_10]",
    "",
  ].join("\n"));
  writeFileSync(join(root, "retrieval-profile.yaml"), [
    "profile_id: qmd-scifact-controlled-v1",
    "collection_name: qmd-expansion-scifact-v1",
    "collection_root: corpus",
    "embedding_model: test-embedding",
    "reranker_model: null",
    "result_limit: 30",
    "per_list_limit: 30",
    "candidate_limit: 40",
    "rerank: false",
    "auto_expand: false",
    "strong_signal_bypass: false",
    "",
  ].join("\n"));
  writeFileSync(join(root, "queries.jsonl"), [
    JSON.stringify({ qid: "q1", query: "first" }),
    JSON.stringify({ qid: "q2", query: "second" }),
    "",
  ].join("\n"));
  writeFileSync(join(root, "documents.jsonl"), [
    ...Array.from({ length: documentCount }, (_, index) => JSON.stringify({
      doc_id: `d${index + 1}`,
      path: `d${index + 1}.md`,
    })),
    "",
  ].join("\n"));
  writeFileSync(join(root, "source-qrels.tsv"), sourceQrels);
  writeFileSync(join(root, "qrels.tsv"), sourceQrels);
  writeFileSync(join(root, "excluded-qids.json"), excludedQids);
  writeFileSync(join(root, "leakage-report.json"), leakageReport);
  writeFileSync(join(root, "index-manifest.json"), "{}\n");
  writeFileSync(join(root, "expansions", "candidate.jsonl"), [
    JSON.stringify({
      qid: "q1",
      query: "first",
      status: "ok",
      raw_output: "lex: first terms\nvec: first semantic",
      output: [["lex", "first terms"], ["vec", "first semantic"]],
      fallback_used: true,
      error: null,
    }),
    JSON.stringify({
      qid: "q2",
      query: "second",
      status: "format_error",
      raw_output: "invalid",
      output: [],
      fallback_used: false,
      error: "invalid format",
    }),
    "",
  ].join("\n"));
  return root;
}

function runnerDependencies(
  retrieve: NonNullable<BenchmarkV2Dependencies["retrieve"]>,
): BenchmarkV2Dependencies {
  const controlled = controlledStore();
  const qmdStore = {
    internal: controlled.store,
    dbPath: controlled.store.dbPath,
    close: async () => controlled.store.close(),
  } as QMDStore;
  return {
    openStore: async () => qmdStore,
    verifyIndex: () => ({
      collection_name: "qmd-expansion-scifact-v1",
      collection_root: "/fixture/corpus",
      documents_sha256: "documents",
      embedding_model: "test-embedding",
      embedding_fingerprint: "fingerprint",
      document_count: 2,
      vector_document_count: 2,
      vector_chunk_count: 2,
      pending_embedding_count: 0,
      index_fingerprint: "index-fingerprint",
    }),
    metadata: () => ({
      qmd_commit: "test-commit",
      qmd_dirty: false,
      qmd_diff_sha256: null,
      qmd_config_sha256: "config",
      embedding_artifact_sha256: "embedding-artifact",
      reranker_artifact_sha256: null,
      runtime: {
        qmd: "test",
        bun_or_node: "test",
        sqlite: "test",
        sqlite_vec: "test",
        platform: "test",
      },
    }),
    retrieve,
  };
}

function benchmarkResult(file: string): Awaited<ReturnType<typeof retrieveForBenchmark>>[number] {
  return {
    file,
    displayPath: file,
    title: file,
    body: "",
    bestChunk: "",
    bestChunkPos: 0,
    score: 1,
    context: null,
    docid: file,
  };
}

describe("v2 runner artifacts", () => {
  test("config and dirty-state hashes are stable and content-sensitive", () => {
    const root = writeV2Fixture();
    const benchmark = loadBenchmarkV2(root);
    const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
    const firstConfig = computeBenchmarkConfigSha256(profile, {
      QMD_EMBED_PARALLELISM: "1",
      IGNORED_VARIABLE: "first",
    });
    expect(computeBenchmarkConfigSha256(profile, {
      IGNORED_VARIABLE: "second",
      QMD_EMBED_PARALLELISM: "1",
    })).toBe(firstConfig);
    expect(computeBenchmarkConfigSha256(profile, {
      QMD_EMBED_PARALLELISM: "2",
    })).not.toBe(firstConfig);

    const files = [
      { path: "z.txt", bytes: Buffer.from("z") },
      { path: "a.txt", bytes: Buffer.from("a") },
    ];
    const firstDiff = computeGitDiffSha256("tracked", files);
    expect(computeGitDiffSha256("tracked", [...files].reverse())).toBe(firstDiff);
    expect(computeGitDiffSha256("changed", files)).not.toBe(firstDiff);
    expect(computeGitDiffSha256("tracked", [
      files[0]!,
      { path: "a.txt", bytes: Buffer.from("changed") },
    ])).not.toBe(firstDiff);
  });

  test("missing index fails without creating a SQLite file", async () => {
    const root = writeV2Fixture();
    const missingDb = join(root, "missing.sqlite");
    await expect(runBenchmarkV2(root, { run: "raw", dbPath: missingDb }))
      .rejects.toThrow("Benchmark index does not exist");
    expect(existsSync(missingDb)).toBe(false);
  });

  test("index manifest fingerprint is collection-scoped and covers vector bytes", () => {
    const root = writeV2Fixture();
    mkdirSync(join(root, "corpus"));
    const { store } = controlledStore();
    const now = "2026-01-01T00:00:00.000Z";
    store.db.prepare(`
      INSERT INTO store_collections (name, path, pattern)
      VALUES (?, ?, ?)
    `).run("qmd-expansion-scifact-v1", join(root, "corpus"), "**/*.md");
    for (const [path, hashValue, byte] of [
      ["d2.md", "hash-2", 2],
      ["d1.md", "hash-1", 1],
    ] as const) {
      store.db.prepare("INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)")
        .run(hashValue, `# ${path}`, now);
      store.db.prepare(`
        INSERT INTO documents
          (collection, path, title, hash, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
      `).run("qmd-expansion-scifact-v1", path, path, hashValue, now, now);
      store.db.prepare(`
        INSERT INTO content_vectors
          (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
        VALUES (?, 0, 0, ?, ?, 1, ?)
      `).run(hashValue, "test-embedding", "fingerprint", now);
      store.db.prepare("INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)")
        .run(`${hashValue}_0`, Buffer.from([byte, 0, 0, 0]));
    }
    try {
      const benchmark = loadBenchmarkV2(root);
      const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
      const first = inspectBenchmarkIndex(store, root, benchmark, profile);
      expect(first).toMatchObject({
        document_count: 2,
        vector_document_count: 2,
        vector_chunk_count: 2,
        pending_embedding_count: 0,
        embedding_fingerprint: "fingerprint",
      });
      const embeddingHash = (bytes: Buffer): string =>
        createHash("sha256").update(bytes).digest("hex");
      const vectorRecord = (docId: string, bytes: Buffer): string => `${[
        "V",
        docId,
        "0",
        "0",
        "1",
        "test-embedding",
        "fingerprint",
        embeddingHash(bytes),
      ].join("\0")}\n`;
      const expectedFingerprint = createHash("sha256")
        .update("D\0d1\0d1.md\0hash-1\n")
        .update("D\0d2\0d2.md\0hash-2\n")
        .update(vectorRecord("d1", Buffer.from([1, 0, 0, 0])))
        .update(vectorRecord("d2", Buffer.from([2, 0, 0, 0])))
        .digest("hex");
      expect(first.index_fingerprint).toBe(expectedFingerprint);
      store.db.prepare("UPDATE vectors_vec SET embedding = ? WHERE hash_seq = ?")
        .run(Buffer.from([9, 0, 0, 0]), "hash-2_0");
      const second = inspectBenchmarkIndex(store, root, benchmark, profile);
      expect(second.index_fingerprint).not.toBe(first.index_fingerprint);
    } finally {
      store.close();
    }
  });

  test("writes repeatable raw results and canonical summary", async () => {
    const root = writeV2Fixture(30);
    const dependencies = runnerDependencies(async (_store, options) => {
      const doc = options.originalQuery === "first" ? "d1.md" : "d2.md";
      return [benchmarkResult(`qmd://qmd-expansion-scifact-v1/${doc}`)];
    });
    const onProgress = vi.fn();
    dependencies.onProgress = onProgress;
    const run = await runBenchmarkV2(
      root,
      { run: "raw", dbPath: join(root, "unused.sqlite") },
      dependencies,
    );

    expect(run.status).toBe("completed");
    expect(run.run_id).toBe("raw");
    expect(run.benchmark_manifest_sha256).toBe(
      createHash("sha256").update(readFileSync(join(root, "benchmark.yaml"))).digest("hex"),
    );
    expect(run.qmd_diff_sha256).toBeNull();
    expect(run.expansion_model).toBeNull();
    expect(run.expansions_sha256).toBeNull();
    expect(run.metrics?.recall_at_20).toBe(1);
    expect(run.metrics?.mrr_at_10).toBe(1);
    expect(run.metrics?.ndcg_at_10).toBe(1);
    expect(run.metrics).toMatchObject({
      expansion_pass_rate: null,
      format_error_rate: null,
      generation_error_rate: null,
      fallback_rate: null,
    });
    expect(run.expansion_failures).toMatchObject({
      expansion_pass_count: 0,
      format_error_count: 0,
      generation_error_count: 0,
      fallback_count: 0,
      expansion_pass_rate: null,
      format_error_rate: null,
      generation_error_rate: null,
      fallback_rate: null,
    });
    expect(existsSync(join(root, "runs", "raw.json"))).toBe(true);
    const rows = readFileSync(join(root, "runs", "results", "raw.jsonl"), "utf8")
      .trim().split("\n").map(line => JSON.parse(line));
    expect(rows).toHaveLength(2);
    expect(rows[0].ranking[0]).toEqual({ rank: 1, doc_id: "d1", relevance: 1 });
    expect(rows[0]).toMatchObject({
      retrieval_status: "ok",
      expansion_status: null,
      fallback_used: null,
      expansion_error: null,
      retrieval_error: null,
      diagnostics: [{
        code: "ranking_below_result_limit",
        expected: 30,
        actual: 1,
      }],
    });
    expect(onProgress.mock.calls.map(([progress]) => ({
      completed: progress.completed,
      total: progress.total,
      qid: progress.qid,
      error_count: progress.error_count,
    }))).toEqual([
      { completed: 0, total: 2, qid: null, error_count: 0 },
      { completed: 1, total: 2, qid: "q1", error_count: 0 },
      { completed: 2, total: 2, qid: "q2", error_count: 0 },
    ]);
  });

  test("applies ablation, records model failures, and falls back to raw", async () => {
    const root = writeV2Fixture();
    const seenExpansions: unknown[] = [];
    const dependencies = runnerDependencies(async (_store, options) => {
      seenExpansions.push(options.expansions);
      const doc = options.originalQuery === "first" ? "d1.md" : "d2.md";
      return [benchmarkResult(`qmd://qmd-expansion-scifact-v1/${doc}`)];
    });
    const run = await runBenchmarkV2(root, {
      run: "candidate",
      model: "model-a",
      only: "lex",
      dbPath: join(root, "unused.sqlite"),
    }, dependencies);

    expect(run.run_id).toBe("candidate-model-a-lex-only");
    expect(seenExpansions).toEqual([
      [{ type: "lex", query: "first terms" }],
      [],
    ]);
    expect(run.expansion_failures).toMatchObject({
      expansion_pass_count: 1,
      format_error_count: 1,
      generation_error_count: 0,
      fallback_count: 1,
      expansion_pass_rate: 0.5,
      format_error_rate: 0.5,
      generation_error_rate: 0,
      fallback_rate: 1,
    });
    expect(run.metrics).toMatchObject({
      expansion_pass_rate: 0.5,
      format_error_rate: 0.5,
      generation_error_rate: 0,
      fallback_rate: 1,
    });
    const rows = readFileSync(
      join(root, "runs", "results", "candidate-model-a-lex-only.jsonl"),
      "utf8",
    ).trim().split("\n").map(line => JSON.parse(line));
    expect(rows[0]).toMatchObject({
      retrieval_status: "ok",
      expansion_status: "ok",
      fallback_used: true,
      expansion_error: null,
      retrieval_error: null,
    });
    expect(rows[1]).toMatchObject({
      retrieval_status: "ok",
      expansion_status: "format_error",
      fallback_used: false,
      expansion_error: "invalid format",
      retrieval_error: null,
    });
  });

  test("uses a null fallback rate when no expansion succeeds", async () => {
    const root = writeV2Fixture();
    writeFileSync(join(root, "expansions", "candidate.jsonl"), [
      JSON.stringify({
        qid: "q1",
        query: "first",
        status: "format_error",
        raw_output: "invalid",
        output: [],
        fallback_used: false,
        error: "invalid format",
      }),
      JSON.stringify({
        qid: "q2",
        query: "second",
        status: "generation_error",
        raw_output: "",
        output: [],
        fallback_used: false,
        error: "generation failed",
      }),
      "",
    ].join("\n"));
    const dependencies = runnerDependencies(async (_store, options) => {
      const doc = options.originalQuery === "first" ? "d1.md" : "d2.md";
      return [benchmarkResult(`qmd://qmd-expansion-scifact-v1/${doc}`)];
    });
    const run = await runBenchmarkV2(root, {
      run: "candidate",
      model: "model-a",
      dbPath: join(root, "unused.sqlite"),
    }, dependencies);

    expect(run.metrics).toMatchObject({
      expansion_pass_rate: 0,
      format_error_rate: 0.5,
      generation_error_rate: 0.5,
      fallback_rate: null,
    });
    expect(run.expansion_failures).toMatchObject({
      expansion_pass_count: 0,
      fallback_count: 0,
      fallback_rate: null,
    });
  });

  test("backend error marks run failed and suppresses official summary", async () => {
    const root = writeV2Fixture();
    const dependencies = runnerDependencies(async (_store, options) => {
      if (options.originalQuery === "second") throw new Error("vector backend unavailable");
      return [benchmarkResult("qmd://qmd-expansion-scifact-v1/d1.md")];
    });
    const run = await runBenchmarkV2(
      root,
      { run: "raw", dbPath: join(root, "unused.sqlite") },
      dependencies,
    );
    expect(run.status).toBe("failed");
    expect(run.metrics).toBeNull();
    const rows = readFileSync(join(root, "runs", "results", "raw.jsonl"), "utf8")
      .trim().split("\n").map(line => JSON.parse(line));
    expect(rows[1]).toMatchObject({
      qid: "q2",
      retrieval_status: "error",
      expansion_status: null,
      metrics: null,
      fallback_used: null,
      expansion_error: null,
      retrieval_error: "vector backend unavailable",
    });
  });

  test("directory dispatch requires --run while v1 JSON dispatch remains legacy", async () => {
    const root = writeV2Fixture();
    await expect(runBenchmarkCommand(root)).rejects.toThrow("requires --run");

    const legacyPath = join(root, "legacy.json");
    writeFileSync(legacyPath, JSON.stringify({
      description: "legacy",
      version: 1,
      queries: [],
    }));
    const legacy = await runBenchmarkCommand(legacyPath, {
      dbPath: join(root, "legacy.sqlite"),
      backends: [],
      json: true,
    });
    expect("fixture" in legacy && legacy.fixture).toBe(resolve(legacyPath));
  });
});
