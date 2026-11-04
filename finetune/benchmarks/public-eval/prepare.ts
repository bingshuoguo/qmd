#!/usr/bin/env node
/**
 * Prepare the frozen independent evaluation benchmarks (spec section 15).
 *
 * Produces four benchmark directories that `qmd bench` can consume directly.
 * SciFact test is already prepared under benchmarks/qmd-expansion-scifact-v1 and
 * is referenced by the evaluation manifest rather than rebuilt here.
 *
 * Leakage runs in the evaluation direction: any benchmark query whose
 * normalized form collides with a public-distill-v1 training query is excluded
 * and recorded.  The distillation pipeline already excluded these holdouts in
 * the other direction, so the expected count is zero.
 *
 * FiQA has a single corpus; dev and test differ only in qrels.  Both benchmarks
 * therefore point `collection_root` at the corpus the v0 release already
 * converted.  QMD keys embeddings globally by content hash, so registering that
 * same corpus under a second collection name reuses the existing vectors.
 *
 * The zip/qrels helpers below deliberately duplicate the ones in
 * public-distill/prepare.ts: that script produced sealed v0 artifacts and is
 * left untouched.
 *
 * Usage: npx tsx finetune/benchmarks/public-eval/prepare.ts
 */

import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { once } from "node:events";
import {
  copyFileSync,
  createReadStream,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, join, relative, resolve } from "node:path";
import { createInterface } from "node:readline";

import { renderPublicCorpusMarkdown } from "../lib/public-corpus.js";

const DISTILL_ROOT = resolve("finetune/data/public-distill-v0");
const ARCHIVES = join(DISTILL_ROOT, "archives");
const EVAL_ROOT = resolve("finetune/data/public-eval-v1");
const BENCHMARKS = join(EVAL_ROOT, "benchmarks");
const TRAINING_SFT = resolve(
  "finetune/data/public-distill-v1/experiments/public-main-v1/sft.jsonl",
);
const SCIFACT_BENCHMARK = resolve("finetune/benchmarks/qmd-expansion-scifact-v1");

const PROFILE_ID = "qmd-public-eval-v1";
const EMBEDDING_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const RERANKER_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

const FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip";
const FIQA_MD5 = "17918ed23cd04fb15047f73e6c3bd9d9";
const CQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip";
const CQA_MD5 = "4e41456d7df8ee7760a7f866133bda78";

type Query = { qid: string; query: string };
type Qrel = { qid: string; doc_id: string; relevance: 1 };
type Document = { doc_id: string; path: string };

type EvalConfig = {
  /** Must equal collection_name; bench.ts enforces the equality. */
  benchmarkId: string;
  role: "formal" | "diagnostic";
  archive: string;
  archiveUrl: string;
  archiveMd5: string;
  dataset: string;
  split: string;
  domain: string | null;
  queryEntry: string;
  corpusEntry: string;
  qrelsEntry: string;
  threshold: number;
  /** Absolute path to an already-converted corpus to share, or null to convert. */
  reuseCorpusFrom: string | null;
};

const EVALS: EvalConfig[] = [
  {
    benchmarkId: "qmd-eval-v1-fiqa-test",
    role: "formal",
    archive: "fiqa.zip",
    archiveUrl: FIQA_URL,
    archiveMd5: FIQA_MD5,
    dataset: "fiqa",
    split: "test",
    domain: null,
    queryEntry: "fiqa/queries.jsonl",
    corpusEntry: "fiqa/corpus.jsonl",
    qrelsEntry: "fiqa/qrels/test.tsv",
    threshold: 1,
    reuseCorpusFrom: join(DISTILL_ROOT, "prepared/fiqa-train"),
  },
  {
    benchmarkId: "qmd-eval-v1-fiqa-dev",
    role: "diagnostic",
    archive: "fiqa.zip",
    archiveUrl: FIQA_URL,
    archiveMd5: FIQA_MD5,
    dataset: "fiqa",
    split: "dev",
    domain: null,
    queryEntry: "fiqa/queries.jsonl",
    corpusEntry: "fiqa/corpus.jsonl",
    qrelsEntry: "fiqa/qrels/dev.tsv",
    threshold: 1,
    reuseCorpusFrom: join(DISTILL_ROOT, "prepared/fiqa-train"),
  },
  {
    benchmarkId: "qmd-eval-v1-cqadup-android",
    role: "formal",
    archive: "cqadupstack.zip",
    archiveUrl: CQA_URL,
    archiveMd5: CQA_MD5,
    dataset: "cqadupstack",
    split: "test",
    domain: "android",
    queryEntry: "cqadupstack/android/queries.jsonl",
    corpusEntry: "cqadupstack/android/corpus.jsonl",
    qrelsEntry: "cqadupstack/android/qrels/test.tsv",
    threshold: 1,
    reuseCorpusFrom: null,
  },
  {
    benchmarkId: "qmd-eval-v1-cqadup-webmasters",
    role: "formal",
    archive: "cqadupstack.zip",
    archiveUrl: CQA_URL,
    archiveMd5: CQA_MD5,
    dataset: "cqadupstack",
    split: "test",
    domain: "webmasters",
    queryEntry: "cqadupstack/webmasters/queries.jsonl",
    corpusEntry: "cqadupstack/webmasters/corpus.jsonl",
    qrelsEntry: "cqadupstack/webmasters/qrels/test.tsv",
    threshold: 1,
    reuseCorpusFrom: null,
  },
];

function hashBytes(algorithm: "md5" | "sha256", bytes: Uint8Array | string): string {
  return createHash(algorithm).update(bytes).digest("hex");
}

async function hashFile(algorithm: "md5" | "sha256", path: string): Promise<string> {
  const hash = createHash(algorithm);
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

function stableJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function writeJsonl(path: string, values: readonly unknown[]): void {
  writeFileSync(path, `${values.map(value => JSON.stringify(value)).join("\n")}\n`, "utf8");
}

function normalizeQuery(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase("und").trim().replace(/\s+/gu, " ");
}

function safeDocId(value: string): string {
  if (!value || value === "." || value === ".." || /[\\/\0]/u.test(value)) {
    throw new Error(`Unsafe corpus id: ${JSON.stringify(value)}`);
  }
  return value;
}

function unzipLines(archive: string, entry: string): {
  lines: AsyncIterable<string>;
  done: Promise<void>;
} {
  const child = spawn("unzip", ["-p", archive, entry], { stdio: ["ignore", "pipe", "pipe"] });
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", chunk => { stderr += chunk; });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const done = once(child, "close").then(([code]) => {
    if (code !== 0) throw new Error(`unzip ${entry} failed (${code}): ${stderr.trim()}`);
  });
  return { lines, done };
}

async function readZipText(archive: string, entry: string): Promise<string> {
  const { lines, done } = unzipLines(archive, entry);
  const output: string[] = [];
  for await (const line of lines) output.push(line);
  await done;
  return `${output.join("\n")}\n`;
}

async function loadQueries(archive: string, entry: string): Promise<Map<string, string>> {
  const result = new Map<string, string>();
  const { lines, done } = unzipLines(archive, entry);
  for await (const line of lines) {
    if (!line.trim()) continue;
    const value = JSON.parse(line) as { _id?: unknown; text?: unknown };
    if (typeof value._id !== "string" || typeof value.text !== "string") {
      throw new Error(`${entry}: query must contain string _id/text`);
    }
    if (result.has(value._id)) throw new Error(`${entry}: duplicate qid ${value._id}`);
    result.set(value._id, value.text);
  }
  await done;
  return result;
}

function parseQrels(text: string, threshold: number): Qrel[] {
  const result: Qrel[] = [];
  for (const [index, line] of text.split(/\r?\n/u).entries()) {
    if (index === 0 || !line) continue;
    const [qid, docId, scoreText] = line.split("\t");
    const score = Number(scoreText);
    if (!qid || !docId || !Number.isFinite(score)) throw new Error(`Invalid qrels line ${index + 1}`);
    if (score >= threshold) result.push({ qid, doc_id: docId, relevance: 1 });
  }
  return result;
}

/** Convert a BEIR corpus entry into one markdown file per document. */
async function writeCorpus(
  archive: string,
  entry: string,
  outputRoot: string,
): Promise<{ documents: Document[]; sourceSha256: string }> {
  const corpusRoot = join(outputRoot, "corpus");
  mkdirSync(corpusRoot, { recursive: true });
  const documents: Document[] = [];
  const sourceHash = createHash("sha256");
  const { lines, done } = unzipLines(archive, entry);
  for await (const line of lines) {
    if (!line.trim()) continue;
    sourceHash.update(`${line}\n`, "utf8");
    const value = JSON.parse(line) as { _id?: unknown; title?: unknown; text?: unknown };
    if (
      typeof value._id !== "string"
      || typeof value.title !== "string"
      || typeof value.text !== "string"
    ) {
      throw new Error(`${entry}: corpus row must contain string _id/title/text`);
    }
    const docId = safeDocId(value._id);
    const path = `${docId}.md`;
    writeFileSync(
      join(corpusRoot, path),
      renderPublicCorpusMarkdown(docId, value.title, value.text),
      "utf8",
    );
    documents.push({ doc_id: docId, path });
  }
  await done;
  return { documents, sourceSha256: sourceHash.digest("hex") };
}

/** Hash a corpus entry without materializing it, for shared-corpus provenance. */
async function hashZipEntry(archive: string, entry: string): Promise<string> {
  const hash = createHash("sha256");
  const { lines, done } = unzipLines(archive, entry);
  for await (const line of lines) {
    if (!line.trim()) continue;
    hash.update(`${line}\n`, "utf8");
  }
  await done;
  return hash.digest("hex");
}

/** Normalized forms of every public-distill-v1 training query. */
function trainingQueries(): Set<string> {
  if (!existsSync(TRAINING_SFT)) {
    throw new Error(`Training release is missing, cannot check leakage: ${TRAINING_SFT}`);
  }
  const normalized = new Set<string>();
  for (const line of readFileSync(TRAINING_SFT, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const record = JSON.parse(line) as { query?: unknown };
    if (typeof record.query !== "string") throw new Error("training record has no query");
    normalized.add(normalizeQuery(record.query));
  }
  return normalized;
}

async function prepareEval(
  config: EvalConfig,
  staging: string,
  training: Set<string>,
): Promise<Record<string, unknown>> {
  const archive = join(ARCHIVES, config.archive);
  const output = join(staging, config.benchmarkId);
  mkdirSync(output, { recursive: true });

  const sourceQrels = await readZipText(archive, config.qrelsEntry);
  const qrels = parseQrels(sourceQrels, config.threshold);
  const queryById = await loadQueries(archive, config.queryEntry);

  let documents: Document[];
  let sourceCorpusSha256: string;
  let collectionRoot: string;
  if (config.reuseCorpusFrom) {
    const shared = config.reuseCorpusFrom;
    copyFileSync(join(shared, "documents.jsonl"), join(output, "documents.jsonl"));
    documents = readFileSync(join(output, "documents.jsonl"), "utf8")
      .split("\n")
      .filter(line => line.trim())
      .map(line => JSON.parse(line) as Document);
    sourceCorpusSha256 = await hashZipEntry(archive, config.corpusEntry);
    // bench.ts resolves collection_root against the benchmark directory.
    collectionRoot = relative(output, join(shared, "corpus"));
  } else {
    const corpus = await writeCorpus(archive, config.corpusEntry, output);
    documents = corpus.documents;
    sourceCorpusSha256 = corpus.sourceSha256;
    collectionRoot = "corpus";
    writeJsonl(join(output, "documents.jsonl"), documents);
  }

  const documentIds = new Set(documents.map(document => document.doc_id));
  const relevantByQid = new Map<string, string[]>();
  for (const qrel of qrels) {
    if (!documentIds.has(qrel.doc_id)) {
      throw new Error(`${config.benchmarkId}: missing qrels doc ${qrel.doc_id}`);
    }
    const values = relevantByQid.get(qrel.qid) ?? [];
    values.push(qrel.doc_id);
    relevantByQid.set(qrel.qid, values);
  }

  const exclusions: { qid: string; reason: string }[] = [];
  const queries: Query[] = [];
  for (const qid of [...relevantByQid.keys()].sort()) {
    const query = queryById.get(qid);
    if (query === undefined) throw new Error(`${config.benchmarkId}: missing query ${qid}`);
    if (!query.trim()) {
      exclusions.push({ qid, reason: "empty_query" });
      continue;
    }
    if (training.has(normalizeQuery(query))) {
      exclusions.push({ qid, reason: "training_query_collision" });
      continue;
    }
    queries.push({ qid, query });
  }
  if (queries.length === 0) throw new Error(`${config.benchmarkId}: no eligible query survived`);

  const admitted = new Set(queries.map(query => query.qid));
  const finalQrels = qrels.filter(qrel => admitted.has(qrel.qid));

  writeJsonl(join(output, "queries.jsonl"), queries);
  writeFileSync(
    join(output, "qrels.tsv"),
    `query-id\tcorpus-id\tscore\n${finalQrels.map(row => `${row.qid}\t${row.doc_id}\t1`).join("\n")}\n`,
    "utf8",
  );
  writeFileSync(join(output, "source-qrels.tsv"), sourceQrels, "utf8");
  writeFileSync(join(output, "exclusions.json"), stableJson(exclusions), "utf8");
  writeFileSync(
    join(output, "excluded-qids.json"),
    stableJson(exclusions.map(exclusion => exclusion.qid)),
    "utf8",
  );
  writeFileSync(join(output, "leakage-report.json"), stableJson({
    normalization: "NFKC-casefold-trim-collapse-whitespace",
    direction: "evaluation-vs-training",
    training_release: "public-distill-v1/public-main-v1",
    training_query_count: training.size,
    exclusions,
  }), "utf8");

  const queriesBytes = readFileSync(join(output, "queries.jsonl"));
  const qrelsBytes = readFileSync(join(output, "qrels.tsv"));
  const documentsBytes = readFileSync(join(output, "documents.jsonl"));
  const excludedQidsBytes = readFileSync(join(output, "excluded-qids.json"));
  const leakageBytes = readFileSync(join(output, "leakage-report.json"));
  const artifactHashes = {
    source_corpus_sha256: sourceCorpusSha256,
    queries_sha256: hashBytes("sha256", queriesBytes),
    qrels_sha256: hashBytes("sha256", qrelsBytes),
    documents_sha256: hashBytes("sha256", documentsBytes),
  };

  writeFileSync(join(output, "benchmark.yaml"), [
    `benchmark_id: ${config.benchmarkId}`,
    "source:",
    `  url: ${config.archiveUrl}`,
    `  archive_md5: ${config.archiveMd5}`,
    `  split: ${config.split}`,
    `source_qrels_sha256: ${hashBytes("sha256", sourceQrels)}`,
    `excluded_qids_sha256: ${hashBytes("sha256", excludedQidsBytes)}`,
    `leakage_report_sha256: ${hashBytes("sha256", leakageBytes)}`,
    `converted_data_sha256: ${hashBytes("sha256", JSON.stringify(artifactHashes))}`,
    "qrels:",
    `  relevant_threshold: ${config.threshold}`,
    "  unjudged: nonrelevant",
    "  graded: false",
    "cutoffs: [1, 3, 5, 10, 20, 30]",
    "metrics: [recall_at_cutoffs, mrr_at_10, ndcg_at_10]",
    "",
  ].join("\n"), "utf8");

  writeFileSync(join(output, "retrieval-profile.yaml"), [
    `profile_id: ${PROFILE_ID}`,
    `collection_name: ${config.benchmarkId}`,
    `collection_root: ${collectionRoot}`,
    `embedding_model: ${EMBEDDING_MODEL}`,
    `reranker_model: ${RERANKER_MODEL}`,
    "result_limit: 30",
    "per_list_limit: 30",
    "candidate_limit: 40",
    "rerank: true",
    "auto_expand: false",
    "strong_signal_bypass: false",
    "",
  ].join("\n"), "utf8");

  const manifest = {
    benchmark_id: config.benchmarkId,
    role: config.role,
    archive: {
      filename: config.archive,
      url: config.archiveUrl,
      official_md5: config.archiveMd5,
      actual_md5: await hashFile("md5", archive),
      sha256: await hashFile("sha256", archive),
    },
    source: { dataset: config.dataset, split: config.split, domain: config.domain },
    relevance_threshold: config.threshold,
    artifact_hashes: artifactHashes,
    collection_name: config.benchmarkId,
    collection_root: collectionRoot,
    shared_corpus_with: config.reuseCorpusFrom
      ? relative(resolve("."), config.reuseCorpusFrom)
      : null,
    counts: {
      corpus: documents.length,
      eligible_queries: queries.length,
      qrels: finalQrels.length,
      exclusions: exclusions.length,
    },
    license: {
      metadata: "Not bundled in the BEIR archives; consult the original dataset documentation.",
      beir_disclaimer: "BEIR redistributes third-party datasets; original dataset terms apply.",
    },
  };
  writeFileSync(join(output, "source-manifest.json"), stableJson(manifest), "utf8");
  return manifest;
}

async function main(): Promise<void> {
  if (existsSync(BENCHMARKS)) {
    throw new Error(`Evaluation benchmarks already exist, refusing to overwrite: ${BENCHMARKS}`);
  }
  if (!existsSync(SCIFACT_BENCHMARK)) {
    throw new Error(`SciFact benchmark is missing: ${SCIFACT_BENCHMARK}`);
  }
  for (const config of EVALS) {
    const path = join(ARCHIVES, config.archive);
    if (!existsSync(path)) throw new Error(`Missing archive: ${path}`);
    const actual = await hashFile("md5", path);
    if (actual !== config.archiveMd5) {
      throw new Error(`${basename(path)} MD5 mismatch: expected ${config.archiveMd5}, got ${actual}`);
    }
  }

  const training = trainingQueries();
  process.stderr.write(`Loaded ${training.size} normalized training queries\n`);

  const staging = `${BENCHMARKS}.staging`;
  rmSync(staging, { recursive: true, force: true });
  mkdirSync(staging, { recursive: true });
  try {
    const manifests = [];
    for (const config of EVALS) {
      process.stderr.write(`Preparing ${config.benchmarkId}...\n`);
      manifests.push(await prepareEval(config, staging, training));
    }
    writeFileSync(join(staging, "prepare-summary.json"), stableJson({
      profile_id: PROFILE_ID,
      training_release: "public-distill-v1/public-main-v1",
      training_query_count: training.size,
      benchmarks: manifests,
      external_benchmarks: [
        {
          benchmark_id: "qmd-expansion-scifact-v1",
          role: "formal",
          path: relative(resolve("."), SCIFACT_BENCHMARK),
          note: "Prepared and indexed previously; referenced, not rebuilt.",
        },
      ],
    }), "utf8");
    mkdirSync(EVAL_ROOT, { recursive: true });
    renameSync(staging, BENCHMARKS);
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }

  process.stderr.write(`\nPrepared ${EVALS.length} benchmarks in ${BENCHMARKS}\n`);
  for (const config of EVALS) {
    process.stderr.write(`  ${config.benchmarkId} (${config.role})\n`);
  }
}

main().catch(error => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exit(1);
});
