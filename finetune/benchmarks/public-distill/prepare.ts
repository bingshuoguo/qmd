#!/usr/bin/env node

import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import {
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { once } from "node:events";
import { renderPublicCorpusMarkdown } from "../lib/public-corpus.js";

const ROOT = resolve("finetune/data/public-distill-v0");
const ARCHIVES = join(ROOT, "archives");
const PREPARED = join(ROOT, "prepared");
const SEED_PREFIX = "qmd-public-v0\0seed=42\0";
const EMBEDDING_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const RERANKER_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

type Query = { qid: string; query: string };
type Qrel = { qid: string; doc_id: string; relevance: 1 };
type SourceConfig = {
  sourceId: string;
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
  specThreshold: number;
  quota: number;
};

const SOURCES: SourceConfig[] = [
  {
    sourceId: "fiqa-train",
    archive: "fiqa.zip",
    archiveUrl: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    archiveMd5: "17918ed23cd04fb15047f73e6c3bd9d9",
    dataset: "fiqa",
    split: "train",
    domain: null,
    queryEntry: "fiqa/queries.jsonl",
    corpusEntry: "fiqa/corpus.jsonl",
    qrelsEntry: "fiqa/qrels/train.tsv",
    threshold: 1,
    specThreshold: 2,
    quota: 750,
  },
  {
    sourceId: "cqadup-programmers",
    archive: "cqadupstack.zip",
    archiveUrl: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
    archiveMd5: "4e41456d7df8ee7760a7f866133bda78",
    dataset: "cqadupstack",
    split: "test",
    domain: "programmers",
    queryEntry: "cqadupstack/programmers/queries.jsonl",
    corpusEntry: "cqadupstack/programmers/corpus.jsonl",
    qrelsEntry: "cqadupstack/programmers/qrels/test.tsv",
    threshold: 1,
    specThreshold: 1,
    quota: 875,
  },
  {
    sourceId: "cqadup-unix",
    archive: "cqadupstack.zip",
    archiveUrl: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
    archiveMd5: "4e41456d7df8ee7760a7f866133bda78",
    dataset: "cqadupstack",
    split: "test",
    domain: "unix",
    queryEntry: "cqadupstack/unix/queries.jsonl",
    corpusEntry: "cqadupstack/unix/corpus.jsonl",
    qrelsEntry: "cqadupstack/unix/qrels/test.tsv",
    threshold: 1,
    specThreshold: 1,
    quota: 875,
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

function sampleKey(sourceId: string, qid: string): string {
  return hashBytes("sha256", `${SEED_PREFIX}${sourceId}\0${qid}`);
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

async function writeCorpus(
  archive: string,
  entry: string,
  outputRoot: string,
): Promise<{ documents: { doc_id: string; path: string }[]; sourceSha256: string }> {
  const corpusRoot = join(outputRoot, "corpus");
  mkdirSync(corpusRoot, { recursive: true });
  const documents: { doc_id: string; path: string }[] = [];
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
    const markdown = renderPublicCorpusMarkdown(docId, value.title, value.text);
    writeFileSync(join(corpusRoot, path), markdown, "utf8");
    documents.push({ doc_id: docId, path });
  }
  await done;
  return { documents, sourceSha256: sourceHash.digest("hex") };
}

async function holdoutQueries(): Promise<Set<string>> {
  const fiqaArchive = join(ARCHIVES, "fiqa.zip");
  const cqaArchive = join(ARCHIVES, "cqadupstack.zip");
  const fiqaAll = await loadQueries(fiqaArchive, "fiqa/queries.jsonl");
  const fiqaQids = new Set<string>();
  for (const entry of ["fiqa/qrels/dev.tsv", "fiqa/qrels/test.tsv"]) {
    for (const qrel of parseQrels(await readZipText(fiqaArchive, entry), 1)) fiqaQids.add(qrel.qid);
  }
  const normalized = new Set<string>();
  for (const qid of fiqaQids) {
    const query = fiqaAll.get(qid);
    if (!query) throw new Error(`FiQA holdout qid missing from queries: ${qid}`);
    normalized.add(normalizeQuery(query));
  }
  for (const domain of ["android", "webmasters"]) {
    const queries = await loadQueries(cqaArchive, `cqadupstack/${domain}/queries.jsonl`);
    for (const query of queries.values()) normalized.add(normalizeQuery(query));
  }
  return normalized;
}

async function prepareSource(config: SourceConfig, staging: string, holdouts: Set<string>) {
  const archive = join(ARCHIVES, config.archive);
  const output = join(staging, config.sourceId);
  mkdirSync(output, { recursive: true });
  const sourceQrels = await readZipText(archive, config.qrelsEntry);
  const qrels = parseQrels(sourceQrels, config.threshold);
  const queryById = await loadQueries(archive, config.queryEntry);
  const corpus = await writeCorpus(archive, config.corpusEntry, output);
  const documentIds = new Set(corpus.documents.map(document => document.doc_id));
  const relevantByQid = new Map<string, string[]>();
  for (const qrel of qrels) {
    if (!documentIds.has(qrel.doc_id)) throw new Error(`${config.sourceId}: missing qrels doc ${qrel.doc_id}`);
    const values = relevantByQid.get(qrel.qid) ?? [];
    values.push(qrel.doc_id);
    relevantByQid.set(qrel.qid, values);
  }
  const exclusions: { qid: string; reason: string }[] = [];
  const queries: Query[] = [];
  for (const qid of [...relevantByQid.keys()].sort()) {
    const query = queryById.get(qid);
    if (query === undefined) throw new Error(`${config.sourceId}: missing query ${qid}`);
    if (!query.trim()) {
      exclusions.push({ qid, reason: "empty_query" });
      continue;
    }
    if (holdouts.has(normalizeQuery(query))) {
      exclusions.push({ qid, reason: "exact_holdout_collision" });
      continue;
    }
    queries.push({ qid, query });
  }
  const admitted = new Set(queries.map(query => query.qid));
  const finalQrels = qrels.filter(qrel => admitted.has(qrel.qid));
  writeJsonl(join(output, "queries.jsonl"), queries);
  writeJsonl(join(output, "documents.jsonl"), corpus.documents);
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
    holdouts: ["fiqa-dev", "fiqa-test", "cqadup-android", "cqadup-webmasters"],
    exclusions,
  }), "utf8");
  const collectionName = `qmd-distill-public-v0-${config.sourceId}`;
  const queriesBytes = readFileSync(join(output, "queries.jsonl"));
  const qrelsBytes = readFileSync(join(output, "qrels.tsv"));
  const documentsBytes = readFileSync(join(output, "documents.jsonl"));
  const excludedQidsBytes = readFileSync(join(output, "excluded-qids.json"));
  const leakageBytes = readFileSync(join(output, "leakage-report.json"));
  const artifactHashes = {
    source_corpus_sha256: corpus.sourceSha256,
    queries_sha256: hashBytes("sha256", queriesBytes),
    qrels_sha256: hashBytes("sha256", qrelsBytes),
    documents_sha256: hashBytes("sha256", documentsBytes),
  };
  writeFileSync(join(output, "benchmark.yaml"), [
    `benchmark_id: ${config.sourceId}`,
    "source:",
    `  url: ${config.archiveUrl}`,
    `  archive_md5: ${config.archiveMd5}`,
    `  split: ${config.split}`,
    `source_qrels_sha256: ${hashBytes("sha256", sourceQrels)}`,
    `excluded_qids_sha256: ${hashBytes("sha256", excludedQidsBytes)}`,
    `leakage_report_sha256: ${hashBytes("sha256", leakageBytes)}`,
    `converted_data_sha256: ${hashBytes("sha256", JSON.stringify(artifactHashes))}`,
    "qrels:",
    "  relevant_threshold: 1",
    "  unjudged: nonrelevant",
    "  graded: false",
    "cutoffs: [1, 3, 5, 10, 20, 30]",
    "metrics: [recall_at_cutoffs, mrr_at_10, ndcg_at_10]",
    "",
  ].join("\n"), "utf8");
  writeFileSync(join(output, "retrieval-profile.yaml"), [
    "profile_id: qmd-public-distill-v0",
    `collection_name: ${collectionName}`,
    "collection_root: corpus",
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
    source_id: config.sourceId,
    archive: {
      filename: config.archive,
      url: config.archiveUrl,
      official_md5: config.archiveMd5,
      actual_md5: await hashFile("md5", archive),
      sha256: await hashFile("sha256", archive),
    },
    source: { dataset: config.dataset, split: config.split, domain: config.domain },
    relevance_threshold: config.threshold,
    spec_relevance_threshold: config.specThreshold,
    spec_deviation: config.threshold === config.specThreshold
      ? null
      : "Official FiQA train qrels contain only relevance=1; smoke uses native positive relevance > 0.",
    artifact_hashes: artifactHashes,
    collection_name: collectionName,
    counts: {
      corpus: corpus.documents.length,
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
  return { config, queries, manifest };
}

async function main(): Promise<void> {
  if (existsSync(PREPARED)) throw new Error(`Prepared output already exists: ${PREPARED}`);
  for (const config of SOURCES) {
    const path = join(ARCHIVES, config.archive);
    if (!existsSync(path)) throw new Error(`Missing archive: ${path}`);
    const actual = await hashFile("md5", path);
    if (actual !== config.archiveMd5) {
      throw new Error(`${basename(path)} MD5 mismatch: expected ${config.archiveMd5}, got ${actual}`);
    }
  }
  const staging = `${PREPARED}.staging`;
  rmSync(staging, { recursive: true, force: true });
  mkdirSync(staging, { recursive: true });
  try {
    const holdouts = await holdoutQueries();
    const prepared = [];
    for (const source of SOURCES) {
      process.stderr.write(`Preparing ${source.sourceId}...\n`);
      prepared.push(await prepareSource(source, staging, holdouts));
    }
    const all = prepared.flatMap(source => source.queries.map(query => ({
      input_id: `${source.config.sourceId}:${query.qid}`,
      source_id: source.config.sourceId,
      qid: query.qid,
      query: query.query,
      sample_key: sampleKey(source.config.sourceId, query.qid),
    })));
    const ownerByQuery = new Map<string, typeof all[number]>();
    for (const input of all) {
      const key = normalizeQuery(input.query);
      const current = ownerByQuery.get(key);
      if (!current || input.sample_key < current.sample_key) ownerByQuery.set(key, input);
    }
    const deduplicated = new Set([...ownerByQuery.values()].map(input => input.input_id));
    const smoke = prepared.flatMap(source => all
      .filter(input => input.source_id === source.config.sourceId && deduplicated.has(input.input_id))
      .sort((left, right) => left.sample_key.localeCompare(right.sample_key) || left.qid.localeCompare(right.qid))
      .slice(0, 10));
    if (smoke.length !== 30) throw new Error(`Expected 30 smoke inputs, got ${smoke.length}`);
    writeJsonl(join(staging, "pool-smoke.jsonl"), smoke);
    const sourceCounts = Object.fromEntries(SOURCES.map(source => [
      source.sourceId,
      smoke.filter(input => input.source_id === source.sourceId).length,
    ]));
    const mainCapacity = Object.fromEntries(prepared.map(source => {
      const available = all.filter(input => (
        input.source_id === source.config.sourceId
        && deduplicated.has(input.input_id)
        && !smoke.some(smokeInput => smokeInput.input_id === input.input_id)
      )).length;
      return [source.config.sourceId, {
        requested: source.config.quota,
        available_after_smoke: available,
        sufficient: available >= source.config.quota,
      }];
    }));
    const poolBytes = readFileSync(join(staging, "pool-smoke.jsonl"));
    writeFileSync(join(staging, "pool-manifest.json"), stableJson({
      version: "qmd-public-pool-v0",
      experiment_seed: 42,
      sampling: "sha256(qmd-public-v0\\0seed=42\\0 + source_id + \\0 + qid)",
      normalization: "NFKC-casefold-trim-collapse-whitespace",
      smoke_count: smoke.length,
      smoke_source_counts: sourceCounts,
      smoke_sha256: hashBytes("sha256", poolBytes),
      holdout_normalized_query_count: holdouts.size,
      cross_source_duplicate_count: all.length - deduplicated.size,
      main_pool_status: "not_materialized_by_smoke_run",
      main_capacity: mainCapacity,
      smoke_only: true,
    }), "utf8");
    renameSync(staging, PREPARED);
    process.stdout.write(stableJson({ prepared: PREPARED, smoke_count: smoke.length, source_counts: sourceCounts }));
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }
}

main().catch(error => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
