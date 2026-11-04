#!/usr/bin/env node

/**
 * Generic BEIR -> QMD benchmark converter.
 *
 * Every BEIR dataset ships the same three files (corpus.jsonl, queries.jsonl,
 * qrels/test.tsv). The only things that differ between datasets are captured in
 * a single BeirDatasetConfig; everything else is shared machinery. Per-dataset
 * wrappers (beir/scifact.ts, beir/fiqa.ts, ...) supply a config and a CLI
 * bootstrap, nothing more.
 *
 * The output qrels are ALWAYS binarized to the QMD bench contract
 * (relevant_threshold 1, graded false, unjudged nonrelevant): a source qrel with
 * score >= config.relevantThreshold becomes binary 1, anything below is dropped.
 * For datasets that are already binary (threshold 1) this is the identity and the
 * emitted bytes are unchanged.
 */

import {
  createHash,
  randomUUID,
} from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, join, posix, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { inflateRawSync } from "node:zlib";
import {
  parseDocumentsJsonl,
  parseQrelsTsv,
  parseQueriesJsonl,
  validateBenchmarkData,
} from "../../../src/bench/qrels.js";

export type BeirDatasetConfig = {
  benchmarkId: string;
  sourceUrl: string;
  archiveMd5: string;
  /** A source qrel with score >= this threshold is relevant (binary 1). */
  relevantThreshold: number;
};

/** The QMD bench contract only supports the test split, which every BEIR dataset stores here. */
const TEST_QRELS_SUFFIX = "/qrels/test.tsv";

type ZipEntry = {
  name: string;
  compression: number;
  compressedSize: number;
  uncompressedSize: number;
  localHeaderOffset: number;
  externalAttributes: number;
  directory: boolean;
};

type SourceCorpusRecord = {
  _id: string;
  title: string;
  text: string;
};

type SourceQueryRecord = {
  _id: string;
  text: string;
};

type TrainingQuery = {
  file: string;
  line: number;
  query: string;
  normalized: string;
  sha256: string;
};

type LeakageCandidate = {
  qid: string;
  benchmark_query: string;
  benchmark_query_sha256: string;
  training_file: string;
  training_line: number;
  training_query: string;
  training_query_sha256: string;
  token_jaccard: number;
  character_3gram_jaccard: number;
  human_confirmation: boolean | null;
};

type LeakageDecision = {
  qid: string;
  training_file: string;
  training_line: number;
  confirmed: boolean;
};

export type PrepareOptions = {
  archive: string;
  output: string;
  trainingFiles: string[];
  decisions?: string;
};

function hash(algorithm: "md5" | "sha256", bytes: Uint8Array | string): string {
  return createHash(algorithm).update(bytes).digest("hex");
}

export function verifyArchiveMd5(bytes: Uint8Array, expected: string): void {
  const actual = hash("md5", bytes);
  if (actual !== expected) {
    throw new Error(`archive MD5 mismatch: expected ${expected}, got ${actual}`);
  }
}

function findEndOfCentralDirectory(zip: Buffer): number {
  const minimum = Math.max(0, zip.length - 65_557);
  for (let offset = zip.length - 22; offset >= minimum; offset--) {
    if (zip.readUInt32LE(offset) === 0x06054b50) return offset;
  }
  throw new Error("Invalid ZIP: end-of-central-directory record not found");
}

function safeZipPath(name: string): string {
  if (
    name.length === 0
    || name.includes("\0")
    || name.includes("\\")
    || name.startsWith("/")
    || /^[A-Za-z]:/.test(name)
    || isAbsolute(name)
  ) {
    throw new Error(`Unsafe ZIP entry path: ${JSON.stringify(name)}`);
  }
  const components = name.split("/");
  if (components.some(component => component === "..")) {
    throw new Error(`Unsafe ZIP entry path traversal: ${JSON.stringify(name)}`);
  }
  const normalized = posix.normalize(name);
  if (normalized === ".." || normalized.startsWith("../")) {
    throw new Error(`Unsafe ZIP entry path traversal: ${JSON.stringify(name)}`);
  }
  return normalized;
}

function readZipEntries(zip: Buffer): ZipEntry[] {
  const eocd = findEndOfCentralDirectory(zip);
  const entryCount = zip.readUInt16LE(eocd + 10);
  const centralSize = zip.readUInt32LE(eocd + 12);
  const centralOffset = zip.readUInt32LE(eocd + 16);
  if (centralOffset + centralSize > eocd) {
    throw new Error("Invalid ZIP: central directory is out of bounds");
  }

  const entries: ZipEntry[] = [];
  let offset = centralOffset;
  for (let index = 0; index < entryCount; index++) {
    if (offset + 46 > zip.length || zip.readUInt32LE(offset) !== 0x02014b50) {
      throw new Error(`Invalid ZIP: central directory entry ${index + 1} is malformed`);
    }
    const flags = zip.readUInt16LE(offset + 8);
    if ((flags & 0x1) !== 0) throw new Error("Encrypted ZIP entries are not supported");
    const compression = zip.readUInt16LE(offset + 10);
    const compressedSize = zip.readUInt32LE(offset + 20);
    const uncompressedSize = zip.readUInt32LE(offset + 24);
    const nameLength = zip.readUInt16LE(offset + 28);
    const extraLength = zip.readUInt16LE(offset + 30);
    const commentLength = zip.readUInt16LE(offset + 32);
    const externalAttributes = zip.readUInt32LE(offset + 38);
    const localHeaderOffset = zip.readUInt32LE(offset + 42);
    const end = offset + 46 + nameLength + extraLength + commentLength;
    if (end > zip.length) throw new Error("Invalid ZIP: central directory entry is out of bounds");
    const rawName = zip.subarray(offset + 46, offset + 46 + nameLength).toString("utf8");
    const name = safeZipPath(rawName);
    const unixFileType = (externalAttributes >>> 16) & 0o170000;
    if (unixFileType === 0o120000) {
      throw new Error(`Unsafe ZIP symlink entry: ${JSON.stringify(name)}`);
    }
    const directory = rawName.endsWith("/") || unixFileType === 0o040000;
    entries.push({
      name,
      compression,
      compressedSize,
      uncompressedSize,
      localHeaderOffset,
      externalAttributes,
      directory,
    });
    offset = end;
  }
  return entries;
}

export function safeExtractZip(zipBytes: Uint8Array, destination: string): void {
  const zip = Buffer.from(zipBytes);
  const destinationRoot = resolve(destination);
  mkdirSync(destinationRoot, { recursive: true });

  for (const entry of readZipEntries(zip)) {
    const outputPath = resolve(destinationRoot, ...entry.name.split("/"));
    if (outputPath !== destinationRoot && !outputPath.startsWith(`${destinationRoot}/`)) {
      throw new Error(`Unsafe ZIP entry escaped destination: ${JSON.stringify(entry.name)}`);
    }
    if (entry.directory) {
      mkdirSync(outputPath, { recursive: true });
      continue;
    }
    const localOffset = entry.localHeaderOffset;
    if (
      localOffset + 30 > zip.length
      || zip.readUInt32LE(localOffset) !== 0x04034b50
    ) {
      throw new Error(`Invalid ZIP local header for ${JSON.stringify(entry.name)}`);
    }
    const localNameLength = zip.readUInt16LE(localOffset + 26);
    const localExtraLength = zip.readUInt16LE(localOffset + 28);
    const localName = zip
      .subarray(localOffset + 30, localOffset + 30 + localNameLength)
      .toString("utf8");
    if (safeZipPath(localName) !== entry.name) {
      throw new Error(`ZIP local/central filename mismatch for ${JSON.stringify(entry.name)}`);
    }
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const dataEnd = dataStart + entry.compressedSize;
    if (dataEnd > zip.length) {
      throw new Error(`Invalid ZIP compressed data for ${JSON.stringify(entry.name)}`);
    }
    const compressed = zip.subarray(dataStart, dataEnd);
    let content: Buffer;
    if (entry.compression === 0) {
      content = Buffer.from(compressed);
    } else if (entry.compression === 8) {
      content = inflateRawSync(compressed);
    } else {
      throw new Error(
        `Unsupported ZIP compression method ${entry.compression} for ${JSON.stringify(entry.name)}`,
      );
    }
    if (content.length !== entry.uncompressedSize) {
      throw new Error(`ZIP size mismatch for ${JSON.stringify(entry.name)}`);
    }
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, content);
  }
}

function jsonLines<T>(text: string, label: string, parse: (value: unknown, line: number) => T): T[] {
  const records: T[] = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (line.trim().length === 0) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      throw new Error(
        `${label}:${index + 1}: invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    records.push(parse(value, index + 1));
  }
  return records;
}

function sourceRecord(value: unknown, line: number, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label}:${line}: expected an object`);
  }
  return value as Record<string, unknown>;
}

function sourceString(value: unknown, location: string): string {
  if (typeof value !== "string") throw new Error(`${location}: expected a string`);
  return value;
}

function utf8Lf(value: string): string {
  return value.replace(/\r\n?/g, "\n");
}

/**
 * Leniently collect the qids referenced by a raw BEIR qrels file. Unlike
 * parseQrelsTsv (which enforces the binary 0/1 output contract), the SOURCE file
 * may carry graded scores (e.g. FiQA's 2/3), so it is only read for membership,
 * never validated. The binarized output is still checked by parseQrelsTsv later.
 */
function parseSourceQrelsQids(text: string): Set<string> {
  const lines = text.split(/\r?\n/);
  const qids = new Set<string>();
  for (let index = 1; index < lines.length; index++) {
    const line = lines[index]!;
    if (line.length === 0) continue;
    qids.add(line.split("\t")[0]!);
  }
  return qids;
}

export function renderCorpusMarkdown(title: string, text: string): string {
  return `# ${utf8Lf(title)}\n\n${utf8Lf(text)}\n`;
}

function byteCompare(left: string, right: string): number {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function findUniqueSourceFile(root: string, suffix: string): string {
  const matches: string[] = [];
  const visit = (directory: string): void => {
    for (const name of readdirSync(directory).sort(byteCompare)) {
      const path = join(directory, name);
      const stat = statSync(path);
      if (stat.isDirectory()) visit(path);
      else if (path.split("\\").join("/").endsWith(suffix)) matches.push(path);
    }
  };
  visit(root);
  if (matches.length !== 1) {
    throw new Error(`Expected exactly one ZIP entry ending in ${suffix}, found ${matches.length}`);
  }
  return matches[0]!;
}

function writeJsonl(path: string, values: unknown[]): void {
  writeFileSync(path, `${values.map(value => JSON.stringify(value)).join("\n")}\n`, "utf8");
}

function walkFiles(directory: string): string[] {
  const files: string[] = [];
  const visit = (current: string): void => {
    for (const name of readdirSync(current).sort(byteCompare)) {
      const path = join(current, name);
      const stat = statSync(path);
      if (stat.isDirectory()) visit(path);
      else if (stat.isFile()) files.push(path);
    }
  };
  visit(directory);
  return files;
}

export function computeConvertedDataSha256(benchmarkRoot: string): string {
  const root = resolve(benchmarkRoot);
  const corpusRoot = join(root, "corpus");
  const paths = [
    ...walkFiles(corpusRoot),
    join(root, "queries.jsonl"),
    join(root, "qrels.tsv"),
    join(root, "documents.jsonl"),
  ].map(path => relative(root, path).split("\\").join("/"))
    .sort(byteCompare);
  const aggregate = createHash("sha256");
  for (const relativePath of paths) {
    const fileHash = hash("sha256", readFileSync(join(root, ...relativePath.split("/"))));
    aggregate.update(`${relativePath}\0${fileHash}\n`, "utf8");
  }
  return aggregate.digest("hex");
}

export function normalizeLeakageQuery(query: string): string {
  return query
    .normalize("NFKC")
    .toLowerCase()
    .trim()
    .replace(/^\/only:(?:lex|vec|hyde)\s*/u, "")
    .replace(/\s+/gu, " ");
}

function terms(value: string): Set<string> {
  return new Set(value.split(/\s+/u).filter(Boolean));
}

function trigrams(value: string): Set<string> {
  const result = new Set<string>();
  const characters = Array.from(value);
  if (characters.length < 3) {
    if (value.length > 0) result.add(value);
    return result;
  }
  for (let index = 0; index <= characters.length - 3; index++) {
    result.add(characters.slice(index, index + 3).join(""));
  }
  return result;
}

function jaccard(left: Set<string>, right: Set<string>): number {
  if (left.size === 0 && right.size === 0) return 1;
  let intersection = 0;
  for (const value of left) {
    if (right.has(value)) intersection++;
  }
  return intersection / (left.size + right.size - intersection);
}

function loadTrainingQueries(files: string[]): {
  files: { path: string; sha256: string; query_count: number }[];
  queries: TrainingQuery[];
} {
  const resultFiles: { path: string; sha256: string; query_count: number }[] = [];
  const queries: TrainingQuery[] = [];
  for (const inputPath of [...files].sort(byteCompare)) {
    const bytes = readFileSync(inputPath);
    let count = 0;
    for (const [index, line] of bytes.toString("utf8").split(/\r?\n/).entries()) {
      if (line.trim().length === 0) continue;
      const value = sourceRecord(JSON.parse(line), index + 1, inputPath);
      const queryValue = value.query ?? value.seed_query;
      if (typeof queryValue !== "string" || queryValue.length === 0) {
        throw new Error(`${inputPath}:${index + 1}: missing query or seed_query`);
      }
      const normalized = normalizeLeakageQuery(queryValue);
      queries.push({
        file: inputPath.split("\\").join("/"),
        line: index + 1,
        query: queryValue,
        normalized,
        sha256: hash("sha256", Buffer.from(queryValue, "utf8")),
      });
      count++;
    }
    resultFiles.push({
      path: inputPath.split("\\").join("/"),
      sha256: hash("sha256", bytes),
      query_count: count,
    });
  }
  return { files: resultFiles, queries };
}

function decisionKey(
  value: Pick<LeakageCandidate, "qid" | "training_file" | "training_line">,
): string {
  return `${value.qid}\0${value.training_file}\0${value.training_line}`;
}

function loadDecisions(path: string | undefined): Map<string, boolean> {
  if (!path) return new Map();
  const values = JSON.parse(readFileSync(path, "utf8"));
  if (!Array.isArray(values)) throw new Error("Leakage decisions must be a JSON array");
  const decisions = new Map<string, boolean>();
  for (const value of values as LeakageDecision[]) {
    if (
      typeof value?.qid !== "string"
      || typeof value.training_file !== "string"
      || !Number.isSafeInteger(value.training_line)
      || typeof value.confirmed !== "boolean"
    ) {
      throw new Error("Invalid leakage decision record");
    }
    const key = decisionKey(value);
    if (decisions.has(key)) throw new Error(`Duplicate leakage decision for ${key}`);
    decisions.set(key, value.confirmed);
  }
  return decisions;
}

function buildLeakageReport(
  config: BeirDatasetConfig,
  queries: SourceQueryRecord[],
  trainingFiles: string[],
  decisionsPath?: string,
): { report: Record<string, unknown>; excludedQids: string[]; pending: number } {
  const training = loadTrainingQueries(trainingFiles);
  const decisions = loadDecisions(decisionsPath);
  const exact_matches: LeakageCandidate[] = [];
  const near_duplicate_candidates: LeakageCandidate[] = [];
  const excluded = new Set<string>();
  const usedDecisions = new Set<string>();

  for (const benchmark of queries) {
    const normalized = normalizeLeakageQuery(benchmark.text);
    const benchmarkHash = hash("sha256", Buffer.from(benchmark.text, "utf8"));
    const benchmarkTerms = terms(normalized);
    const benchmarkTrigrams = trigrams(normalized);
    for (const candidate of training.queries) {
      const tokenScore = jaccard(benchmarkTerms, terms(candidate.normalized));
      const trigramScore = jaccard(benchmarkTrigrams, trigrams(candidate.normalized));
      const base = {
        qid: benchmark._id,
        benchmark_query: benchmark.text,
        benchmark_query_sha256: benchmarkHash,
        training_file: candidate.file,
        training_line: candidate.line,
        training_query: candidate.query,
        training_query_sha256: candidate.sha256,
        token_jaccard: tokenScore,
        character_3gram_jaccard: trigramScore,
      };
      if (normalized === candidate.normalized) {
        exact_matches.push({ ...base, human_confirmation: true });
        excluded.add(benchmark._id);
      } else if (tokenScore >= 0.8 || trigramScore >= 0.85) {
        const key = decisionKey(base);
        const confirmation = decisions.get(key) ?? null;
        if (decisions.has(key)) usedDecisions.add(key);
        near_duplicate_candidates.push({ ...base, human_confirmation: confirmation });
        if (confirmation === true) excluded.add(benchmark._id);
      }
    }
  }
  const unused = [...decisions.keys()].filter(key => !usedDecisions.has(key));
  if (unused.length > 0) {
    throw new Error(`Leakage decisions do not match current candidates: ${unused.join(", ")}`);
  }
  const pending = near_duplicate_candidates.filter(item => item.human_confirmation === null).length;
  const benchmark_query_hashes = queries
    .map(query => ({
      qid: query._id,
      sha256: hash("sha256", Buffer.from(query.text, "utf8")),
    }))
    .sort((left, right) => byteCompare(left.qid, right.qid));
  const report = {
    benchmark_id: config.benchmarkId,
    benchmark_query_hashes,
    training_data: training.files,
    normalization: [
      "Unicode NFKC",
      "lowercase",
      "trim",
      "collapse consecutive whitespace",
      "remove /only:lex, /only:vec, or /only:hyde prefix",
    ],
    thresholds: {
      token_jaccard: 0.8,
      character_3gram_jaccard: 0.85,
    },
    exact_matches,
    near_duplicate_candidates,
    confirmed_leakage_qids: [...excluded].sort(byteCompare),
    review_status: pending === 0 ? "complete" : "pending",
  };
  return { report, excludedQids: [...excluded].sort(byteCompare), pending };
}

/**
 * Map one raw qrels data line to its binarized form, or null to drop it.
 * Returns null for empty lines, sub-threshold (nonrelevant) rows, and excluded
 * qids; otherwise emits `qid\tdocid\t1`. For an already-binary dataset
 * (threshold 1, all scores 1) this reproduces the input line verbatim.
 */
function binarizeQrelsLine(line: string, threshold: number, excluded: ReadonlySet<string>): string | null {
  if (line.length === 0) return null;
  const fields = line.split("\t");
  const qid = fields[0]!;
  const docid = fields[1];
  const score = Number(fields[2]);
  if (excluded.has(qid)) return null;
  if (!(score >= threshold)) return null;
  return `${qid}\t${docid}\t1`;
}

function benchmarkYaml(config: BeirDatasetConfig, hashes: {
  sourceQrels: string;
  excludedQids: string;
  leakageReport: string;
  convertedData: string;
}): string {
  return `benchmark_id: ${config.benchmarkId}
source:
  url: ${config.sourceUrl}
  archive_md5: ${config.archiveMd5}
  split: test
source_qrels_sha256: ${hashes.sourceQrels}
excluded_qids_sha256: ${hashes.excludedQids}
leakage_report_sha256: ${hashes.leakageReport}
converted_data_sha256: ${hashes.convertedData}
qrels:
  relevant_threshold: 1
  unjudged: nonrelevant
  graded: false
cutoffs: [1, 3, 5, 10, 20, 30]
metrics: [recall_at_cutoffs, mrr_at_10, ndcg_at_10]
`;
}

function convertArchive(
  config: BeirDatasetConfig,
  options: PrepareOptions,
  outputRoot: string,
): { pending: number; hash: string } {
  const archiveBytes = readFileSync(options.archive);
  verifyArchiveMd5(archiveBytes, config.archiveMd5);
  const extracted = join(dirname(outputRoot), `.beir-extracted-${randomUUID()}`);
  mkdirSync(extracted, { recursive: true });
  try {
    safeExtractZip(archiveBytes, extracted);
    const corpusPath = findUniqueSourceFile(extracted, "/corpus.jsonl");
    const queriesPath = findUniqueSourceFile(extracted, "/queries.jsonl");
    const sourceQrelsPath = findUniqueSourceFile(extracted, TEST_QRELS_SUFFIX);
    const corpus = jsonLines<SourceCorpusRecord>(
      readFileSync(corpusPath, "utf8"),
      "corpus.jsonl",
      (value, line) => {
        const item = sourceRecord(value, line, "corpus.jsonl");
        return {
          _id: sourceString(item._id, `corpus.jsonl:${line}._id`),
          title: sourceString(item.title, `corpus.jsonl:${line}.title`),
          text: sourceString(item.text, `corpus.jsonl:${line}.text`),
        };
      },
    );
    const allQueries = jsonLines<SourceQueryRecord>(
      readFileSync(queriesPath, "utf8"),
      "queries.jsonl",
      (value, line) => {
        const item = sourceRecord(value, line, "queries.jsonl");
        return {
          _id: sourceString(item._id, `queries.jsonl:${line}._id`),
          text: sourceString(item.text, `queries.jsonl:${line}.text`),
        };
      },
    );
    const sourceQrelsBytes = readFileSync(sourceQrelsPath);
    const qids = parseSourceQrelsQids(sourceQrelsBytes.toString("utf8"));
    const benchmarkQueries = allQueries
      .filter(query => qids.has(query._id))
      .sort((left, right) => byteCompare(left._id, right._id));
    if (benchmarkQueries.length !== qids.size) {
      throw new Error(
        `Source qrels reference ${qids.size} qids but only ${benchmarkQueries.length} exist in queries.jsonl`,
      );
    }
    const leakage = buildLeakageReport(
      config,
      benchmarkQueries,
      options.trainingFiles,
      options.decisions,
    );
    const excluded = new Set(leakage.excludedQids);
    const finalQueries = benchmarkQueries.filter(query => !excluded.has(query._id));
    const qrelsLines = sourceQrelsBytes.toString("utf8").split(/\r?\n/);
    const finalQrelsText = `${qrelsLines
      .map((line, index) => (index === 0 ? line : binarizeQrelsLine(line, config.relevantThreshold, excluded)))
      .filter((line): line is string => line !== null && line.length > 0)
      .join("\n")}\n`;

    const corpusIds = new Set<string>();
    mkdirSync(join(outputRoot, "corpus"), { recursive: true });
    const documents = corpus
      .sort((left, right) => byteCompare(left._id, right._id))
      .map(document => {
        if (corpusIds.has(document._id)) {
          throw new Error(`corpus.jsonl: duplicate _id "${document._id}"`);
        }
        if (
          document._id.length === 0
          || document._id.includes("/")
          || document._id.includes("\\")
          || document._id === "."
          || document._id === ".."
        ) {
          throw new Error(`corpus.jsonl: unsafe _id ${JSON.stringify(document._id)}`);
        }
        corpusIds.add(document._id);
        const relativePath = `${document._id}.md`;
        writeFileSync(
          join(outputRoot, "corpus", relativePath),
          renderCorpusMarkdown(document.title, document.text),
          "utf8",
        );
        return { doc_id: document._id, path: relativePath };
      });

    const queryRows = finalQueries.map(query => ({ qid: query._id, query: query.text }));
    writeJsonl(join(outputRoot, "queries.jsonl"), queryRows);
    writeFileSync(join(outputRoot, "source-qrels.tsv"), sourceQrelsBytes);
    writeFileSync(join(outputRoot, "qrels.tsv"), finalQrelsText, "utf8");
    writeJsonl(join(outputRoot, "documents.jsonl"), documents);
    writeFileSync(
      join(outputRoot, "excluded-qids.json"),
      `${JSON.stringify(leakage.excludedQids, null, 2)}\n`,
      "utf8",
    );
    writeFileSync(
      join(outputRoot, "leakage-report.json"),
      `${JSON.stringify(leakage.report, null, 2)}\n`,
      "utf8",
    );

    const parsedQueries = parseQueriesJsonl(readFileSync(join(outputRoot, "queries.jsonl"), "utf8"));
    const parsedDocuments = parseDocumentsJsonl(
      readFileSync(join(outputRoot, "documents.jsonl"), "utf8"),
    );
    const parsedQrels = parseQrelsTsv(readFileSync(join(outputRoot, "qrels.tsv"), "utf8"));
    validateBenchmarkData(parsedQueries, parsedQrels, parsedDocuments);

    const convertedData = computeConvertedDataSha256(outputRoot);
    const hashes = {
      sourceQrels: hash("sha256", sourceQrelsBytes),
      excludedQids: hash("sha256", readFileSync(join(outputRoot, "excluded-qids.json"))),
      leakageReport: hash("sha256", readFileSync(join(outputRoot, "leakage-report.json"))),
      convertedData,
    };
    writeFileSync(join(outputRoot, "benchmark.yaml"), benchmarkYaml(config, hashes), "utf8");
    return { pending: leakage.pending, hash: convertedData };
  } finally {
    rmSync(extracted, { recursive: true, force: true });
  }
}

export function prepareBeir(
  config: BeirDatasetConfig,
  options: PrepareOptions,
): { pending: number; hash: string } {
  const output = resolve(options.output);
  if (existsSync(output)) throw new Error(`Output already exists: ${output}`);
  const staging = `${output}.staging-${randomUUID()}`;
  mkdirSync(staging, { recursive: true });
  try {
    const result = convertArchive(config, options, staging);
    mkdirSync(dirname(output), { recursive: true });
    renameSync(staging, output);
    return result;
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }
}

export function verifyPreparedBeir(output: string): { hash: string; pending: number } {
  const root = resolve(output);
  const manifestText = readFileSync(join(root, "benchmark.yaml"), "utf8");
  const convertedMatch = manifestText.match(/^converted_data_sha256:\s*([a-f0-9]{64})$/m);
  if (!convertedMatch) throw new Error("benchmark.yaml: missing converted_data_sha256");
  const actual = computeConvertedDataSha256(root);
  if (actual !== convertedMatch[1]) {
    throw new Error(
      `converted_data_sha256 mismatch: expected ${convertedMatch[1]}, got ${actual}`,
    );
  }
  const report = JSON.parse(readFileSync(join(root, "leakage-report.json"), "utf8"));
  const pending = Array.isArray(report.near_duplicate_candidates)
    ? report.near_duplicate_candidates.filter(
      (candidate: { human_confirmation?: unknown }) => candidate.human_confirmation === null,
    ).length
    : 0;
  if (pending > 0) {
    throw new Error(`Leakage review is incomplete: ${pending} near-duplicate candidates pending`);
  }
  const queries = parseQueriesJsonl(readFileSync(join(root, "queries.jsonl"), "utf8"));
  const documents = parseDocumentsJsonl(readFileSync(join(root, "documents.jsonl"), "utf8"));
  const qrels = parseQrelsTsv(readFileSync(join(root, "qrels.tsv"), "utf8"));
  validateBenchmarkData(queries, qrels, documents);
  return { hash: actual, pending };
}

function usage(): never {
  throw new Error(
    "Usage:\n"
    + "  beir/<dataset>.ts prepare --archive <dataset.zip> --output <benchmark-dir> "
    + "--training <data.jsonl> [--training <data.jsonl> ...] [--decisions <decisions.json>]\n"
    + "  beir/<dataset>.ts verify --output <benchmark-dir>",
  );
}

function parseArgs(argv: string[]): { mode: "prepare" | "verify"; options: PrepareOptions } {
  const mode = argv.shift();
  if (mode !== "prepare" && mode !== "verify") usage();
  let archive = "";
  let output = "";
  let decisions: string | undefined;
  const trainingFiles: string[] = [];
  while (argv.length > 0) {
    const flag = argv.shift();
    const value = argv.shift();
    if (!value) usage();
    if (flag === "--archive") archive = value;
    else if (flag === "--output") output = value;
    else if (flag === "--training") trainingFiles.push(value);
    else if (flag === "--decisions") decisions = value;
    else usage();
  }
  if (!output || (mode === "prepare" && (!archive || trainingFiles.length === 0))) usage();
  return { mode, options: { archive, output, trainingFiles, decisions } };
}

/** Shared CLI driver. Each per-dataset wrapper calls this behind its own entrypoint guard. */
export function runBeirCli(config: BeirDatasetConfig, argv: string[]): void {
  try {
    const { mode, options } = parseArgs(argv);
    const result = mode === "prepare"
      ? prepareBeir(config, options)
      : verifyPreparedBeir(options.output);
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    if (result.pending > 0) process.exitCode = 2;
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

/** Returns true when the given module URL is the invoked script (per-wrapper entrypoint guard). */
export function isMainModule(moduleUrl: string): boolean {
  return process.argv[1] !== undefined
    && resolve(process.argv[1]) === resolve(fileURLToPath(moduleUrl));
}
