#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

const ROOT = resolve("finetune/data/public-distill-v0");
const PREPARED = join(ROOT, "prepared");
const ARCHIVES = join(ROOT, "archives");
const POOL_PATH = join(PREPARED, "pool-main.jsonl");
const MANIFEST_PATH = join(PREPARED, "pool-main-manifest.json");
const SEED_PREFIX = "qmd-public-v0\0seed=42\0";

const SOURCES = [
  { sourceId: "fiqa-train", quota: 750 },
  { sourceId: "cqadup-programmers", quota: 866 },
  { sourceId: "cqadup-unix", quota: 884 },
] as const;

type Input = {
  input_id: string;
  source_id: string;
  qid: string;
  query: string;
  sample_key: string;
};

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function normalizeQuery(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase("und").trim().replace(/\s+/gu, " ");
}

function sampleKey(sourceId: string, qid: string): string {
  return sha256(`${SEED_PREFIX}${sourceId}\0${qid}`);
}

function readJsonl(path: string): any[] {
  return readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
}

function readZipJsonl(archive: string, entry: string): any[] {
  const text = execFileSync("unzip", ["-p", archive, entry], { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
  return text.trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
}

function readZipQrelQids(archive: string, entry: string): Set<string> {
  const text = execFileSync("unzip", ["-p", archive, entry], { encoding: "utf8" });
  return new Set(text.trim().split(/\r?\n/u).slice(1).map(line => line.split("\t")[0]!).filter(Boolean));
}

function loadHoldouts(): Set<string> {
  const fiqaArchive = join(ARCHIVES, "fiqa.zip");
  const cqaArchive = join(ARCHIVES, "cqadupstack.zip");
  const fiqaQueries = new Map(readZipJsonl(fiqaArchive, "fiqa/queries.jsonl").map(row => [row._id, row.text]));
  const holdouts = new Set<string>();
  for (const entry of ["fiqa/qrels/dev.tsv", "fiqa/qrels/test.tsv"]) {
    for (const qid of readZipQrelQids(fiqaArchive, entry)) {
      const query = fiqaQueries.get(qid);
      if (typeof query !== "string") throw new Error(`FiQA holdout qid missing from queries: ${qid}`);
      holdouts.add(normalizeQuery(query));
    }
  }
  for (const domain of ["android", "webmasters"]) {
    for (const row of readZipJsonl(cqaArchive, `cqadupstack/${domain}/queries.jsonl`)) {
      if (typeof row.text !== "string") throw new Error(`${domain}: holdout query text must be a string`);
      holdouts.add(normalizeQuery(row.text));
    }
  }
  return holdouts;
}

function buildArtifacts(): { poolBytes: string; manifestBytes: string; summary: unknown } {
  const smokePath = join(PREPARED, "pool-smoke.jsonl");
  const smokeBytes = readFileSync(smokePath);
  const smoke = readJsonl(smokePath) as Input[];
  if (smoke.length !== 30) throw new Error(`Expected 30 frozen smoke inputs, got ${smoke.length}`);
  const smokeIds = new Set(smoke.map(input => input.input_id));
  if (smokeIds.size !== smoke.length) throw new Error("Frozen smoke pool contains duplicate input IDs");

  const sourceArtifacts: Record<string, unknown> = {};
  const all: Input[] = [];
  for (const source of SOURCES) {
    const queriesPath = join(PREPARED, source.sourceId, "queries.jsonl");
    const sourceManifestPath = join(PREPARED, source.sourceId, "source-manifest.json");
    for (const row of readJsonl(queriesPath)) {
      if (typeof row.qid !== "string" || typeof row.query !== "string" || !row.query.trim()) {
        throw new Error(`${source.sourceId}: invalid prepared query`);
      }
      all.push({
        input_id: `${source.sourceId}:${row.qid}`,
        source_id: source.sourceId,
        qid: row.qid,
        query: row.query,
        sample_key: sampleKey(source.sourceId, row.qid),
      });
    }
    sourceArtifacts[source.sourceId] = {
      queries_sha256: sha256(readFileSync(queriesPath)),
      source_manifest_sha256: sha256(readFileSync(sourceManifestPath)),
    };
  }

  const ownerByQuery = new Map<string, Input>();
  for (const input of all) {
    const normalized = normalizeQuery(input.query);
    const current = ownerByQuery.get(normalized);
    if (!current || input.sample_key < current.sample_key) ownerByQuery.set(normalized, input);
  }
  const deduplicatedIds = new Set([...ownerByQuery.values()].map(input => input.input_id));
  const main = SOURCES.flatMap(source => {
    const eligible = all
      .filter(input => input.source_id === source.sourceId && deduplicatedIds.has(input.input_id) && !smokeIds.has(input.input_id))
      .sort((left, right) => left.sample_key.localeCompare(right.sample_key) || left.qid.localeCompare(right.qid));
    if (eligible.length < source.quota) {
      throw new Error(`${source.sourceId}: needs ${source.quota} inputs, only ${eligible.length} available`);
    }
    return eligible.slice(0, source.quota);
  });
  if (main.length !== 2500) throw new Error(`Expected 2500 main inputs, got ${main.length}`);

  const mainIds = new Set(main.map(input => input.input_id));
  const normalizedMain = new Set(main.map(input => normalizeQuery(input.query)));
  const normalizedSmoke = new Set(smoke.map(input => normalizeQuery(input.query)));
  if (mainIds.size !== main.length || normalizedMain.size !== main.length) {
    throw new Error("Main pool contains duplicate input IDs or normalized queries");
  }
  if (main.some(input => smokeIds.has(input.input_id))) throw new Error("Main pool overlaps frozen smoke pool");
  if ([...normalizedMain].some(query => normalizedSmoke.has(query))) {
    throw new Error("Main pool has a normalized-query overlap with the frozen smoke pool");
  }
  const holdouts = loadHoldouts();
  if (main.some(input => holdouts.has(normalizeQuery(input.query)))) throw new Error("Main pool overlaps a frozen holdout");

  const poolBytes = `${main.map(input => JSON.stringify(input)).join("\n")}\n`;
  const sourceCounts = Object.fromEntries(SOURCES.map(source => [
    source.sourceId,
    main.filter(input => input.source_id === source.sourceId).length,
  ]));
  const manifest = {
    version: "qmd-public-main-pool-v0",
    status: "frozen_pre_generation",
    experiment_seed: 42,
    sampling: "sha256(qmd-public-v0\\0seed=42\\0 + source_id + \\0 + qid), then sample_key/qid ascending per source",
    normalization: "NFKC-casefold-trim-collapse-whitespace",
    quotas: sourceCounts,
    total_count: main.length,
    pool_path: `prepared/${basename(POOL_PATH)}`,
    pool_sha256: sha256(poolBytes),
    smoke_pool_path: "prepared/pool-smoke.jsonl",
    smoke_pool_sha256: sha256(smokeBytes),
    smoke_overlap_count: 0,
    smoke_normalized_query_overlap_count: 0,
    holdouts: ["fiqa-dev", "fiqa-test", "cqadup-android", "cqadup-webmasters"],
    holdout_normalized_query_count: holdouts.size,
    holdout_overlap_count: 0,
    cross_source_normalized_duplicate_count: 0,
    unique_input_id_count: mainIds.size,
    unique_normalized_query_count: normalizedMain.size,
    source_artifacts: sourceArtifacts,
    smoke_only: false,
    final_sft_eligible: false,
  };
  return {
    poolBytes,
    manifestBytes: `${JSON.stringify(manifest, null, 2)}\n`,
    summary: { pool: POOL_PATH, manifest: MANIFEST_PATH, total_count: main.length, source_counts: sourceCounts, pool_sha256: manifest.pool_sha256 },
  };
}

function atomicWrite(path: string, bytes: string): void {
  const temporary = `${path}.tmp`;
  writeFileSync(temporary, bytes, { encoding: "utf8", flag: "wx" });
  renameSync(temporary, path);
}

const artifacts = buildArtifacts();
if (existsSync(POOL_PATH) || existsSync(MANIFEST_PATH)) {
  if (!existsSync(POOL_PATH) || !existsSync(MANIFEST_PATH)) throw new Error("Main pool artifacts are incomplete");
  if (readFileSync(POOL_PATH, "utf8") !== artifacts.poolBytes || readFileSync(MANIFEST_PATH, "utf8") !== artifacts.manifestBytes) {
    throw new Error("Frozen main pool artifacts do not match deterministic reconstruction");
  }
  process.stdout.write(`${JSON.stringify({ ...artifacts.summary as object, verified_existing: true }, null, 2)}\n`);
} else {
  atomicWrite(POOL_PATH, artifacts.poolBytes);
  atomicWrite(MANIFEST_PATH, artifacts.manifestBytes);
  process.stdout.write(`${JSON.stringify({ ...artifacts.summary as object, verified_existing: false }, null, 2)}\n`);
}
