import { openDatabase } from "../db.js";
import type { Database } from "../db.js";
import fastGlob from "fast-glob";
import { spawn as nodeSpawn } from "child_process";
import { fileURLToPath } from "url";
import { basename, dirname, join as pathJoin, relative as relativePath, resolve as pathResolve } from "path";
import { parseArgs } from "util";
import { readFileSync, readdirSync, realpathSync, statSync, existsSync, unlinkSync, writeFileSync, openSync, closeSync, mkdirSync } from "fs";
import {
  getPwd,
  getRealPath,
  homedir,
  resolve,
  enableProductionMode,
  searchFTS,
  getContextForFile,
  listCollections,
  removeCollection,
  renameCollection,
  findSimilarFiles,
  getHashesNeedingEmbedding,
  clearAllEmbeddings,
  insertEmbedding,
  getStatus,
  hashContent,
  extractTitle,
  clearCache,
  getCacheKey,
  getCachedResult,
  setCachedResult,
  getIndexHealth,
  parseVirtualPath,
  resolveVirtualPath,
  toVirtualPath,
  insertContent,
  insertDocument,
  findActiveDocument,
  findOrMigrateLegacyDocument,
  updateDocumentTitle,
  updateDocument,
  deactivateDocument,
  getActiveDocumentPaths,
  cleanupOrphanedContent,
  deleteLLMCache,
  deleteInactiveDocuments,
  cleanupOrphanedVectors,
  vacuumDatabase,
  getCollectionsWithoutContext,
  getTopLevelPathsWithoutContext,
  handelize,
  hybridQuery,
  vectorSearchQuery,
  structuredSearch,
  type ExpandedQuery,
  DEFAULT_EMBED_MODEL,
  DEFAULT_EMBED_MAX_BATCH_BYTES,
  DEFAULT_EMBED_MAX_DOCS_PER_BATCH,
  DEFAULT_RERANK_MODEL,
  DEFAULT_GLOB,
  DEFAULT_MULTI_GET_MAX_BYTES,
  createStore,
  reindexCollection,
  generateEmbeddings,
  type ReindexResult,
  type ChunkStrategy,
} from "../store.js";
import { disposeDefaultLlamaCpp, withLLMSession, pullModels, DEFAULT_MODEL_CACHE_DIR, resolveEmbedModel, resolveGenerateModel, resolveRerankModel } from "../llm.js";
import {
  formatSearchResults,
  formatDocuments,
  type OutputFormat,
} from "./formatter.js";
import {
  getCollection as getCollectionFromYaml,
  listCollections as yamlListCollections,
  getDefaultCollectionNames,
  removeCollection as yamlRemoveCollectionFn,
  renameCollection as yamlRenameCollectionFn,
  listAllContexts,
  setConfigIndexName,
  setConfigSource,
  findLocalConfigPath,
  getLocalDbPath,
  getConfigPath,
  configExists,
} from "../collections.js";
import {
  getStore,
  getDb,
  resyncConfig,
  closeDb,
  getDbPath,
  setStoreDbPathOverride,
  getActiveIndexName,
  setIndexName,
  ensureVecTable,
  ensureModelsConfiguredForCli,
  resolveEmbedModelForCli,
  resolveModelsForCli,
} from "./context.js";
import {
  c,
  cursor,
  isTTY,
  progress,
  formatETA,
  formatMs,
  formatBytes,
  renderProgressBar,
  highlightTerms,
  formatScore,
  formatExplainNumber,
} from "./term.js";
import {
  printEmptySearchResults,
  outputResults,
  type OutputOptions,
} from "./output.js";
import {
  showHelp,
  showSkillsHelp,
  printDoctorHint,
  readPackageJson,
  showVersion,
} from "./help.js";
import { runSkillsCommand, showSkill, installSkill, outputSkillsJson } from "./commands/skills.js";
import { showStatus, showDoctor, formatCount, shortModelName } from "./commands/doctor.js";
import { getDocument, multiGet, listFiles } from "./commands/docs.js";
import {
  initLocalIndex,
  contextAdd,
  contextList,
  contextRemove,
  collectionList,
  collectionRemove,
  collectionRename,
} from "./commands/collections.js";

// I4: these remain importable from src/cli/qmd.ts for existing consumers
// (test/cli.test.ts, cli-exit-lifecycle.test.ts) after moving to context.ts.
export {
  resolveEmbedModelForCli,
  resolveGenerateModelForCli,
  resolveRerankModelForCli,
} from "./context.js";
export { buildEditorUri, termLink } from "./output.js";

// NOTE: enableProductionMode() is intentionally NOT called at module scope here.
// Importing this module for its exports (e.g. buildEditorUri, termLink from
// test/cli.test.ts) must not flip the global production flag, as that leaks
// into unrelated tests that rely on the default (development) database path
// resolution. The flag is flipped inside the CLI's main-module guard below so
// it only fires when qmd is actually invoked as a script.

type CliLifecycleWritable = {
  write(chunk: string | Uint8Array, callback?: (error?: Error | null) => void): boolean;
};

type FinishSuccessfulCliCommandOptions = {
  command: string;
  format?: OutputFormat;
  cleanup?: () => Promise<void>;
  exit?: (code: number) => void;
  stdout?: CliLifecycleWritable;
  stderr?: CliLifecycleWritable;
};

async function flushWritable(stream: CliLifecycleWritable): Promise<void> {
  await new Promise<void>((resolve) => {
    stream.write("", () => resolve());
  });
}

/**
 * Finish a successful CLI command after output has been flushed.
 *
 * We deliberately do NOT call `process.exit(0)`. `process.exit()` skips
 * Node's `beforeExit` event, and node-llama-cpp registers a `beforeExit` hook
 * that auto-disposes its native handles. On darwin, without that hook firing,
 * libggml-metal's static `ggml_metal_device` destructor asserts on a
 * non-empty residency-set collection during `__cxa_finalize_ranges` and
 * dumps a multi-kB backtrace (upstream ggml-org/llama.cpp#22593, fix open as
 * PR #22595). Empirically, even with explicit `disposeDefaultLlamaCpp()` the
 * direct `process.exit(0)` path still trips the assertion — letting the
 * event loop drain naturally is what actually clears the rsets.
 *
 * So: set `process.exitCode = 0` and return. The main module finishes, the
 * event loop drains, `beforeExit` fires, native resources tear down in
 * order, and the process exits cleanly. The `GGML_METAL_NO_RESIDENCY=1` env
 * var that `bin/qmd` exports is a defense-in-depth safety net for paths
 * that still call `process.exit()` after loading the native binding
 * (signal handlers, error paths, `bun test`).
 *
 * If the caller passes an explicit `exit` for testability, we honor it —
 * the lifecycle tests verify the legacy flush → cleanup → exit ordering.
 * Production callers must not pass `exit`.
 */
export async function finishSuccessfulCliCommand(options: FinishSuccessfulCliCommandOptions): Promise<void> {
  const stderr = options.stderr ?? process.stderr;

  await flushWritable(options.stdout ?? process.stdout);

  try {
    await (options.cleanup ?? disposeDefaultLlamaCpp)();
  } catch (error) {
    stderr.write(
      `QMD Warning: cleanup after successful output failed (${error instanceof Error ? error.message : String(error)}); exiting 0 because command output completed.\n`
    );
  }
  await flushWritable(stderr);

  if (options.exit) {
    options.exit(0);
    return;
  }

  process.exitCode = 0;
}


// Check index health and print warnings/tips
function checkIndexHealth(db: Database, model: string = resolveEmbedModelForCli()): void {
  const { needsEmbedding, totalDocs, daysStale } = getIndexHealth(db, model);

  // Warn if many docs need embedding
  if (needsEmbedding > 0) {
    const pct = Math.round((needsEmbedding / totalDocs) * 100);
    if (pct >= 10) {
      process.stderr.write(`${c.yellow}Warning: ${needsEmbedding} documents (${pct}%) need embeddings. Run 'qmd embed' for better results.${c.reset}\n`);
    } else {
      process.stderr.write(`${c.dim}Tip: ${needsEmbedding} documents need embeddings. Run 'qmd embed' to index them.${c.reset}\n`);
    }
  }

  // Check if most recent document update is older than 2 weeks
  if (daysStale !== null && daysStale >= 14) {
    process.stderr.write(`${c.dim}Tip: Index last updated ${daysStale} days ago. Run 'qmd update' to refresh.${c.reset}\n`);
  }
}

// Compute unique display path for a document
// Always include at least parent folder + filename, add more parent dirs until unique
async function updateCollections(): Promise<void> {
  const db = getDb();
  const storeInstance = getStore();
  // Collections are defined in YAML; no duplicate cleanup needed.

  // Clear Ollama cache on update
  clearCache(db);

  const collections = listCollections(db);

  if (collections.length === 0) {
    console.log(`${c.dim}No collections found. Run 'qmd collection add .' to index markdown files.${c.reset}`);
    closeDb();
    return;
  }

  console.log(`${c.bold}Updating ${collections.length} collection(s)...${c.reset}\n`);

  for (let i = 0; i < collections.length; i++) {
    const col = collections[i];
    if (!col) continue;
    console.log(`${c.cyan}[${i + 1}/${collections.length}]${c.reset} ${c.bold}${col.name}${c.reset} ${c.dim}(${col.glob_pattern})${c.reset}`);

    // Execute custom update command if specified in YAML
    const yamlCol = getCollectionFromYaml(col.name);
    if (yamlCol?.update) {
      console.log(`${c.dim}    Running update command: ${yamlCol.update}${c.reset}`);
      try {
        const proc = nodeSpawn("bash", ["-c", yamlCol.update], {
          cwd: col.pwd,
          stdio: ["ignore", "pipe", "pipe"],
        });

        const [output, errorOutput, exitCode] = await new Promise<[string, string, number]>((resolve, reject) => {
          let out = "";
          let err = "";
          proc.stdout?.on("data", (d: Buffer) => { out += d.toString(); });
          proc.stderr?.on("data", (d: Buffer) => { err += d.toString(); });
          proc.on("error", reject);
          proc.on("close", (code) => resolve([out, err, code ?? 1]));
        });

        if (output.trim()) {
          console.log(output.trim().split('\n').map(l => `    ${l}`).join('\n'));
        }
        if (errorOutput.trim()) {
          console.log(errorOutput.trim().split('\n').map(l => `    ${l}`).join('\n'));
        }

        if (exitCode !== 0) {
          console.log(`${c.yellow}✗ Update command failed with exit code ${exitCode}${c.reset}`);
          process.exit(exitCode);
        }
      } catch (err) {
        console.log(`${c.yellow}✗ Update command failed: ${err}${c.reset}`);
        process.exit(1);
      }
    }

    const startTime = Date.now();
    console.log(`Collection: ${col.pwd} (${col.glob_pattern})`);
    progress.indeterminate();

    const result = await reindexCollection(storeInstance, col.pwd, col.glob_pattern, col.name, {
      ignorePatterns: yamlCol?.ignore,
      onProgress: (info) => {
        progress.set((info.current / info.total) * 100);
        const elapsed = (Date.now() - startTime) / 1000;
        const rate = info.current / elapsed;
        const remaining = (info.total - info.current) / rate;
        const eta = info.current > 2 ? ` ETA: ${formatETA(remaining)}` : "";
        if (isTTY) process.stderr.write(`\rIndexing: ${info.current}/${info.total}${eta}        `);
      },
    });

    progress.clear();
    console.log(`\nIndexed: ${result.indexed} new, ${result.updated} updated, ${result.unchanged} unchanged, ${result.removed} removed`);
    if (result.orphanedCleaned > 0) {
      console.log(`Cleaned up ${result.orphanedCleaned} orphaned content hash(es)`);
    }
    console.log("");
  }

  // Check if any documents need embedding (show once at end)
  const needsEmbedding = getHashesNeedingEmbedding(db);
  closeDb();

  console.log(`${c.green}✓ All collections updated.${c.reset}`);
  if (needsEmbedding > 0) {
    console.log(`\nRun 'qmd embed' to update embeddings (${needsEmbedding} unique hashes need vectors)`);
  }
}

/**
 * Detect which collection (if any) contains the given filesystem path.
 * Returns { collectionId, collectionName, relativePath } or null if not in any collection.
 */
async function collectionAdd(pwd: string, globPattern: string, name?: string): Promise<void> {
  // If name not provided, generate from pwd basename
  let collName = name;
  if (!collName) {
    const parts = pwd.split('/').filter(Boolean);
    collName = parts[parts.length - 1] || 'root';
  }

  // Check if collection with this name already exists in YAML
  const existing = getCollectionFromYaml(collName);
  if (existing) {
    console.error(`${c.yellow}Collection '${collName}' already exists.${c.reset}`);
    console.error(`Use a different name with --name <name>`);
    process.exit(1);
  }

  // Check if a collection with this pwd+glob already exists in YAML
  const allCollections = yamlListCollections();
  const existingPwdGlob = allCollections.find(c => c.path === pwd && c.pattern === globPattern);

  if (existingPwdGlob) {
    console.error(`${c.yellow}A collection already exists for this path and pattern:${c.reset}`);
    console.error(`  Name: ${existingPwdGlob.name} (qmd://${existingPwdGlob.name}/)`);
    console.error(`  Pattern: ${globPattern}`);
    console.error(`\nUse 'qmd update' to re-index it, or remove it first with 'qmd collection remove ${existingPwdGlob.name}'`);
    process.exit(1);
  }

  // Add to YAML config + sync to SQLite
  const { addCollection } = await import("../collections.js");
  addCollection(collName, pwd, globPattern);
  resyncConfig();

  // Create the collection and index files
  console.log(`Creating collection '${collName}'...`);
  const newColl = getCollectionFromYaml(collName);
  await indexFiles(pwd, globPattern, collName, false, newColl?.ignore);
  console.log(`${c.green}✓${c.reset} Collection '${collName}' created successfully`);
}

async function indexFiles(pwd?: string, globPattern: string = DEFAULT_GLOB, collectionName?: string, suppressEmbedNotice: boolean = false, ignorePatterns?: string[]): Promise<void> {
  const db = getDb();
  const resolvedPwd = pwd || getPwd();
  const now = new Date().toISOString();
  const excludeDirs = ["node_modules", ".git", ".cache", "vendor", "dist", "build"];

  // Clear Ollama cache on index
  clearCache(db);

  // Collection name must be provided (from YAML)
  if (!collectionName) {
    throw new Error("Collection name is required. Collections must be defined in ~/.config/qmd/index.yml");
  }

  console.log(`Collection: ${resolvedPwd} (${globPattern})`);

  progress.indeterminate();
  const allIgnore = [
    ...excludeDirs.map(d => `**/${d}/**`),
    ...(ignorePatterns || []),
  ];
  const allFiles: string[] = await fastGlob(globPattern, {
    cwd: resolvedPwd,
    onlyFiles: true,
    followSymbolicLinks: false,
    dot: false,
    ignore: allIgnore,
  });
  // Filter hidden files/folders (dot: false handles top-level but not nested)
  const files = allFiles.filter(file => {
    const parts = file.split("/");
    return !parts.some(part => part.startsWith("."));
  });

  const total = files.length;
  const hasNoFiles = total === 0;
  if (hasNoFiles) {
    progress.clear();
    console.log("No files found matching pattern.");
    // Continue so the deactivation pass can mark previously indexed docs as inactive.
  }

  let indexed = 0, updated = 0, unchanged = 0, processed = 0;
  const seenPaths = new Set<string>();
  const startTime = Date.now();

  for (const relativeFile of files) {
    const filepath = getRealPath(resolve(resolvedPwd, relativeFile));
    // Store the literal relative path — handelize() is NOT applied at index time.
    const path = relativeFile.replace(/\\/g, '/');
    seenPaths.add(path);

    let content: string;
    try {
      content = readFileSync(filepath, "utf-8");
    } catch {
      // Skip files that can't be read (e.g. iCloud evicted files returning EAGAIN)
      processed++;
      progress.set((processed / total) * 100);
      continue;
    }

    // Skip empty files - nothing useful to index
    if (!content.trim()) {
      processed++;
      continue;
    }

    const hash = await hashContent(content);
    const title = extractTitle(content, relativeFile);

    // Check if document exists (also migrates legacy lowercase paths)
    const existing = findOrMigrateLegacyDocument(db, collectionName, path);

    if (existing) {
      if (existing.hash === hash) {
        // Hash unchanged, but check if title needs updating
        if (existing.title !== title) {
          updateDocumentTitle(db, existing.id, title, now);
          updated++;
        } else {
          unchanged++;
        }
      } else {
        // Content changed - insert new content hash and update document
        insertContent(db, hash, content, now);
        const stat = statSync(filepath);
        updateDocument(db, existing.id, title, hash,
          stat ? new Date(stat.mtime).toISOString() : now);
        updated++;
      }
    } else {
      // New document - insert content and document
      indexed++;
      insertContent(db, hash, content, now);
      const stat = statSync(filepath);
      insertDocument(db, collectionName, path, title, hash,
        stat ? new Date(stat.birthtime).toISOString() : now,
        stat ? new Date(stat.mtime).toISOString() : now);
    }

    processed++;
    progress.set((processed / total) * 100);
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = processed / elapsed;
    const remaining = (total - processed) / rate;
    const eta = processed > 2 ? ` ETA: ${formatETA(remaining)}` : "";
    if (isTTY) process.stderr.write(`\rIndexing: ${processed}/${total}${eta}        `);
  }

  // Deactivate documents in this collection that no longer exist
  const allActive = getActiveDocumentPaths(db, collectionName);
  let removed = 0;
  for (const path of allActive) {
    if (!seenPaths.has(path)) {
      deactivateDocument(db, collectionName, path);
      removed++;
    }
  }

  // Clean up orphaned content hashes (content not referenced by any document)
  const orphanedContent = cleanupOrphanedContent(db);

  // Check if vector index needs updating
  const needsEmbedding = getHashesNeedingEmbedding(db);

  progress.clear();
  console.log(`\nIndexed: ${indexed} new, ${updated} updated, ${unchanged} unchanged, ${removed} removed`);
  if (orphanedContent > 0) {
    console.log(`Cleaned up ${orphanedContent} orphaned content hash(es)`);
  }

  if (needsEmbedding > 0 && !suppressEmbedNotice) {
    console.log(`\nRun 'qmd embed' to update embeddings (${needsEmbedding} unique hashes need vectors)`);
  }

  closeDb();
}

function parseEmbedBatchOption(name: string, value: unknown): number | undefined {
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function parseChunkStrategy(value: unknown): ChunkStrategy | undefined {
  if (value === undefined) return undefined;
  const s = String(value);
  if (s === "auto" || s === "regex") return s;
  throw new Error(`--chunk-strategy must be "auto" or "regex" (got "${s}")`);
}

// --timeout for `qmd embed`: a cap on the whole embed session, in minutes. Returns
// the value in milliseconds, or undefined to use the default. 0 disables the cap.
function parseEmbedTimeoutOption(value: unknown): number | undefined {
  if (value === undefined) return undefined;
  const minutes = Number(value);
  if (!Number.isFinite(minutes) || minutes < 0) {
    throw new Error(`--timeout must be a non-negative number of minutes (0 = no limit)`);
  }
  return minutes * 60 * 1000;
}

async function vectorIndex(
  model: string = resolveEmbedModelForCli(),
  force: boolean = false,
  batchOptions?: { maxDocsPerBatch?: number; maxBatchBytes?: number; chunkStrategy?: ChunkStrategy; collection?: string; maxDurationMs?: number },
): Promise<void> {
  const storeInstance = getStore();
  const db = storeInstance.db;

  if (force) {
    console.log(`${c.yellow}Force re-indexing: clearing all vectors...${c.reset}`);
  }

  // Check if there's work to do before starting
  const hashesToEmbed = getHashesNeedingEmbedding(db, batchOptions?.collection, model);
  if (hashesToEmbed === 0 && !force) {
    console.log(`${c.green}✓ All content hashes already have embeddings.${c.reset}`);
    closeDb();
    return;
  }

  console.log(`${c.dim}Model: ${shortModelName(model)}${c.reset}\n`);
  if (batchOptions?.maxDocsPerBatch !== undefined || batchOptions?.maxBatchBytes !== undefined) {
    const maxDocsPerBatch = batchOptions.maxDocsPerBatch ?? DEFAULT_EMBED_MAX_DOCS_PER_BATCH;
    const maxBatchBytes = batchOptions.maxBatchBytes ?? DEFAULT_EMBED_MAX_BATCH_BYTES;
    console.log(`${c.dim}Batch: ${maxDocsPerBatch} docs / ${formatBytes(maxBatchBytes)}${c.reset}\n`);
  }
  cursor.hide();
  progress.indeterminate();

  const startTime = Date.now();

  const result = await generateEmbeddings(storeInstance, {
    force,
    model,
    collection: batchOptions?.collection,
    maxDocsPerBatch: batchOptions?.maxDocsPerBatch,
    maxBatchBytes: batchOptions?.maxBatchBytes,
    chunkStrategy: batchOptions?.chunkStrategy,
    maxDurationMs: batchOptions?.maxDurationMs,
    onProgress: (info) => {
      if (info.totalBytes === 0) return;
      // Progress is measured by input bytes, not by chunks. The final chunk
      // count is discovered lazily batch-by-batch, so displaying
      // chunksEmbedded/totalChunks makes the percent look wrong when a few
      // large documents remain. Show chunks as a count and label the byte
      // percentage explicitly as input progress.
      const percent = Math.min(100, (info.bytesProcessed / info.totalBytes) * 100);
      progress.set(percent);

      const elapsed = (Date.now() - startTime) / 1000;
      const bytesPerSec = elapsed > 0 ? info.bytesProcessed / elapsed : 0;
      const remainingBytes = Math.max(0, info.totalBytes - info.bytesProcessed);
      const etaSec = bytesPerSec > 0 ? remainingBytes / bytesPerSec : Number.POSITIVE_INFINITY;

      const bar = renderProgressBar(percent);
      const percentStr = percent.toFixed(0).padStart(3);
      const throughput = bytesPerSec > 0 ? `${formatBytes(bytesPerSec)}/s` : ".../s";
      const eta = elapsed > 2 && Number.isFinite(etaSec) ? formatETA(etaSec) : "...";
      const inputStr = `${formatBytes(info.bytesProcessed)}/${formatBytes(info.totalBytes)} input`;
      const chunkStr = `${formatCount(info.chunksEmbedded)} chunks`;
      const errStr = info.errors > 0 ? ` ${c.yellow}${formatCount(info.errors)} err${c.reset}` : "";

      if (isTTY) process.stderr.write(`\r${c.cyan}${bar}${c.reset} ${c.bold}${percentStr}% input${c.reset} ${c.dim}${chunkStr}${errStr} · ${inputStr} · ${throughput} · ETA ${eta}${c.reset}   `);
    },
  });

  progress.clear();
  cursor.show();

  const totalTimeSec = result.durationMs / 1000;

  if (result.chunksEmbedded === 0 && result.docsProcessed === 0) {
    console.log(`${c.green}✓ No non-empty documents to embed.${c.reset}`);
  } else {
    console.log(`\r${c.green}${renderProgressBar(100)}${c.reset} ${c.bold}100%${c.reset}                                    `);
    console.log(`\n${c.green}✓ Done!${c.reset} Embedded ${c.bold}${result.chunksEmbedded}${c.reset} chunks from ${c.bold}${result.docsProcessed}${c.reset} documents in ${c.bold}${formatETA(totalTimeSec)}${c.reset}`);
    if (result.errors > 0) {
      console.log(`${c.yellow}⚠ ${formatCount(result.errors)} chunks still failed after retries${c.reset}`);
      for (const failure of (result.failures ?? []).slice(0, 8)) {
        console.log(`  ${c.dim}${failure.path}#${failure.seq} (${failure.attempts} attempts): ${failure.reason}${c.reset}`);
      }
      if ((result.failures?.length ?? 0) > 8) {
        console.log(`  ${c.dim}...and ${formatCount((result.failures?.length ?? 0) - 8)} more${c.reset}`);
      }
    }
  }

  closeDb();
}

// Sanitize a term for FTS5: remove punctuation except apostrophes
function sanitizeFTS5Term(term: string): string {
  // Remove all non-alphanumeric except apostrophes (for contractions like "don't")
  return term.replace(/[^\w']/g, '').trim();
}

// Build FTS5 query: phrase-aware with fallback to individual terms
function buildFTS5Query(query: string): string {
  // Sanitize the full query for phrase matching
  const sanitizedQuery = query.replace(/[^\w\s']/g, '').trim();

  const terms = query
    .split(/\s+/)
    .map(sanitizeFTS5Term)
    .filter(term => term.length >= 2); // Skip single chars and empty

  if (terms.length === 0) return "";
  if (terms.length === 1) return `"${terms[0]!.replace(/"/g, '""')}"`;

  // Strategy: exact phrase OR proximity match OR individual terms
  // Exact phrase matches rank highest, then close proximity, then any term
  const phrase = `"${sanitizedQuery.replace(/"/g, '""')}"`;
  const quotedTerms = terms.map(t => `"${t.replace(/"/g, '""')}"`);

  // FTS5 NEAR syntax: NEAR(term1 term2, distance)
  const nearPhrase = `NEAR(${quotedTerms.join(' ')}, 10)`;
  const orTerms = quotedTerms.join(' OR ');

  // Exact phrase > proximity > any term
  return `(${phrase}) OR (${nearPhrase}) OR (${orTerms})`;
}

// Normalize BM25 score to 0-1 range using sigmoid
function normalizeBM25(score: number): number {
  // BM25 scores are negative in SQLite (lower = better)
  // Typical range: -15 (excellent) to -2 (weak match)
  // Map to 0-1 where higher is better
  const absScore = Math.abs(score);
  // Sigmoid-ish normalization: maps ~2-15 range to ~0.1-0.95
  return 1 / (1 + Math.exp(-(absScore - 5) / 3));
}

// Resolve -c collection filter: supports single string, array, or undefined.
// Returns validated collection names (exits on unknown collection).
function resolveCollectionFilter(raw: string | string[] | undefined, useDefaults: boolean = false): string[] {
  // If no filter specified and useDefaults is true, use default collections
  if (!raw && useDefaults) {
    return getDefaultCollectionNames();
  }
  if (!raw) return [];
  const names = Array.isArray(raw) ? raw : [raw];
  const validated: string[] = [];
  for (const name of names) {
    const coll = getCollectionFromYaml(name);
    if (!coll) {
      console.error(`Collection not found: ${name}`);
      closeDb();
      process.exit(1);
    }
    validated.push(name);
  }
  return validated;
}

// Post-filter results to only include files from specified collections.
function filterByCollections<T extends { filepath?: string; file?: string }>(results: T[], collectionNames: string[]): T[] {
  if (collectionNames.length <= 1) return results;
  const prefixes = collectionNames.map(n => `qmd://${n}/`);
  return results.filter(r => {
    const path = r.filepath || r.file || '';
    return prefixes.some(p => path.startsWith(p));
  });
}

/**
 * Parse structured search query syntax.
 * Lines starting with lex:, vec:, or hyde: are routed directly.
 * Plain lines without prefix go through query expansion.
 * 
 * Returns null if this is a plain query (single line, no prefix).
 * Returns ExpandedQuery[] if structured syntax detected.
 * Throws if multiple plain lines (ambiguous).
 * 
 * Examples:
 *   "CAP theorem"                    -> null (plain query, use expansion)
 *   "lex: CAP theorem"               -> [{ type: 'lex', query: 'CAP theorem' }]
 *   "lex: CAP\nvec: consistency"     -> [{ type: 'lex', ... }, { type: 'vec', ... }]
 *   "CAP\nconsistency"               -> throws (multiple plain lines)
 */
interface ParsedStructuredQuery {
  searches: ExpandedQuery[];
  intent?: string;
}

function parseStructuredQuery(query: string): ParsedStructuredQuery | null {
  const rawLines = query.split('\n').map((line, idx) => ({
    raw: line,
    trimmed: line.trim(),
    number: idx + 1,
  })).filter(line => line.trimmed.length > 0);

  if (rawLines.length === 0) return null;

  const prefixRe = /^(lex|vec|hyde):\s*/i;
  const expandRe = /^expand:\s*/i;
  const intentRe = /^intent:\s*/i;
  const typed: ExpandedQuery[] = [];
  let intent: string | undefined;

  for (const line of rawLines) {
    if (expandRe.test(line.trimmed)) {
      if (rawLines.length > 1) {
        throw new Error(`Line ${line.number} starts with expand:, but query documents cannot mix expand with typed lines. Submit a single expand query instead.`);
      }
      const text = line.trimmed.replace(expandRe, '').trim();
      if (!text) {
        throw new Error('expand: query must include text.');
      }
      return null; // treat as standalone expand query
    }

    // Parse intent: lines
    if (intentRe.test(line.trimmed)) {
      if (intent !== undefined) {
        throw new Error(`Line ${line.number}: only one intent: line is allowed per query document.`);
      }
      const text = line.trimmed.replace(intentRe, '').trim();
      if (!text) {
        throw new Error(`Line ${line.number}: intent: must include text.`);
      }
      intent = text;
      continue;
    }

    const match = line.trimmed.match(prefixRe);
    if (match) {
      const type = match[1]!.toLowerCase() as 'lex' | 'vec' | 'hyde';
      const text = line.trimmed.slice(match[0].length).trim();
      if (!text) {
        throw new Error(`Line ${line.number} (${type}:) must include text.`);
      }
      if (/\r|\n/.test(text)) {
        throw new Error(`Line ${line.number} (${type}:) contains a newline. Keep each query on a single line.`);
      }
      typed.push({ type, query: text, line: line.number });
      continue;
    }

    if (rawLines.length === 1) {
      // Single plain line -> implicit expand
      return null;
    }

    throw new Error(`Line ${line.number} is missing a lex:/vec:/hyde:/intent: prefix. Each line in a query document must start with one.`);
  }

  // intent: alone is not a valid query — must have at least one search
  if (intent && typed.length === 0) {
    throw new Error('intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line.');
  }

  return typed.length > 0 ? { searches: typed, intent } : null;
}

function search(query: string, opts: OutputOptions): void {
  const db = getDb();

  // Validate collection filter (supports multiple -c flags)
  // Use default collections if none specified
  const collectionNames = resolveCollectionFilter(opts.collection, true);
  const singleCollection = collectionNames.length === 1 ? collectionNames[0] : undefined;

  // Use large limit for --all, otherwise fetch more than needed and let outputResults filter
  const fetchLimit = opts.all ? 100000 : Math.max(50, opts.limit * 2);
  const results = filterByCollections(
    searchFTS(db, query, fetchLimit, singleCollection),
    collectionNames
  );

  // Add context to results
  const resultsWithContext = results.map(r => ({
    file: r.filepath,
    displayPath: r.displayPath,
    title: r.title,
    body: r.body || "",
    score: r.score,
    context: getContextForFile(db, r.filepath),
    hash: r.hash,
    docid: r.docid,
  }));

  closeDb();

  if (resultsWithContext.length === 0) {
    printEmptySearchResults(opts.format);
    return;
  }
  outputResults(resultsWithContext, query, opts);
}

// Log query expansion as a tree to stderr (CLI progress feedback)
function logExpansionTree(originalQuery: string, expanded: ExpandedQuery[]): void {
  const lines: string[] = [];
  lines.push(`${c.dim}├─ ${originalQuery}${c.reset}`);
  for (const q of expanded) {
    let preview = q.query.replace(/\n/g, ' ');
    if (preview.length > 72) preview = preview.substring(0, 69) + '...';
    lines.push(`${c.dim}├─ ${q.type}: ${preview}${c.reset}`);
  }
  if (lines.length > 0) {
    lines[lines.length - 1] = lines[lines.length - 1]!.replace('├─', '└─');
  }
  for (const line of lines) process.stderr.write(line + '\n');
}

async function vectorSearch(query: string, opts: OutputOptions, _model: string = DEFAULT_EMBED_MODEL): Promise<void> {
  const store = getStore();

  // Validate collection filter (supports multiple -c flags)
  // Use default collections if none specified
  const collectionNames = resolveCollectionFilter(opts.collection, true);
  const singleCollection = collectionNames.length === 1 ? collectionNames[0] : undefined;

  checkIndexHealth(store.db);

  await withLLMSession(async () => {
    let results = await vectorSearchQuery(store, query, {
      collection: singleCollection,
      limit: opts.all ? 500 : (opts.limit || 10),
      minScore: opts.minScore || 0.3,
      intent: opts.intent,
      hooks: {
        onExpand: (original, expanded) => {
          logExpansionTree(original, expanded);
          process.stderr.write(`${c.dim}Searching ${expanded.length + 1} vector queries...${c.reset}\n`);
        },
      },
    });

    // Post-filter for multi-collection
    if (collectionNames.length > 1) {
      results = results.filter(r => {
        const prefixes = collectionNames.map(n => `qmd://${n}/`);
        return prefixes.some(p => r.file.startsWith(p));
      });
    }

    closeDb();

    if (results.length === 0) {
      printEmptySearchResults(opts.format);
      return;
    }

    outputResults(results.map(r => ({
      file: r.file,
      displayPath: r.displayPath,
      title: r.title,
      body: r.body,
      score: r.score,
      context: r.context,
      docid: r.docid,
    })), query, { ...opts, limit: results.length });
  }, { maxDuration: 10 * 60 * 1000, name: 'vectorSearch' });
}

async function querySearch(query: string, opts: OutputOptions, _embedModel: string = DEFAULT_EMBED_MODEL, _rerankModel: string = DEFAULT_RERANK_MODEL): Promise<void> {
  const store = getStore();

  // Validate collection filter (supports multiple -c flags)
  // Use default collections if none specified
  const collectionNames = resolveCollectionFilter(opts.collection, true);
  const singleCollection = collectionNames.length === 1 ? collectionNames[0] : undefined;

  checkIndexHealth(store.db);

  // Check for structured query syntax (lex:/vec:/hyde:/intent: prefixes)
  const parsed = parseStructuredQuery(query);
  // Intent can come from --intent flag or from intent: line in query document
  const intent = opts.intent || parsed?.intent;

  await withLLMSession(async () => {
    let results;

    if (parsed) {
      const structuredQueries = parsed.searches;
      // Structured search — user provided their own query expansions
      const typeLabels = structuredQueries.map(s => s.type).join('+');
      process.stderr.write(`${c.dim}Structured search: ${structuredQueries.length} queries (${typeLabels})${c.reset}\n`);
      if (intent) {
        process.stderr.write(`${c.dim}├─ intent: ${intent}${c.reset}\n`);
      }

      // Log each sub-query
      for (const s of structuredQueries) {
        let preview = s.query.replace(/\n/g, ' ');
        if (preview.length > 72) preview = preview.substring(0, 69) + '...';
        process.stderr.write(`${c.dim}├─ ${s.type}: ${preview}${c.reset}\n`);
      }
      process.stderr.write(`${c.dim}└─ Searching...${c.reset}\n`);

      results = await structuredSearch(store, structuredQueries, {
        collections: singleCollection ? [singleCollection] : undefined,
        limit: opts.all ? 500 : (opts.limit || 10),
        minScore: opts.minScore || 0,
        candidateLimit: opts.candidateLimit,
        skipRerank: opts.skipRerank,
        explain: !!opts.explain,
        intent,
        chunkStrategy: opts.chunkStrategy,
        hooks: {
          onEmbedStart: (count) => {
            process.stderr.write(`${c.dim}Embedding ${count} ${count === 1 ? 'query' : 'queries'}...${c.reset}`);
          },
          onEmbedDone: (ms) => {
            process.stderr.write(`${c.dim} (${formatMs(ms)})${c.reset}\n`);
          },
          onRerankStart: (chunkCount) => {
            process.stderr.write(`${c.dim}Reranking ${chunkCount} chunks...${c.reset}`);
            progress.indeterminate();
          },
          onRerankDone: (ms) => {
            progress.clear();
            process.stderr.write(`${c.dim} (${formatMs(ms)})${c.reset}\n`);
          },
        },
      });
    } else {
      // Standard hybrid query with automatic expansion
      results = await hybridQuery(store, query, {
        collection: singleCollection,
        limit: opts.all ? 500 : (opts.limit || 10),
        minScore: opts.minScore || 0,
        candidateLimit: opts.candidateLimit,
        skipRerank: opts.skipRerank,
        explain: !!opts.explain,
        intent,
        chunkStrategy: opts.chunkStrategy,
        hooks: {
          onStrongSignal: (score) => {
            process.stderr.write(`${c.dim}Strong BM25 signal (${score.toFixed(2)}) — skipping expansion${c.reset}\n`);
          },
          onExpandStart: () => {
            process.stderr.write(`${c.dim}Expanding query...${c.reset}`);
          },
          onExpand: (original, expanded, ms) => {
            process.stderr.write(`${c.dim} (${formatMs(ms)})${c.reset}\n`);
            logExpansionTree(original, expanded);
            process.stderr.write(`${c.dim}Searching ${expanded.length + 1} queries...${c.reset}\n`);
          },
          onEmbedStart: (count) => {
            process.stderr.write(`${c.dim}Embedding ${count} ${count === 1 ? 'query' : 'queries'}...${c.reset}`);
          },
          onEmbedDone: (ms) => {
            process.stderr.write(`${c.dim} (${formatMs(ms)})${c.reset}\n`);
          },
          onRerankStart: (chunkCount) => {
            process.stderr.write(`${c.dim}Reranking ${chunkCount} chunks...${c.reset}`);
            progress.indeterminate();
          },
          onRerankDone: (ms) => {
            progress.clear();
            process.stderr.write(`${c.dim} (${formatMs(ms)})${c.reset}\n`);
          },
        },
      });
    }

    // Post-filter for multi-collection
    if (collectionNames.length > 1) {
      results = results.filter(r => {
        const prefixes = collectionNames.map(n => `qmd://${n}/`);
        return prefixes.some(p => r.file.startsWith(p));
      });
    }

    closeDb();

    if (results.length === 0) {
      printEmptySearchResults(opts.format);
      return;
    }

    // Use first lex/vec query for output context, or original query
    const structuredQueries = parsed?.searches;
    const displayQuery = structuredQueries
      ? (structuredQueries.find(s => s.type === 'lex')?.query || structuredQueries.find(s => s.type === 'vec')?.query || query)
      : query;

    outputResults(results.map(r => ({
      file: r.file,
      displayPath: r.displayPath,
      title: r.title,
      body: r.body,
      chunkPos: r.bestChunkPos,
      chunkLen: r.bestChunk.length,
      score: r.score,
      context: r.context,
      docid: r.docid,
      explain: r.explain,
    })), displayQuery, { ...opts, limit: results.length });
  }, { maxDuration: 10 * 60 * 1000, name: 'querySearch' });
}

// Parse CLI arguments using util.parseArgs
function parseCLI() {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2), // Skip node and script path
    options: {
      // Global options
      index: {
        type: "string",
      },
      context: {
        type: "string",
      },
      help: { type: "boolean", short: "h" },
      version: { type: "boolean", short: "v" },
      skill: { type: "boolean" },
      global: { type: "boolean" },
      yes: { type: "boolean" },
      // Search options
      n: { type: "string" },
      "min-score": { type: "string" },
      all: { type: "boolean" },
      full: { type: "boolean" },
      format: { type: "string" },          // preferred: --format cli|json|csv|md|xml|files
      // Legacy boolean format aliases. Kept working for back-compat but
      // omitted from the documented help; prefer `--format <kind>`.
      csv: { type: "boolean" },
      md: { type: "boolean" },
      xml: { type: "boolean" },
      files: { type: "boolean" },
      json: { type: "boolean" },
      explain: { type: "boolean" },
      collection: { type: "string", short: "c", multiple: true },  // Filter by collection(s)
      // Collection options
      name: { type: "string" },  // collection name
      mask: { type: "string" },  // glob pattern
      // Embed options
      force: { type: "boolean", short: "f" },
      "max-docs-per-batch": { type: "string" },
      "max-batch-mb": { type: "string" },
      timeout: { type: "string" },  // embed session cap in minutes (0 = no limit; default 30)
      // Update options
      pull: { type: "boolean" },  // git pull before update
      refresh: { type: "boolean" },
      // Get options
      l: { type: "string" },  // max lines
      from: { type: "string" },  // start line
      "max-bytes": { type: "string" },  // max bytes for multi-get
      "line-numbers": { type: "boolean" },  // add line numbers to output (search; default on for get/multi-get)
      "no-line-numbers": { type: "boolean" },  // disable line numbers for get/multi-get
      "full-path": { type: "boolean" },  // show on-disk paths instead of qmd:// (get/multi-get/search/query)
      // Query options
      "candidate-limit": { type: "string", short: "C" },
      "no-rerank": { type: "boolean", default: false },
      "no-gpu": { type: "boolean", default: false },
      intent: { type: "string" },
      // Benchmark v2 options
      run: { type: "string" },
      model: { type: "string" },
      only: { type: "string" },
      "check-index": { type: "boolean" },
      // Chunking options
      "chunk-strategy": { type: "string" },  // "regex" (default) or "auto" (AST for code files)
      // MCP HTTP transport options
      http: { type: "boolean" },
      daemon: { type: "boolean" },
      port: { type: "string" },
      host: { type: "string" },
    },
    allowPositionals: true,
    strict: false, // Allow unknown options to pass through
  });

  if (values["no-gpu"]) {
    process.env.QMD_FORCE_CPU = "1";
  }

  // Select index name (default: "index"). If no explicit --index is supplied,
  // a project-local .qmd/index.yaml overrides the global config/cache paths.
  const indexName = values.index as string | undefined;
  if (indexName) {
    setIndexName(indexName);
    setConfigIndexName(indexName);
    setConfigSource();
  } else {
    const localConfigPath = findLocalConfigPath();
    if (localConfigPath) {
      setConfigSource({ configPath: localConfigPath });
      setStoreDbPathOverride(getLocalDbPath(localConfigPath));
      closeDb();
    } else {
      setConfigSource();
    }
  }

  // Determine output format. Prefer --format <kind>; fall back to the
  // legacy boolean aliases (--csv/--md/--xml/--files/--json) which remain
  // wired up for back-compat but are no longer documented.
  let format: OutputFormat = "cli";
  const rawFormat = typeof values.format === "string" ? values.format.toLowerCase().trim() : "";
  const VALID_FORMATS: ReadonlyArray<OutputFormat> = ["cli", "json", "csv", "md", "xml", "files"];
  if (rawFormat) {
    if ((VALID_FORMATS as ReadonlyArray<string>).includes(rawFormat)) {
      format = rawFormat as OutputFormat;
    } else {
      console.error(`Unknown --format value: ${values.format}`);
      console.error(`Valid: ${VALID_FORMATS.join(", ")}`);
      process.exit(1);
    }
  } else if (values.csv) format = "csv";
  else if (values.md) format = "md";
  else if (values.xml) format = "xml";
  else if (values.files) format = "files";
  else if (values.json) format = "json";

  // Default limit: 20 for --files/--json, 5 otherwise
  // --all means return all results (use very large limit)
  const defaultLimit = (format === "files" || format === "json") ? 20 : 5;
  const isAll = !!values.all;

  const opts: OutputOptions = {
    format,
    full: !!values.full,
    limit: isAll ? 100000 : (values.n ? parseInt(String(values.n), 10) || defaultLimit : defaultLimit),
    minScore: values["min-score"] ? parseFloat(String(values["min-score"])) || 0 : 0,
    all: isAll,
    collection: values.collection as string[] | undefined,
    lineNumbers: !!values["line-numbers"],
    candidateLimit: values["candidate-limit"] ? parseInt(String(values["candidate-limit"]), 10) : undefined,
    skipRerank: !!values["no-rerank"],
    explain: !!values.explain,
    intent: values.intent as string | undefined,
    chunkStrategy: parseChunkStrategy(values["chunk-strategy"]),
    fullPath: !!values["full-path"],
  };

  return {
    command: positionals[0] || "",
    args: positionals.slice(1),
    query: positionals.slice(1).join(" "),
    opts,
    values,
  };
}

function exitWithError(error: unknown, code = 1): never {
  console.error(error instanceof Error ? error.message : String(error));
  printDoctorHint();
  process.exit(code);
}

// Main CLI - only run if this is the main module
const __filename = fileURLToPath(import.meta.url);
const argv1 = process.argv[1];
const isMain = argv1 === __filename
  || argv1?.endsWith("/qmd.ts")
  || argv1?.endsWith("/qmd.js")
  || (argv1 != null && realpathSync(argv1) === __filename);
if (isMain) {
  // Flip to production mode only when this module is executed as the CLI
  // entrypoint, not when imported for its exports. Tests must set INDEX_PATH
  // or use createStore() with an explicit path.
  enableProductionMode();

  const cli = parseCLI();

  if (cli.values.version) {
    await showVersion();
    process.exit(0);
  }

  if (cli.values.skill) {
    showSkill();
    process.exit(0);
  }

  if (cli.values.help && cli.command === "skill") {
    console.log("Usage: qmd skill <show|install> [options]");
    console.log("");
    console.log("Commands:");
    console.log("  show                 Print the QMD skill");
    console.log("  install              Install QMD skill into ./.agents/skills/qmd");
    console.log("");
    console.log("Options:");
    console.log("  --global             Install into ~/.agents/skills/qmd");
    console.log("  --yes                Also create the .claude/skills/qmd symlink");
    console.log("  -f, --force          Replace existing install or symlink");
    process.exit(0);
  }

  if (!cli.command || cli.values.help) {
    showHelp();
    process.exit(cli.values.help ? 0 : 1);
  }

  switch (cli.command) {
    case "context": {
      const subcommand = cli.args[0];
      if (!subcommand) {
        console.error("Usage: qmd context <add|list|rm>");
        console.error("");
        console.error("Commands:");
        console.error("  qmd context add [path] \"text\"  - Add context (defaults to current dir)");
        console.error("  qmd context add / \"text\"       - Add global context to all collections");
        console.error("  qmd context list                - List all contexts");
        console.error("  qmd context rm <path>           - Remove context");
        process.exit(1);
      }

      switch (subcommand) {
        case "add": {
          if (cli.args.length < 2) {
            console.error("Usage: qmd context add [path] \"text\"");
            console.error("");
            console.error("Examples:");
            console.error("  qmd context add \"Context for current directory\"");
            console.error("  qmd context add . \"Context for current directory\"");
            console.error("  qmd context add /subfolder \"Context for subfolder\"");
            console.error("  qmd context add / \"Global context for all collections\"");
            console.error("");
            console.error("  Using virtual paths:");
            console.error("  qmd context add qmd://journals/ \"Context for entire journals collection\"");
            console.error("  qmd context add qmd://journals/2024 \"Context for 2024 journals\"");
            process.exit(1);
          }

          let pathArg: string | undefined;
          let contextText: string;

          // Check if first arg looks like a path or if it's the context text
          const firstArg = cli.args[1] || '';
          const secondArg = cli.args[2];

          if (secondArg) {
            // Two args: path + context
            pathArg = firstArg;
            contextText = cli.args.slice(2).join(" ");
          } else {
            // One arg: context only (use current directory)
            pathArg = undefined;
            contextText = firstArg;
          }

          await contextAdd(pathArg, contextText);
          break;
        }

        case "list": {
          contextList();
          break;
        }

        case "rm":
        case "remove": {
          if (cli.args.length < 2 || !cli.args[1]) {
            console.error("Usage: qmd context rm <path>");
            console.error("Examples:");
            console.error("  qmd context rm /");
            console.error("  qmd context rm qmd://journals/2024");
            process.exit(1);
          }
          contextRemove(cli.args[1]);
          break;
        }

        default:
          console.error(`Unknown subcommand: ${subcommand}`);
          console.error("Available: add, list, rm");
          process.exit(1);
      }
      break;
    }

    case "get": {
      if (!cli.args[0]) {
        console.error("Usage: qmd get <filepath>[:from[:count]] [--from <line>] [-l <lines>] [--no-line-numbers] [--full-path]");
        process.exit(1);
      }
      const fromLine = cli.values.from ? parseInt(cli.values.from as string, 10) : undefined;
      const maxLines = cli.values.l ? parseInt(cli.values.l as string, 10) : undefined;
      // Line numbers default ON for get; opt out with --no-line-numbers.
      const getLineNumbers = !cli.values["no-line-numbers"];
      getDocument(cli.args[0], fromLine, maxLines, getLineNumbers, !!cli.values["full-path"]);
      break;
    }

    case "multi-get": {
      if (!cli.args[0]) {
        console.error("Usage: qmd multi-get <pattern> [-l <lines>] [--max-bytes <bytes>] [--no-line-numbers] [--full-path] [--format json|csv|md|xml|files]");
        console.error("  pattern: glob (e.g., 'journals/2025-05*.md') or comma-separated list");
        process.exit(1);
      }
      const maxLinesMulti = cli.values.l ? parseInt(cli.values.l as string, 10) : undefined;
      const maxBytes = cli.values["max-bytes"] ? parseInt(cli.values["max-bytes"] as string, 10) : DEFAULT_MULTI_GET_MAX_BYTES;
      // Line numbers default ON for multi-get; opt out with --no-line-numbers.
      const mgLineNumbers = !cli.values["no-line-numbers"];
      multiGet(cli.args[0], maxLinesMulti, maxBytes, cli.opts.format, mgLineNumbers, !!cli.values["full-path"]);
      break;
    }

    case "ls": {
      listFiles(cli.args[0]);
      break;
    }

    case "collection": {
      const subcommand = cli.args[0];
      switch (subcommand) {
        case "list": {
          collectionList();
          break;
        }

        case "add": {
          const pwd = cli.args[1] || getPwd();
          const resolvedPwd = pwd === '.' ? getPwd() : getRealPath(resolve(pwd));
          const globPattern = cli.values.mask as string || DEFAULT_GLOB;
          const name = cli.values.name as string | undefined;

          await collectionAdd(resolvedPwd, globPattern, name);
          break;
        }

        case "remove":
        case "rm": {
          if (!cli.args[1]) {
            console.error("Usage: qmd collection remove <name>");
            console.error("  Use 'qmd collection list' to see available collections");
            process.exit(1);
          }
          collectionRemove(cli.args[1]);
          break;
        }

        case "rename":
        case "mv": {
          if (!cli.args[1] || !cli.args[2]) {
            console.error("Usage: qmd collection rename <old-name> <new-name>");
            console.error("  Use 'qmd collection list' to see available collections");
            process.exit(1);
          }
          collectionRename(cli.args[1], cli.args[2]);
          break;
        }

        case "set-update":
        case "update-cmd": {
          const name = cli.args[1];
          const cmd = cli.args.slice(2).join(' ') || null;
          if (!name) {
            console.error("Usage: qmd collection update-cmd <name> [command]");
            console.error("  Set the command to run before indexing (e.g., 'git pull')");
            console.error("  Omit command to clear it");
            process.exit(1);
          }
          const { updateCollectionSettings, getCollection } = await import("../collections.js");
          const col = getCollection(name);
          if (!col) {
            console.error(`Collection not found: ${name}`);
            process.exit(1);
          }
          updateCollectionSettings(name, { update: cmd });
          if (cmd) {
            console.log(`✓ Set update command for '${name}': ${cmd}`);
          } else {
            console.log(`✓ Cleared update command for '${name}'`);
          }
          break;
        }

        case "include":
        case "exclude": {
          const name = cli.args[1];
          if (!name) {
            console.error(`Usage: qmd collection ${subcommand} <name>`);
            console.error(`  ${subcommand === 'include' ? 'Include' : 'Exclude'} collection in default queries`);
            process.exit(1);
          }
          const { updateCollectionSettings, getCollection } = await import("../collections.js");
          const col = getCollection(name);
          if (!col) {
            console.error(`Collection not found: ${name}`);
            process.exit(1);
          }
          const include = subcommand === 'include';
          updateCollectionSettings(name, { includeByDefault: include });
          console.log(`✓ Collection '${name}' ${include ? 'included in' : 'excluded from'} default queries`);
          break;
        }

        case "show":
        case "info": {
          const name = cli.args[1];
          if (!name) {
            console.error("Usage: qmd collection show <name>");
            process.exit(1);
          }
          const { getCollection } = await import("../collections.js");
          const col = getCollection(name);
          if (!col) {
            console.error(`Collection not found: ${name}`);
            process.exit(1);
          }
          console.log(`Collection: ${name}`);
          console.log(`  Path:     ${col.path}`);
          console.log(`  Pattern:  ${col.pattern}`);
          console.log(`  Include:  ${col.includeByDefault !== false ? 'yes (default)' : 'no'}`);
          if (col.update) {
            console.log(`  Update:   ${col.update}`);
          }
          if (col.context) {
            const ctxCount = Object.keys(col.context).length;
            console.log(`  Contexts: ${ctxCount}`);
          }
          break;
        }

        case "help":
        case undefined: {
          console.log("Usage: qmd collection <command> [options]");
          console.log("");
          console.log("Commands:");
          console.log("  list                      List all collections");
          console.log("  add <path> [--name NAME]  Add a collection");
          console.log("  remove <name>             Remove a collection");
          console.log("  rename <old> <new>        Rename a collection");
          console.log("  show <name>               Show collection details");
          console.log("  update-cmd <name> [cmd]   Set pre-update command (e.g., 'git pull')");
          console.log("  include <name>            Include in default queries");
          console.log("  exclude <name>            Exclude from default queries");
          console.log("");
          console.log("Examples:");
          console.log("  qmd collection add ~/notes --name notes");
          console.log("  qmd collection update-cmd brain 'git pull'");
          console.log("  qmd collection exclude archive");
          process.exit(0);
        }

        default:
          console.error(`Unknown subcommand: ${subcommand}`);
          console.error("Run 'qmd collection help' for usage");
          printDoctorHint();
          process.exit(1);
      }
      break;
    }

    case "init":
      try {
        initLocalIndex();
      } catch (error) {
        exitWithError(error);
      }
      break;

    case "status":
      await showStatus();
      break;

    case "doctor":
      await showDoctor();
      break;

    case "update":
      await updateCollections();
      break;

    case "embed":
      try {
        const maxDocsPerBatch = parseEmbedBatchOption("maxDocsPerBatch", cli.values["max-docs-per-batch"]);
        const maxBatchMb = parseEmbedBatchOption("maxBatchBytes", cli.values["max-batch-mb"]);
        const embedChunkStrategy = parseChunkStrategy(cli.values["chunk-strategy"]);
        const embedMaxDurationMs = parseEmbedTimeoutOption(cli.values["timeout"]);
        // Validate -c against configured collections before dispatching, so a
        // typo errors with "Collection not found: X" instead of silently
        // reporting success because no pending docs match a nonexistent name.
        // embed operates on a single collection; only the first value is used.
        const embedValidatedCollections = resolveCollectionFilter(cli.opts.collection, false);
        const embedCollection = embedValidatedCollections[0];
        await vectorIndex(resolveEmbedModelForCli(), !!cli.values.force, {
          maxDocsPerBatch,
          maxBatchBytes: maxBatchMb === undefined ? undefined : maxBatchMb * 1024 * 1024,
          chunkStrategy: embedChunkStrategy,
          collection: embedCollection,
          maxDurationMs: embedMaxDurationMs,
        });
      } catch (error) {
        exitWithError(error);
      }
      break;

    case "pull": {
      const refresh = cli.values.refresh === undefined ? false : Boolean(cli.values.refresh);
      const activeModels = resolveModelsForCli();
      const models = [
        activeModels.embed,
        activeModels.generate,
        activeModels.rerank,
      ];
      console.log(`${c.bold}Pulling models${c.reset}`);
      const results = await pullModels(models, {
        refresh,
        cacheDir: DEFAULT_MODEL_CACHE_DIR,
      });
      for (const result of results) {
        const size = formatBytes(result.sizeBytes);
        const note = result.refreshed ? "refreshed" : "cached/checked";
        console.log(`- ${result.model} -> ${result.path} (${size}, ${note})`);
      }
      break;
    }

    case "search":
      if (!cli.query) {
        console.error("Usage: qmd search [options] <query>");
        process.exit(1);
      }
      search(cli.query, cli.opts);
      break;

    case "vsearch":
    case "vector-search": // undocumented alias
      if (!cli.query) {
        console.error("Usage: qmd vsearch [options] <query>");
        process.exit(1);
      }
      // Default min-score for vector search is 0.3
      if (!cli.values["min-score"]) {
        cli.opts.minScore = 0.3;
      }
      await vectorSearch(cli.query, cli.opts);
      break;

    case "query":
    case "deep-search": // undocumented alias
      if (!cli.query) {
        console.error("Usage: qmd query [options] <query>");
        process.exit(1);
      }
      await querySearch(cli.query, cli.opts);
      break;

    case "bench": {
      const fixturePath = cli.args[0];
      if (!fixturePath) {
        console.error("Usage: qmd bench <fixture.json> [--json] [-c collection]");
        console.error("       qmd bench <benchmark-dir> --run <name> [--model ID] [--only lex|vec|hyde]");
        console.error("       qmd bench <benchmark-dir> --check-index");
        console.error("");
        console.error("Run legacy fixture or canonical qrels benchmarks.");
        console.error("See src/bench/fixtures/example.json for the fixture format.");
        process.exit(1);
      }
      const runValue = typeof cli.values.run === "string"
        ? cli.values.run
        : undefined;
      if (runValue !== undefined) {
        const { validateBenchmarkRunName } = await import("../bench/run-name.js");
        try {
          validateBenchmarkRunName(runValue);
        } catch (error) {
          console.error(error instanceof Error ? error.message : String(error));
          process.exit(1);
        }
      }
      const onlyValue = cli.values.only;
      if (
        onlyValue !== undefined
        && onlyValue !== "lex"
        && onlyValue !== "vec"
        && onlyValue !== "hyde"
      ) {
        console.error("--only must be lex, vec, or hyde");
        process.exit(1);
      }
      const modelValue = typeof cli.values.model === "string"
        ? cli.values.model
        : undefined;
      const { runBenchmarkCommand, writeBenchmarkIndexManifest } = await import("../bench/bench.js");
      const benchCollection = cli.opts.collection;
      const common = {
        dbPath: getDbPath(),
        configPath: configExists() ? getConfigPath() : undefined,
      };
      if (cli.values["check-index"]) {
        if (runValue || modelValue || onlyValue) {
          console.error("--check-index cannot be combined with --run, --model, or --only");
          process.exit(1);
        }
        const manifest = await writeBenchmarkIndexManifest(fixturePath, common);
        console.log(JSON.stringify(manifest, null, 2));
      } else {
        await runBenchmarkCommand(fixturePath, {
          json: !!cli.values.json,
          collection: Array.isArray(benchCollection) ? benchCollection[0] : benchCollection,
          ...common,
          run: runValue,
          model: modelValue,
          only: onlyValue,
        });
      }
      break;
    }

    case "mcp": {
      const sub = cli.args[0]; // stop | status | undefined

      // Cache dir for PID/log files — same dir as the index
      const cacheDir = process.env.XDG_CACHE_HOME
        ? resolve(process.env.XDG_CACHE_HOME, "qmd")
        : resolve(homedir(), ".cache", "qmd");
      const pidPath = resolve(cacheDir, "mcp.pid");

      // Subcommands take priority over flags
      if (sub === "stop") {
        if (!existsSync(pidPath)) {
          console.log("Not running (no PID file).");
          process.exit(0);
        }
        const pid = parseInt(readFileSync(pidPath, "utf-8").trim());
        try {
          process.kill(pid, 0); // alive?
          process.kill(pid, "SIGTERM");
          unlinkSync(pidPath);
          console.log(`Stopped QMD MCP server (PID ${pid}).`);
        } catch {
          unlinkSync(pidPath);
          console.log("Cleaned up stale PID file (server was not running).");
        }
        process.exit(0);
      }

      if (cli.values.http) {
        const port = Number(cli.values.port) || 8181;
        // --host overrides the default localhost bind; QMD_HOST env is the
        // fallback (resolved in startMcpHttpServer). Use "0.0.0.0" to accept
        // off-host connections, e.g. a container liveness probe.
        const host = cli.values.host ? String(cli.values.host) : undefined;

        if (cli.values.daemon) {
          // Guard: check if already running
          if (existsSync(pidPath)) {
            const existingPid = parseInt(readFileSync(pidPath, "utf-8").trim());
            try {
              process.kill(existingPid, 0); // alive?
              console.error(`Already running (PID ${existingPid}). Run 'qmd mcp stop' first.`);
              process.exit(1);
            } catch {
              // Stale PID file — continue
            }
          }

          mkdirSync(cacheDir, { recursive: true });
          const logPath = resolve(cacheDir, "mcp.log");
          const logFd = openSync(logPath, "w"); // truncate — fresh log per daemon run
          const selfPath = fileURLToPath(import.meta.url);
          const indexArgs = cli.values.index ? ["--index", String(cli.values.index)] : [];
          const hostArgs = host ? ["--host", host] : [];
          const spawnArgs = selfPath.endsWith(".ts")
            ? ["--import", pathJoin(dirname(selfPath), "..", "..", "node_modules", "tsx", "dist", "esm", "index.mjs"), selfPath, ...indexArgs, "mcp", "--http", "--port", String(port), ...hostArgs]
            : [selfPath, ...indexArgs, "mcp", "--http", "--port", String(port), ...hostArgs];
          const child = nodeSpawn(process.execPath, spawnArgs, {
            stdio: ["ignore", logFd, logFd],
            detached: true,
          });
          child.unref();
          closeSync(logFd); // parent's copy; child inherited the fd

          writeFileSync(pidPath, String(child.pid));
          console.log(`Started on http://${host ?? "localhost"}:${port}/mcp (PID ${child.pid})`);
          console.log(`Logs: ${logPath}`);
          process.exit(0);
        }

        // Foreground HTTP mode — remove top-level cursor handlers so the
        // async cleanup handlers in startMcpHttpServer actually run.
        process.removeAllListeners("SIGTERM");
        process.removeAllListeners("SIGINT");
        const { startMcpHttpServer } = await import("../mcp/server.js");
        try {
          await startMcpHttpServer(port, { dbPath: getDbPath(), host });
        } catch (e: unknown) {
          if (typeof e === "object" && e !== null && "code" in e && e.code === "EADDRINUSE") {
            console.error(`Port ${port} already in use. Try a different port with --port.`);
            process.exit(1);
          }
          throw e;
        }
      } else {
        // Default: stdio transport
        const { startMcpServer } = await import("../mcp/server.js");
        await startMcpServer({ dbPath: getDbPath() });
      }
      break;
    }

    case "skills": {
      try {
        if (cli.values.help || cli.args[0] === "help") {
          showSkillsHelp();
        } else {
          runSkillsCommand(cli.args, Boolean(cli.values.json), Boolean(cli.values.full), Boolean(cli.values.all));
        }
      } catch (error) {
        if (cli.values.json) {
          outputSkillsJson({ success: false, error: error instanceof Error ? error.message : String(error) });
        } else {
          console.error(error instanceof Error ? error.message : String(error));
        }
        process.exit(1);
      }
      break;
    }

    case "skill": {
      const subcommand = cli.args[0];
      switch (subcommand) {
        case "show": {
          showSkill();
          break;
        }

        case "install": {
          try {
            await installSkill(Boolean(cli.values.global), Boolean(cli.values.force), Boolean(cli.values.yes));
          } catch (error) {
            exitWithError(error);
          }
          break;
        }

        case "help":
        case undefined: {
          console.log("Usage: qmd skill <show|install> [options]");
          console.log("");
          console.log("Commands:");
          console.log("  show                 Print the QMD skill");
          console.log("  install              Install QMD skill into ./.agents/skills/qmd");
          console.log("");
          console.log("Options:");
          console.log("  --global             Install into ~/.agents/skills/qmd");
          console.log("  --yes                Also create the .claude/skills/qmd symlink");
          console.log("  -f, --force          Replace existing install or symlink");
          process.exit(0);
        }

        default:
          console.error(`Unknown subcommand: ${subcommand}`);
          console.error("Run 'qmd skill help' for usage");
          printDoctorHint();
          process.exit(1);
      }
      break;
    }

    case "cleanup": {
      const db = getDb();

      // 1. Clear llm_cache
      const cacheCount = deleteLLMCache(db);
      console.log(`${c.green}✓${c.reset} Cleared ${cacheCount} cached API responses`);

      // 2. Remove orphaned vectors
      const orphanedVecs = cleanupOrphanedVectors(db);
      if (orphanedVecs > 0) {
        console.log(`${c.green}✓${c.reset} Removed ${orphanedVecs} orphaned embedding chunks`);
      } else {
        console.log(`${c.dim}No orphaned embeddings to remove${c.reset}`);
      }

      // 3. Remove inactive documents
      const inactiveDocs = deleteInactiveDocuments(db);
      if (inactiveDocs > 0) {
        console.log(`${c.green}✓${c.reset} Removed ${inactiveDocs} inactive document records`);
      }

      // 4. Vacuum to reclaim space
      vacuumDatabase(db);
      console.log(`${c.green}✓${c.reset} Database vacuumed`);

      closeDb();
      break;
    }

    default:
      console.error(`Unknown command: ${cli.command}`);
      console.error("Run 'qmd --help' for usage.");
      printDoctorHint();
      process.exit(1);
  }

  if (cli.command !== "mcp") {
    await finishSuccessfulCliCommand({
      command: cli.command,
      format: cli.opts.format,
    });
  }

} // end if (main module)
