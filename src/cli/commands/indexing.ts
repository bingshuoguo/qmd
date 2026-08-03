/**
 * commands/indexing.ts - index / embed / update commands: filesystem
 * re-indexing (indexFiles, updateCollections) and vector embedding
 * (vectorIndex) with their progress rendering. Pulls formatCount/
 * shortModelName from doctor.ts (one-way edge: doctor never imports this
 * module).
 */
import fastGlob from "fast-glob";
import { spawn as nodeSpawn } from "child_process";
import { readFileSync, statSync } from "fs";
import {
  getPwd,
  getRealPath,
  resolve,
  getHashesNeedingEmbedding,
  clearCache,
  hashContent,
  extractTitle,
  listCollections,
  reindexCollection,
  findOrMigrateLegacyDocument,
  updateDocumentTitle,
  insertContent,
  insertDocument,
  updateDocument,
  getActiveDocumentPaths,
  deactivateDocument,
  cleanupOrphanedContent,
  generateEmbeddings,
  DEFAULT_GLOB,
  DEFAULT_EMBED_MAX_BATCH_BYTES,
  DEFAULT_EMBED_MAX_DOCS_PER_BATCH,
  type ChunkStrategy,
} from "../../store.js";
import {
  getCollection as getCollectionFromYaml,
} from "../../collections.js";
import {
  getStore,
  getDb,
  closeDb,
  resolveEmbedModelForCli,
} from "../context.js";
import {
  c,
  cursor,
  isTTY,
  progress,
  formatETA,
  formatBytes,
  renderProgressBar,
} from "../term.js";
import { formatCount, shortModelName } from "./doctor.js";

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

export {
  updateCollections,
  indexFiles,
  vectorIndex,
  parseEmbedBatchOption,
  parseChunkStrategy,
  parseEmbedTimeoutOption,
};
