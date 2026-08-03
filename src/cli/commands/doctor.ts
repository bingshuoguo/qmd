/**
 * commands/doctor.ts - `qmd doctor` and `qmd status`: index health, model
 * defaults/cache, device probing, embedding freshness/fingerprint/vector
 * sampling, and environment override reporting. Depends on context.ts for
 * store lifecycle; the two command entry points (showStatus/showDoctor) are
 * the only exports — every check/device helper stays module-private.
 */
import { readFileSync, readdirSync, statSync, existsSync, unlinkSync } from "fs";
import { join as pathJoin } from "path";
import { isBun } from "../../db.js";
import type { Database } from "../../db.js";
import {
  homedir,
  resolve,
  listCollections,
  getHashesNeedingEmbedding,
  countActiveDocuments,
  countContentVectors,
  getLatestDocumentModifiedAt,
  getEmbeddingFingerprint,
  formatDocForEmbedding,
  extractTitle,
  chunkDocumentByTokens,
  hasVectorTable,
  sampleEmbeddedChunks,
  getStoredEmbedding,
  getSqliteVersion,
  getVecVersion,
  getEmbeddingFingerprintGroups,
  maybeAdoptLegacyEmbeddingFingerprint,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
} from "../../store.js";
import { withLLMSession, getDefaultLlamaCpp, DEFAULT_MODEL_CACHE_DIR, inspectGgufFile, isDarwinMetalMitigationActive } from "../../llm.js";
import {
  getCollection as getCollectionFromYaml,
  listAllContexts,
  loadConfig,
  getConfigPath,
  type CollectionConfig,
  type ModelsConfig,
} from "../../collections.js";
import {
  getStore,
  getDb,
  getDbPath,
  closeDb,
  resolveEmbedModelForCli,
  resolveModelsForCli,
} from "../context.js";
import { c, formatBytes, formatTimeAgo } from "../term.js";
import { readPackageJson } from "../help.js";

function isForceCpuEnabled(): boolean {
  const value = process.env.QMD_FORCE_CPU;
  return !!value && !["false", "off", "none", "disable", "disabled", "0"].includes(value.trim().toLowerCase());
}

function configuredGpuModeLabel(): string {
  return isForceCpuEnabled()
    ? "CPU forced (QMD_FORCE_CPU)"
    : (process.env.QMD_LLAMA_GPU?.trim() || "auto");
}

function summarizeDeviceNames(names: string[]): string {
  const counts = new Map<string, number>();
  for (const name of names) {
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([name, count]) => count > 1 ? `${count}× ${name}` : name)
    .join(", ");
}

function sanitizeDiagnosticMessage(message: string): string {
  const home = homedir();
  return message
    .replaceAll(home, "~")
    .replaceAll(process.cwd(), ".")
    .split("\n")
    .map(line => line.trim())
    .filter(Boolean)
    .slice(0, 3)
    .join("; ");
}

async function showStatus(): Promise<void> {
  const dbPath = getDbPath();
  const db = getDb();

  // Collections are defined in YAML; no duplicate cleanup needed.
  // Collections are defined in YAML; no duplicate cleanup needed.

  // Index size
  let indexSize = 0;
  try {
    const stat = statSync(dbPath).size;
    indexSize = stat;
  } catch { }

  // Collections info (from YAML + database stats)
  const collections = listCollections(db);

  // Overall stats
  const totalDocs = countActiveDocuments(db);
  const vectorCount = countContentVectors(db);
  const statusEmbedModel = resolveEmbedModelForCli();
  const needsEmbedding = getHashesNeedingEmbedding(db, undefined, statusEmbedModel);

  // Most recent update across all collections
  const mostRecent = getLatestDocumentModifiedAt(db);

  console.log(`${c.bold}QMD Status${c.reset}\n`);
  console.log(`Index: ${dbPath}`);
  console.log(`Size:  ${formatBytes(indexSize)}`);

  // MCP daemon status (check PID file liveness)
  const mcpCacheDir = process.env.XDG_CACHE_HOME
    ? resolve(process.env.XDG_CACHE_HOME, "qmd")
    : resolve(homedir(), ".cache", "qmd");
  const mcpPidPath = resolve(mcpCacheDir, "mcp.pid");
  if (existsSync(mcpPidPath)) {
    const mcpPid = parseInt(readFileSync(mcpPidPath, "utf-8").trim());
    try {
      process.kill(mcpPid, 0);
      console.log(`MCP:   ${c.green}running${c.reset} (PID ${mcpPid})`);
    } catch {
      unlinkSync(mcpPidPath);
      // Stale PID file cleaned up silently
    }
  }
  console.log("");

  console.log(`${c.bold}Documents${c.reset}`);
  console.log(`  Total:    ${totalDocs} files indexed`);
  console.log(`  Vectors:  ${vectorCount} embedded`);
  if (needsEmbedding > 0) {
    console.log(`  ${c.yellow}Pending:  ${needsEmbedding} need embedding${c.reset} (run 'qmd embed')`);
  }
  if (mostRecent) {
    const lastUpdate = new Date(mostRecent);
    console.log(`  Updated:  ${formatTimeAgo(lastUpdate)}`);
  }

  // Get all contexts grouped by collection (from YAML)
  const allContexts = listAllContexts();
  const contextsByCollection = new Map<string, { path_prefix: string; context: string }[]>();

  for (const ctx of allContexts) {
    // Group contexts by collection name
    if (!contextsByCollection.has(ctx.collection)) {
      contextsByCollection.set(ctx.collection, []);
    }
    contextsByCollection.get(ctx.collection)!.push({
      path_prefix: ctx.path,
      context: ctx.context
    });
  }

  // AST chunking status
  try {
    const { getASTStatus } = await import("../../ast.js");
    const ast = await getASTStatus();
    console.log(`\n${c.bold}AST Chunking${c.reset}`);
    if (ast.available) {
      const ok = ast.languages.filter(l => l.available).map(l => l.language);
      const fail = ast.languages.filter(l => !l.available);
      console.log(`  Status:   ${c.green}active${c.reset}`);
      console.log(`  Languages: ${ok.join(", ")}`);
      if (fail.length > 0) {
        for (const f of fail) {
          console.log(`  ${c.yellow}Unavailable: ${f.language} (${f.error})${c.reset}`);
        }
      }
    } else {
      console.log(`  Status:   ${c.yellow}unavailable${c.reset} (falling back to regex chunking)`);
      for (const l of ast.languages) {
        if (l.error) console.log(`  ${c.dim}${l.language}: ${l.error}${c.reset}`);
      }
    }
  } catch {
    console.log(`\n${c.bold}AST Chunking${c.reset}`);
    console.log(`  Status:   ${c.dim}not available${c.reset}`);
  }

  if (collections.length > 0) {
    console.log(`\n${c.bold}Collections${c.reset}`);
    for (const col of collections) {
      const lastMod = col.last_modified ? formatTimeAgo(new Date(col.last_modified)) : "never";
      const contexts = contextsByCollection.get(col.name) || [];

      console.log(`  ${c.cyan}${col.name}${c.reset} ${c.dim}(qmd://${col.name}/)${c.reset}`);
      console.log(`    ${c.dim}Pattern:${c.reset}  ${col.glob_pattern}`);
      console.log(`    ${c.dim}Files:${c.reset}    ${col.active_count} (updated ${lastMod})`);

      if (contexts.length > 0) {
        console.log(`    ${c.dim}Contexts:${c.reset} ${contexts.length}`);
        for (const ctx of contexts) {
          // Handle both empty string and '/' as root context
          const pathDisplay = (ctx.path_prefix === '' || ctx.path_prefix === '/') ? '/' : `/${ctx.path_prefix}`;
          const contextPreview = ctx.context.length > 60
            ? ctx.context.substring(0, 57) + '...'
            : ctx.context;
          console.log(`      ${c.dim}${pathDisplay}:${c.reset} ${contextPreview}`);
        }
      }
    }

    // Show examples of virtual paths
    console.log(`\n${c.bold}Examples${c.reset}`);
    console.log(`  ${c.dim}# List files in a collection${c.reset}`);
    if (collections.length > 0 && collections[0]) {
      console.log(`  qmd ls ${collections[0].name}`);
    }
    console.log(`  ${c.dim}# Get a document${c.reset}`);
    if (collections.length > 0 && collections[0]) {
      console.log(`  qmd get qmd://${collections[0].name}/path/to/file.md`);
    }
    console.log(`  ${c.dim}# Search within a collection${c.reset}`);
    if (collections.length > 0 && collections[0]) {
      console.log(`  qmd search "query" -c ${collections[0].name}`);
    }
  } else {
    console.log(`\n${c.dim}No collections. Run 'qmd collection add .' to index markdown files.${c.reset}`);
  }

  // Models
  {
    // hf:org/repo/file.gguf → https://huggingface.co/org/repo
    const hfLink = (uri: string) => {
      const match = uri.match(/^hf:([^/]+\/[^/]+)\//);
      return match ? `https://huggingface.co/${match[1]}` : uri;
    };
    const activeModels = resolveModelsForCli();
    console.log(`\n${c.bold}Models${c.reset}`);
    console.log(`  Embedding:   ${hfLink(activeModels.embed)}`);
    console.log(`  Reranking:   ${hfLink(activeModels.rerank)}`);
    console.log(`  Generation:  ${hfLink(activeModels.generate)}`);
  }


  // Tips section
  const tips: string[] = [];

  // Check for collections without context
  const collectionsWithoutContext = collections.filter(col => {
    const contexts = contextsByCollection.get(col.name) || [];
    return contexts.length === 0;
  });
  if (collectionsWithoutContext.length > 0) {
    const names = collectionsWithoutContext.map(c => c.name).slice(0, 3).join(', ');
    const more = collectionsWithoutContext.length > 3 ? ` +${collectionsWithoutContext.length - 3} more` : '';
    tips.push(`Add context to collections for better search results: ${names}${more}`);
    tips.push(`  ${c.dim}qmd context add qmd://<name>/ "What this collection contains"${c.reset}`);
    tips.push(`  ${c.dim}qmd context add qmd://<name>/meeting-notes "Weekly team meeting notes"${c.reset}`);
  }

  // Check for collections without update commands
  const collectionsWithoutUpdate = collections.filter(col => {
    const yamlCol = getCollectionFromYaml(col.name);
    return !yamlCol?.update;
  });
  if (collectionsWithoutUpdate.length > 0 && collections.length > 1) {
    const names = collectionsWithoutUpdate.map(c => c.name).slice(0, 3).join(', ');
    const more = collectionsWithoutUpdate.length > 3 ? ` +${collectionsWithoutUpdate.length - 3} more` : '';
    tips.push(`Add update commands to keep collections fresh: ${names}${more}`);
    tips.push(`  ${c.dim}qmd collection update-cmd <name> 'git stash && git pull --rebase --ff-only && git stash pop'${c.reset}`);
  }

  if (tips.length > 0) {
    console.log(`\n${c.bold}Tips${c.reset}`);
    for (const tip of tips) {
      console.log(`  ${tip}`);
    }
  }

  closeDb();
}

function doctorCheck(label: string, ok: boolean, details: string): void {
  const mark = ok ? `${c.green}✓${c.reset}` : `${c.yellow}⚠${c.reset}`;
  console.log(`${mark} ${label}: ${details}`);
}

function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}

function shortModelName(model: string): string {
  if (model.startsWith("hf:")) {
    return model.split("/").pop() || model;
  }
  return model.length > 56 ? `${model.slice(0, 53)}...` : model;
}

function normalizedDoctorNextSteps(steps: string[]): string[] {
  const unique = Array.from(new Set(steps));
  const hasForceEmbed = unique.some(step => step.includes("qmd embed --force"));
  if (!hasForceEmbed) return unique;
  return unique.filter(step => !step.includes("qmd embed") || step.startsWith("Run `qmd embed --force`"));
}

function shortHashSeq(hashSeq: string): string {
  const idx = hashSeq.lastIndexOf("_");
  if (idx < 0) return hashSeq.length > 18 ? `${hashSeq.slice(0, 18)}...` : hashSeq;
  return `${hashSeq.slice(0, 12)}_${hashSeq.slice(idx + 1)}`;
}

type DoctorVectorSampleResult = {
  ok: boolean;
  details: string;
};

function decodeStoredEmbedding(bytes: Uint8Array): Float32Array {
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

function cosineDistance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length || a.length === 0) return Number.POSITIVE_INFINITY;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) return Number.POSITIVE_INFINITY;
  return 1 - (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
}

type CachedModelInspection = {
  path: string | null;
  invalid: string[];
};

function formatModelDiagnosticPath(path: string): string {
  return sanitizeDiagnosticMessage(path);
}

function findCachedModelInspection(model: string): CachedModelInspection {
  const invalid: string[] = [];
  if (model.startsWith("hf:")) {
    const filename = model.split("/").pop();
    if (!filename || !existsSync(DEFAULT_MODEL_CACHE_DIR)) return { path: null, invalid };
    const entries = readdirSync(DEFAULT_MODEL_CACHE_DIR, { withFileTypes: true });
    for (const entry of entries) {
      // Skip the `<filename>.etag` HTTP sidecar that `qmd pull` writes next to
      // each blob. It satisfies `includes(filename)` but is not a GGUF, so
      // inspecting it as one surfaces a spurious "invalid" model in `qmd
      // doctor` whenever readdir happens to yield the sidecar before the blob.
      if (!entry.isFile() || entry.name.endsWith(".etag") || !entry.name.includes(filename)) continue;
      const candidate = pathJoin(DEFAULT_MODEL_CACHE_DIR, entry.name);
      const inspection = inspectGgufFile(candidate);
      if (inspection.valid) return { path: candidate, invalid };
      invalid.push(`${formatModelDiagnosticPath(candidate)}: ${inspection.details}`);
    }
    return { path: null, invalid };
  }

  const inspection = inspectGgufFile(model);
  if (inspection.valid) return { path: model, invalid };
  if (inspection.exists) invalid.push(`${formatModelDiagnosticPath(model)}: ${inspection.details}`);
  return { path: null, invalid };
}

type EnvOverride = {
  name: string;
  value: string;
  consequence: string;
};

function envValueForDisplay(value: string): string {
  const sanitized = sanitizeDiagnosticMessage(value);
  return sanitized.length > 96 ? `${sanitized.slice(0, 93)}...` : sanitized;
}

function collectEnvironmentOverrides(activeModels: { embed: string; generate: string; rerank: string }, configModels: ModelsConfig = {}): EnvOverride[] {
  const overrides: EnvOverride[] = [];
  const add = (name: string, consequence: string) => {
    const raw = process.env[name]?.trim();
    if (!raw) return;
    overrides.push({ name, value: envValueForDisplay(raw), consequence });
  };
  const addModel = (name: string, key: "embed" | "generate" | "rerank", active: string) => {
    const raw = process.env[name]?.trim();
    if (!raw) return;
    const configured = configModels[key];
    const consequence = configured && configured !== raw
      ? `set but ignored because index models.${key} is configured as ${configured}`
      : `sets the active ${key} model to ${active}; changes embedding/search semantics and may require \`qmd pull\` plus \`qmd embed\``;
    overrides.push({ name, value: envValueForDisplay(raw), consequence });
  };

  add("INDEX_PATH", "overrides the SQLite index path; QMD reads/writes a different database");
  add("QMD_CONFIG_DIR", "overrides the QMD config directory and takes precedence over XDG_CONFIG_HOME");
  add("XDG_CONFIG_HOME", "moves QMD config to $XDG_CONFIG_HOME/qmd when QMD_CONFIG_DIR is not set");
  add("XDG_CACHE_HOME", "moves the default index cache, model cache, and MCP daemon PID files");
  addModel("QMD_EMBED_MODEL", "embed", activeModels.embed);
  addModel("QMD_GENERATE_MODEL", "generate", activeModels.generate);
  addModel("QMD_RERANK_MODEL", "rerank", activeModels.rerank);
  add("QMD_FORCE_CPU", "forces llama.cpp to bypass GPU backends; embeddings/query will be slower but GPU crashes are avoided");
  add("QMD_LLAMA_GPU", "selects llama.cpp GPU backend (metal/cuda/vulkan) or disables GPU when set to false/off/0");
  add("QMD_DOCTOR_DEVICE_PROBE", "controls qmd doctor native device probing; 0/off skips GPU probing");
  add("QMD_EMBED_PARALLELISM", "overrides embedding parallel context count; too high can exhaust RAM/VRAM");
  add("QMD_EXPAND_CONTEXT_SIZE", "overrides query expansion context size; larger values use more memory");
  add("QMD_RERANK_CONTEXT_SIZE", "overrides reranker context size; larger values use more memory");
  add("QMD_EMBED_CONTEXT_SIZE", "overrides embed context size; larger values use more memory");
  add("QMD_EDITOR_URI", "overrides clickable editor link template in terminal output");
  add("QMD_SKILLS_DIR", "overrides where qmd skills are discovered from");
  add("QMD_METAL_KEEP_RESIDENCY", "opts back into libggml-metal residency sets on darwin; restores ~0ms perf wins for long-lived processes but re-exposes the static-destructor backtrace at process exit (ggml-org/llama.cpp#22593)");
  add("GGML_METAL_NO_RESIDENCY", "set automatically by the launcher on darwin to disable Metal residency sets (avoids ggml-org/llama.cpp#22593); override via QMD_METAL_KEEP_RESIDENCY=1");
  add("NO_COLOR", "disables colored terminal output");
  add("CI", "disables real LLM operations inside QMD's LlamaCpp wrapper");
  add("HF_ENDPOINT", "changes Hugging Face download endpoint used when pulling models");
  add("QMD_WRAPPER_CAPTURE", "test/debug hook for the qmd shell wrapper; should not be set in normal use");
  add("WSL_DISTRO_NAME", "enables WSL path handling heuristics");
  add("WSL_INTEROP", "enables WSL path handling heuristics");
  return overrides;
}

type DoctorConfigCheck = {
  config: CollectionConfig | null;
  valid: boolean;
};

function checkDoctorIndexConfig(nextSteps: string[]): DoctorConfigCheck {
  try {
    const config = loadConfig();
    const collectionCount = Object.keys(config.collections ?? {}).length;
    if (collectionCount === 0) {
      doctorCheck("index config", false, "no collections configured. Next: `qmd collection add .`");
      nextSteps.push("Run `qmd collection add . --name <name>` from the folder you want to index, or edit .qmd/index.yml manually.");
    } else {
      doctorCheck("index config", true, `${formatCount(collectionCount)} ${collectionCount === 1 ? "collection" : "collections"} configured`);
    }
    return { config, valid: true };
  } catch (error) {
    const message = error instanceof Error ? sanitizeDiagnosticMessage(error.message) : sanitizeDiagnosticMessage(String(error));
    const configPath = getConfigPath();
    doctorCheck("index config", false, `invalid index.yml at ${configPath}: ${message}. Next: fix the YAML and rerun \`qmd doctor\``);
    nextSteps.push(`Fix invalid YAML in ${configPath}, then rerun \`qmd doctor\`.`);
    return { config: null, valid: false };
  }
}

function checkEnvironmentOverrides(activeModels: { embed: string; generate: string; rerank: string }, configModels: ModelsConfig = {}): void {
  const overrides = collectEnvironmentOverrides(activeModels, configModels);
  if (overrides.length === 0) {
    doctorCheck("environment overrides", true, "none");
    return;
  }

  doctorCheck("environment overrides", false, `${overrides.length} set`);
  for (const override of overrides) {
    console.log(`  - ${override.name}=${override.value}: ${override.consequence}`);
  }
}

function checkModelDefaults(activeModels: { embed: string; generate: string; rerank: string }, configModels: ModelsConfig = {}): void {
  const checks = [
    { role: "embedding", key: "embed", active: activeModels.embed, configured: configModels.embed, defaultModel: DEFAULT_EMBED_MODEL, envName: "QMD_EMBED_MODEL", envValue: process.env.QMD_EMBED_MODEL },
    { role: "generation", key: "generate", active: activeModels.generate, configured: configModels.generate, defaultModel: DEFAULT_QUERY_MODEL, envName: "QMD_GENERATE_MODEL", envValue: process.env.QMD_GENERATE_MODEL },
    { role: "reranking", key: "rerank", active: activeModels.rerank, configured: configModels.rerank, defaultModel: DEFAULT_RERANK_MODEL, envName: "QMD_RERANK_MODEL", envValue: process.env.QMD_RERANK_MODEL },
  ] as const;

  const notes: string[] = [];
  for (const check of checks) {
    const envValue = check.envValue?.trim();
    if (envValue && check.active === envValue) {
      notes.push(`${check.role}: env ${check.envName}=${check.active} (default ${check.defaultModel}; might be ok)`);
    } else if (check.configured && check.configured !== check.defaultModel) {
      notes.push(`${check.role}: index ${check.configured} (default ${check.defaultModel}; might be ok)`);
    } else if (envValue && check.active !== envValue) {
      notes.push(`${check.role}: ${check.envName} is set to ${envValue} but index config uses ${check.active}`);
    }
  }

  if (notes.length === 0) {
    doctorCheck("model defaults", true, "using QMD codebase defaults");
    return;
  }

  doctorCheck("model defaults", false, `non-default model configuration: ${notes.join("; ")}`);
}

function checkModelCache(activeModels: { embed: string; generate: string; rerank: string }, nextSteps: string[]): void {
  const models = [
    ["embedding", activeModels.embed],
    ["generation", activeModels.generate],
    ["reranking", activeModels.rerank],
  ] as const;
  const unique = new Map<string, string[]>();
  for (const [role, model] of models) {
    unique.set(model, [...(unique.get(model) ?? []), role]);
  }

  const missing: string[] = [];
  const cached: string[] = [];
  const invalid: string[] = [];
  for (const [model, roles] of unique) {
    const label = `${roles.join("+")}: ${model}`;
    const inspection = findCachedModelInspection(model);
    invalid.push(...inspection.invalid.map(detail => `${label} (${detail})`));
    if (inspection.path) {
      cached.push(label);
    } else {
      missing.push(label);
    }
  }

  if (missing.length === 0 && invalid.length === 0) {
    doctorCheck("model cache", true, `${cached.length} active ${cached.length === 1 ? "model is" : "models are"} downloaded and valid GGUF`);
    return;
  }

  const parts: string[] = [];
  if (invalid.length > 0) parts.push(`invalid ${invalid.length}: ${invalid.join("; ")}`);
  if (missing.length > 0) parts.push(`missing ${missing.length}/${unique.size}: ${missing.join("; ")}`);
  const next = invalid.length > 0
    ? "Next: run `qmd pull --refresh` (or remove the bad cached file)"
    : "Next: run `qmd pull`";
  doctorCheck("model cache", false, `${parts.join("; ")}. ${next}`);
  if (invalid.length > 0) {
    nextSteps.push("Run `qmd pull --refresh` to replace invalid cached model files, or delete the listed file and rerun `qmd pull`.");
  } else {
    nextSteps.push("Run `qmd pull` to download missing embedding/generation/reranking models before `qmd embed` or `qmd query`.");
  }
}

async function checkEmbeddingVectorSamples(db: Database, model: string, fingerprint: string, sampleSize: number = 3): Promise<DoctorVectorSampleResult> {
  const activeDocs = countActiveDocuments(db);
  if (activeDocs === 0) {
    return { ok: true, details: "no active documents indexed" };
  }

  if (!hasVectorTable(db)) {
    return { ok: false, details: "no vector table to test; please run qmd embed again" };
  }

  const samples = sampleEmbeddedChunks(db, model, fingerprint, sampleSize);

  if (samples.length === 0) {
    return { ok: false, details: "no current embedded chunks to test; please run qmd embed again" };
  }

  const threshold = 0.0001;
  const mismatches: string[] = [];

  await withLLMSession(async (session) => {
    for (const sample of samples) {
      const hashSeq = `${sample.hash}_${sample.seq}`;
      const chunks = await chunkDocumentByTokens(sample.body, undefined, undefined, undefined, sample.path, undefined, session.signal);
      const chunk = chunks[sample.seq];
      if (!chunk) {
        mismatches.push(`${shortHashSeq(hashSeq)}: chunk no longer exists`);
        continue;
      }

      const title = extractTitle(sample.body, sample.path);
      const result = await session.embed(formatDocForEmbedding(chunk.text, title, model), { model });
      if (!result) {
        mismatches.push(`${shortHashSeq(hashSeq)}: embedding failed`);
        continue;
      }

      const storedEmbedding = getStoredEmbedding(db, hashSeq);
      if (!storedEmbedding) {
        mismatches.push(`${shortHashSeq(hashSeq)}: stored vector missing`);
        continue;
      }

      const distance = cosineDistance(result.embedding, decodeStoredEmbedding(storedEmbedding));
      if (distance > threshold) {
        mismatches.push(`${shortHashSeq(hashSeq)}: stored vector distance ${distance.toFixed(6)}`);
      }
    }
  }, { maxDuration: 10 * 60 * 1000, name: "doctorEmbeddingVectorSample" });

  if (mismatches.length > 0) {
    return {
      ok: false,
      details: `${mismatches.length}/${samples.length} sampled chunks differ from stored vectors (${mismatches[0]}). Rebuild with \`qmd embed --force\``,
    };
  }

  return {
    ok: true,
    details: `${samples.length} sampled ${samples.length === 1 ? "chunk" : "chunks"} reproduce stored vectors`,
  };
}

function hasLibraryInDirs(libraryBaseName: string, dirs: string[]): boolean {
  for (const dir of dirs) {
    if (!dir || !existsSync(dir)) continue;
    try {
      for (const entry of readdirSync(dir)) {
        if (entry === libraryBaseName || entry.startsWith(`${libraryBaseName}.`)) return true;
      }
    } catch { /* ignore unreadable system library dirs */ }
  }
  return false;
}

function linuxCudaRuntimeDiagnostic(): string | null {
  if (process.platform !== "linux") return null;

  const dirs = new Set<string>();
  for (const value of [process.env.LD_LIBRARY_PATH, process.env.CUDA_PATH]) {
    for (const part of (value ?? "").split(":")) {
      if (part) dirs.add(part);
    }
  }
  if (process.env.CUDA_PATH) {
    dirs.add(pathJoin(process.env.CUDA_PATH, "lib64"));
    dirs.add(pathJoin(process.env.CUDA_PATH, "targets", "x86_64-linux", "lib"));
  }
  for (const dir of ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"]) {
    dirs.add(dir);
  }
  try {
    for (const entry of readdirSync("/usr/local")) {
      if (!entry.toLowerCase().startsWith("cuda-")) continue;
      const cudaRoot = pathJoin("/usr/local", entry);
      dirs.add(pathJoin(cudaRoot, "lib64"));
      dirs.add(pathJoin(cudaRoot, "targets", "x86_64-linux", "lib"));
    }
  } catch { /* /usr/local may not be readable in restricted environments */ }

  const searchDirs = [...dirs];
  const hasDriver = hasLibraryInDirs("libcuda.so", searchDirs) || hasLibraryInDirs("libnvidia-ml.so", searchDirs);
  if (!hasDriver) return null;

  const cudaLibraries: [library: string, label: string][] = [
    ["libcudart.so", "CUDA runtime"],
    ["libcublas.so", "cuBLAS"],
    ["libcublasLt.so", "cuBLASLt"],
  ];
  const missing = cudaLibraries
    .filter(([library]) => !hasLibraryInDirs(library, searchDirs))
    .map(([, label]) => label);

  if (missing.length === 0) return null;
  return `NVIDIA driver libraries are visible, but CUDA user-space libraries are missing from loader paths (${missing.join(", ")})`;
}

async function runDoctorDeviceChecks(nextSteps: string[]): Promise<void> {
  const mode = configuredGpuModeLabel();
  doctorCheck("device mode", true, mode);

  const skipProbe = ["0", "false", "off", "no", "skip"].includes((process.env.QMD_DOCTOR_DEVICE_PROBE ?? "").trim().toLowerCase());
  if (skipProbe) {
    doctorCheck("device probe", false, "skipped by QMD_DOCTOR_DEVICE_PROBE=0. Next: unset it and rerun `qmd doctor` to verify GPU/CPU acceleration");
    nextSteps.push("Unset `QMD_DOCTOR_DEVICE_PROBE` and rerun `qmd doctor` when you want to verify llama.cpp device acceleration.");
    return;
  }

  const crashHint = "Probing native llama backend now. If qmd crashes here, rerun with `QMD_FORCE_CPU=1 qmd doctor` (or `QMD_DOCTOR_DEVICE_PROBE=0 qmd doctor` to skip this probe).";
  if (process.stdout.isTTY) {
    process.stdout.write(`${c.dim}${crashHint}${c.reset}`);
  }

  try {
    const device = await getDefaultLlamaCpp().getDeviceInfo({ allowBuild: false });
    if (process.stdout.isTTY) {
      process.stdout.write(`\r${" ".repeat(crashHint.length)}\r`);
    }
    if (device.gpu) {
      const gpuLabel = device.gpu === "metal" && process.platform === "darwin"
        ? "metal (macOS Metal backend)"
        : String(device.gpu);
      const parts = [`GPU ${gpuLabel}`, `offloading ${device.gpuOffloading ? "enabled" : "disabled"}`];
      if (device.gpuDevices.length > 0) parts.push(`devices: ${summarizeDeviceNames(device.gpuDevices)}`);
      if (device.vram) parts.push(`VRAM ${formatBytes(device.vram.free)} free / ${formatBytes(device.vram.total)} total`);
      parts.push(`${device.cpuCores} CPU math cores`);
      doctorCheck("device probe", device.gpuOffloading, device.gpuOffloading
        ? parts.join("; ")
        : `${parts.join("; ")}. Next: check QMD_LLAMA_GPU and llama.cpp backend support`);
      if (!device.gpuOffloading) {
        nextSteps.push("GPU was detected but offloading is disabled; check `QMD_LLAMA_GPU=metal|cuda|vulkan` and rerun `qmd doctor`.");
      }

      // Surface the darwin residency-set mitigation. libggml-metal's
      // process-static device dtor asserts on un-expired residency sets
      // during libc exit() (ggml-org/llama.cpp#22593), producing a giant
      // stderr backtrace after correct output. The bin/qmd launcher exports
      // GGML_METAL_NO_RESIDENCY=1 on darwin to skip the assertion entirely.
      // No measurable perf cost on short-lived CLI calls.
      if (device.gpu === "metal" && process.platform === "darwin") {
        if (isDarwinMetalMitigationActive()) {
          doctorCheck(
            "darwin metal residency",
            true,
            "GGML_METAL_NO_RESIDENCY=1 set by launcher; clean process exit (avoids ggml-org/llama.cpp#22593). Opt back in with QMD_METAL_KEEP_RESIDENCY=1 if you run long-lived qmd processes."
          );
        } else {
          doctorCheck(
            "darwin metal residency",
            false,
            "residency sets active (QMD_METAL_KEEP_RESIDENCY=1 or launcher bypassed); llama-using commands may dump a libggml-metal backtrace at exit (ggml-org/llama.cpp#22593) even when output succeeded."
          );
          nextSteps.push("Unset `QMD_METAL_KEEP_RESIDENCY` so the launcher can disable Metal residency sets; without this, query/vsearch/embed dump a stack trace at exit even on success.");
        }
      }
    } else {
      const cudaDiagnostic = linuxCudaRuntimeDiagnostic();
      const diagnosticSuffix = cudaDiagnostic ? ` ${cudaDiagnostic}.` : "";
      doctorCheck("device probe", false, `running on CPU (${device.cpuCores} math cores).${diagnosticSuffix} Next: install/configure Metal, CUDA, or Vulkan for faster embeddings, or set QMD_FORCE_CPU=1 to make CPU mode explicit`);
      if (cudaDiagnostic) {
        nextSteps.push(`${cudaDiagnostic}; install CUDA runtime/cuBLAS libraries or add their directory to LD_LIBRARY_PATH, then rerun \`qmd doctor\`.`);
      } else {
        nextSteps.push("Vector operations are running on CPU; install/configure Metal, CUDA, or Vulkan if embedding/query performance is too slow.");
      }
    }
  } catch (error) {
    if (process.stdout.isTTY) {
      process.stdout.write(`\r${" ".repeat(crashHint.length)}\r`);
    }
    const message = error instanceof Error ? sanitizeDiagnosticMessage(error.message) : sanitizeDiagnosticMessage(String(error));
    doctorCheck("device probe", false, `probe failed: ${message}. Next: run with QMD_FORCE_CPU=1 to bypass GPU probing, or set QMD_LLAMA_GPU=metal|cuda|vulkan and retry`);
    nextSteps.push("GPU probe failed; try `QMD_FORCE_CPU=1 qmd doctor` to confirm CPU fallback, then fix GPU drivers/backend if acceleration is expected.");
  }
}

async function showDoctor(): Promise<void> {
  const storeInstance = getStore();
  const db = storeInstance.db;
  const pkg = readPackageJson();
  const activeModels = resolveModelsForCli();
  const embedModel = activeModels.embed;
  const fingerprint = getEmbeddingFingerprint(embedModel);
  const nextSteps: string[] = [];

  console.log(`${c.bold}QMD Doctor${c.reset}\n`);
  console.log(`Index: ${getDbPath()}`);
  console.log(`Runtime: ${isBun ? "bun:sqlite" : "better-sqlite3"}`);

  try {
    doctorCheck("SQLite runtime", true, getSqliteVersion(db));
  } catch (error) {
    doctorCheck("SQLite runtime", false, error instanceof Error ? error.message : String(error));
  }

  const betterSqliteVersion = pkg.dependencies?.["better-sqlite3"] ?? pkg.devDependencies?.["better-sqlite3"] ?? "not declared";
  doctorCheck("better-sqlite3 package", true, String(betterSqliteVersion));

  try {
    doctorCheck("sqlite-vec", true, getVecVersion(db));
  } catch (error) {
    doctorCheck("sqlite-vec", false, error instanceof Error ? error.message : String(error));
  }

  const configCheck = checkDoctorIndexConfig(nextSteps);
  const configModels = configCheck.config?.models ?? {};
  checkEnvironmentOverrides(activeModels, configModels);
  checkModelDefaults(activeModels, configModels);
  checkModelCache(activeModels, nextSteps);

  await runDoctorDeviceChecks(nextSteps);

  try {
    const adoption = await maybeAdoptLegacyEmbeddingFingerprint(storeInstance, embedModel);
    if (adoption.checked || adoption.adopted > 0) {
      doctorCheck("legacy fingerprint adoption", adoption.adopted > 0, adoption.adopted > 0 ? `adopted ${adoption.adopted} legacy chunks; ${adoption.reason}` : adoption.reason);
    }
  } catch (error) {
    doctorCheck("legacy fingerprint adoption", false, error instanceof Error ? error.message : String(error));
  }

  try {
    const pending = getHashesNeedingEmbedding(db, undefined, embedModel);
    doctorCheck("embedding freshness", pending === 0, pending === 0 ? "all active documents match current fingerprint" : `${formatCount(pending)} active documents need embeddings. Next: \`qmd embed\``);
    if (pending > 0) {
      nextSteps.push(`Run \`qmd embed\` to generate ${formatCount(pending)} missing/stale document embeddings.`);
    }
  } catch (error) {
    doctorCheck("embedding freshness", false, error instanceof Error ? error.message : String(error));
  }

  try {
    const rows = getEmbeddingFingerprintGroups(db);
    const uniqueFingerprints = new Set(rows.map(row => row.fingerprint));
    const offCurrent = rows.filter(row => row.model === embedModel && row.fingerprint !== fingerprint);
    const ok = rows.length === 0 || (uniqueFingerprints.size === 1 && rows[0]?.fingerprint === fingerprint && offCurrent.length === 0);
    const currentDocs = rows
      .filter(row => row.model === embedModel && row.fingerprint === fingerprint)
      .reduce((sum, row) => sum + row.docs, 0);
    const otherDocs = rows.reduce((sum, row) => sum + row.docs, 0) - currentDocs;
    const groups = rows.map(row => {
      const label = row.fingerprint === fingerprint ? "current" : (row.fingerprint || "legacy");
      return `${shortModelName(row.model)}:${label} ${formatCount(row.docs)} docs/${formatCount(row.chunks)} chunks`;
    }).join("; ");
    const namedFingerprintRows = rows.filter(row => row.fingerprint);
    const namedFingerprints = [...new Set(namedFingerprintRows.map(row => row.fingerprint))];
    if (namedFingerprints.length > 1) {
      const namedGroups = namedFingerprintRows
        .map(row => `${row.fingerprint}${row.fingerprint === fingerprint ? " (current)" : ""}: ${shortModelName(row.model)} ${formatCount(row.docs)} docs/${formatCount(row.chunks)} chunks`)
        .join("; ");
      doctorCheck("mixed named embedding fingerprints", false, `content_vectors contains ${namedFingerprints.length} named fingerprints: ${namedGroups}. Next: \`qmd embed\` or \`qmd embed --force\``);
      nextSteps.push("Run `qmd embed` to converge mixed named embedding fingerprints; use `qmd embed --force` if old named fingerprints or vector sample mismatches remain.");
    }
    const details = rows.length === 0
      ? `no vectors yet; current fingerprint ${fingerprint}`
      : ok
        ? `${formatCount(currentDocs)} docs on current fingerprint (${fingerprint})`
        : `${formatCount(currentDocs)} docs current, ${formatCount(otherDocs)} docs legacy/stale. ${groups}. Next: \`qmd embed\``;
    doctorCheck("embedding fingerprints", ok, details);
    if (!ok) {
      nextSteps.push("Run `qmd embed` to migrate active documents to the current embedding fingerprint; use `qmd embed --force` if vector samples still fail afterward.");
    }
  } catch (error) {
    doctorCheck("embedding fingerprints", false, error instanceof Error ? error.message : String(error));
  }

  try {
    const vectorSample = await checkEmbeddingVectorSamples(db, embedModel, fingerprint);
    doctorCheck("embedding vector sample", vectorSample.ok, vectorSample.details);
    if (!vectorSample.ok) {
      nextSteps.push("Run `qmd embed --force` to rebuild existing vectors that no longer reproduce under the current embedding pipeline.");
    }
  } catch (error) {
    const message = error instanceof Error ? sanitizeDiagnosticMessage(error.message) : sanitizeDiagnosticMessage(String(error));
    doctorCheck("embedding vector sample", false, `${message}; rebuild with \`qmd embed --force\``);
    nextSteps.push("Run `qmd embed --force` to rebuild existing vectors, then rerun `qmd doctor`.");
  }

  const steps = normalizedDoctorNextSteps(nextSteps);
  if (steps.length > 0) {
    console.log(`\n${c.bold}Recommended next step${steps.length === 1 ? "" : "s"}${c.reset}`);
    for (const step of steps) {
      console.log(`  - ${step}`);
    }
  }

  closeDb();
}

export {
  showStatus,
  showDoctor,
  formatCount,
  shortModelName,
};
