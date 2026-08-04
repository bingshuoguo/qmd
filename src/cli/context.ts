/**
 * context.ts - store/index lifecycle for the qmd CLI.
 *
 * Holds the only mutable module-level state in the CLI layer: the open store
 * handle, an optional db-path override, and the active index name. Everything
 * else in src/cli/ is a consumer of these functions.
 */
import type { Database } from "../db.js";
import { resolve as pathResolve } from "path";
import {
  createStore,
  syncConfigToDb,
  invalidateConfigCache,
  getDefaultDbPath,
} from "../store.js";
import { loadConfig, saveConfig } from "../collections.js";
import { setDefaultLlamaCpp, LlamaCpp, resolveModels } from "../llm.js";

let store: ReturnType<typeof createStore> | null = null;
let storeDbPathOverride: string | undefined;
let currentIndexName = "index";

function getStore(): ReturnType<typeof createStore> {
  if (!store) {
    store = createStore(storeDbPathOverride);
    // Sync YAML config into SQLite store_collections so store.ts reads from DB
    try {
      const activeModels = ensureModelsConfiguredForCli();
      const config = loadConfig();
      syncConfigToDb(store.db, config);
      setDefaultLlamaCpp(new LlamaCpp({
        embedModel: activeModels.embed,
        generateModel: activeModels.generate,
        rerankModel: activeModels.rerank,
      }));
    } catch {
      // Config may not exist yet — that's fine, DB works without it
    }
  }
  return store;
}

function getDb(): Database {
  return getStore().db;
}

/** Re-sync YAML config into SQLite after CLI mutations (add/remove/rename collection, context changes) */
function resyncConfig(): void {
  const s = getStore();
  try {
    const config = loadConfig();
    // Clear config hash to force re-sync
    invalidateConfigCache(s.db);
    syncConfigToDb(s.db, config);
  } catch {
    // Config may not exist — that's fine
  }
}

function closeDb(): void {
  if (store) {
    store.close();
    store = null;
  }
}

function getDbPath(): string {
  return store?.dbPath ?? storeDbPathOverride ?? getDefaultDbPath();
}

function setStoreDbPathOverride(dbPath: string | undefined): void {
  storeDbPathOverride = dbPath;
}

function getActiveIndexName(): string {
  return currentIndexName;
}

function setIndexName(name: string | null): void {
  let normalizedName = name;
  // Normalize relative paths to prevent malformed database paths
  if (name && name.includes('/')) {
    const absolutePath = pathResolve(process.cwd(), name);
    // Replace path separators with underscores to create a valid filename
    normalizedName = absolutePath.replace(/\//g, '_').replace(/^_/, '');
  }
  currentIndexName = normalizedName || "index";
  storeDbPathOverride = normalizedName ? getDefaultDbPath(normalizedName) : undefined;
  // Reset open handle so next use opens the new index
  closeDb();
}

function ensureVecTable(_db: Database, dimensions: number): void {
  // Store owns the DB; ignore `_db` and ensure vec table on the active store
  getStore().ensureVecTable(dimensions);
}

function ensureModelsConfiguredForCli(): { embed: string; generate: string; rerank: string } {
  try {
    const config = loadConfig();
    const models = resolveModels(config.models);
    const current = config.models ?? {};
    if (current.embed !== models.embed || current.generate !== models.generate || current.rerank !== models.rerank) {
      saveConfig({
        ...config,
        models: {
          ...current,
          embed: models.embed,
          generate: models.generate,
          rerank: models.rerank,
        },
      });
    }
    return models;
  } catch {
    return resolveModels();
  }
}

function resolveEmbedModelForCli(): string {
  return ensureModelsConfiguredForCli().embed;
}

function resolveGenerateModelForCli(): string {
  return ensureModelsConfiguredForCli().generate;
}

function resolveRerankModelForCli(): string {
  return ensureModelsConfiguredForCli().rerank;
}

function resolveModelsForCli(): { embed: string; generate: string; rerank: string } {
  return ensureModelsConfiguredForCli();
}

export {
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
  resolveGenerateModelForCli,
  resolveRerankModelForCli,
  resolveModelsForCli,
};
