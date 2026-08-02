/**
 * store-queries.test.ts - Unit tests for the narrow query functions sunk out of
 * the CLI layer (src/cli/qmd.ts) into the store, per
 * docs/specs/qmd-cli-sql-ownership.md (Phase 1).
 *
 * These pin the *existing* behavior of the SQL that used to be inline in the
 * CLI, so the relocation is provably behavior-preserving (spec invariant I2).
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Database } from "../src/db.js";
import type { CollectionConfig } from "../src/collections.js";
import {
  createStore,
  syncConfigToDb,
  insertContent,
  insertDocument,
  hashContent,
  invalidateConfigCache,
  type Store,
} from "../src/store.js";

let store: Store;
let db: Database;
let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "qmd-storeq-"));
  store = createStore(join(dir, "test.sqlite"));
  db = store.db;
});

afterEach(async () => {
  store.close();
  await rm(dir, { recursive: true, force: true });
});

/** Seed a document (and its content) directly, mirroring how the CLI sees it. */
async function seedDoc(
  collection: string,
  path: string,
  opts: { title?: string; body?: string; active?: number } = {}
): Promise<string> {
  const now = new Date().toISOString();
  const body = opts.body ?? `# ${opts.title ?? path}\n\nbody`;
  const hash = await hashContent(body);
  insertContent(db, hash, body, now);
  insertDocument(db, collection, path, opts.title ?? path, hash, now, now);
  if (opts.active === 0) {
    db.prepare(`UPDATE documents SET active = 0 WHERE collection = ? AND path = ?`).run(collection, path);
  }
  return hash;
}

describe("invalidateConfigCache", () => {
  test("deletes only the config_hash row", async () => {
    const config: CollectionConfig = { collections: {} };
    syncConfigToDb(db, config);
    expect(db.prepare(`SELECT value FROM store_config WHERE key = 'config_hash'`).get()).toBeTruthy();

    // An unrelated key must survive invalidation.
    db.prepare(`INSERT INTO store_config (key, value) VALUES ('other_key', 'x') ON CONFLICT(key) DO UPDATE SET value = excluded.value`).run();

    invalidateConfigCache(db);

    expect(db.prepare(`SELECT value FROM store_config WHERE key = 'config_hash'`).get()).toBeUndefined();
    const other = db.prepare(`SELECT value FROM store_config WHERE key = 'other_key'`).get() as { value: string } | undefined;
    expect(other?.value).toBe("x");
  });

  test("forces syncConfigToDb to re-sync on next call", () => {
    const config: CollectionConfig = { collections: {} };
    syncConfigToDb(db, config);
    invalidateConfigCache(db);
    // With the hash cleared, the next sync rewrites it (i.e. it did NOT early-return).
    syncConfigToDb(db, config);
    expect(db.prepare(`SELECT value FROM store_config WHERE key = 'config_hash'`).get()).toBeTruthy();
  });
});
