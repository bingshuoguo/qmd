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
  insertEmbedding,
  hashContent,
  invalidateConfigCache,
  countActiveDocuments,
  countContentVectors,
  getLatestDocumentModifiedAt,
  findDocumentRef,
  getDocumentHash,
  getDocumentContent,
  countDocumentsInCollection,
  listDocumentsWithMeta,
  hasVectorTable,
  sampleEmbeddedChunks,
  getStoredEmbedding,
  getSqliteVersion,
  getVecVersion,
  getEmbeddingFingerprintGroups,
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

describe("countActiveDocuments", () => {
  test("counts only active documents", async () => {
    expect(countActiveDocuments(db)).toBe(0);
    await seedDoc("c1", "a.md");
    await seedDoc("c1", "b.md");
    await seedDoc("c1", "gone.md", { active: 0 });
    expect(countActiveDocuments(db)).toBe(2);
  });
});

describe("countContentVectors", () => {
  test("counts content_vectors rows", async () => {
    expect(countContentVectors(db)).toBe(0);
    const hash = await seedDoc("c1", "a.md");
    store.ensureVecTable(3);
    insertEmbedding(db, hash, 0, 0, new Float32Array([1, 2, 3]), "m", new Date().toISOString());
    expect(countContentVectors(db)).toBe(1);
  });
});

describe("getLatestDocumentModifiedAt", () => {
  test("null with no active docs; otherwise MAX(modified_at) over active only", () => {
    expect(getLatestDocumentModifiedAt(db)).toBeNull();
    const now = new Date().toISOString();
    insertContent(db, "h1", "b", now);
    insertDocument(db, "c1", "old.md", "t", "h1", "2020-01-01T00:00:00.000Z", "2020-01-01T00:00:00.000Z");
    insertContent(db, "h2", "b", now);
    insertDocument(db, "c1", "new.md", "t", "h2", "2021-01-01T00:00:00.000Z", "2021-06-01T00:00:00.000Z");
    expect(getLatestDocumentModifiedAt(db)).toBe("2021-06-01T00:00:00.000Z");

    // Inactive documents are excluded from the MAX.
    db.prepare(`UPDATE documents SET active = 0 WHERE path = 'new.md'`).run();
    expect(getLatestDocumentModifiedAt(db)).toBe("2020-01-01T00:00:00.000Z");
  });
});

describe("findDocumentRef", () => {
  test("resolves a qmd:// virtual path by exact collection+path", async () => {
    await seedDoc("c1", "a/b.md", { body: "hello body" });
    const ref = findDocumentRef(db, "qmd://c1/a/b.md");
    expect(ref).not.toBeNull();
    expect(ref?.collection).toBe("c1");
    expect(ref?.path).toBe("a/b.md");
    expect(ref?.virtual_path).toBe("qmd://c1/a/b.md");
    expect(ref?.body_length).toBe("hello body".length);
  });

  test("resolves a bare path by exact match", async () => {
    await seedDoc("c1", "a/b.md");
    expect(findDocumentRef(db, "a/b.md")?.path).toBe("a/b.md");
  });

  test("falls back to suffix match when there is no exact path", async () => {
    await seedDoc("c1", "a/b.md");
    expect(findDocumentRef(db, "b.md")?.path).toBe("a/b.md");
  });

  test("returns null when nothing matches", async () => {
    await seedDoc("c1", "a/b.md");
    expect(findDocumentRef(db, "nope.md")).toBeNull();
  });

  test("excludes inactive documents", async () => {
    await seedDoc("c1", "a/b.md", { active: 0 });
    expect(findDocumentRef(db, "a/b.md")).toBeNull();
    expect(findDocumentRef(db, "qmd://c1/a/b.md")).toBeNull();
  });
});

describe("getDocumentHash", () => {
  test("returns the content hash for an active doc, null otherwise", async () => {
    const hash = await seedDoc("c1", "a.md");
    expect(getDocumentHash(db, "c1", "a.md")).toBe(hash);
    expect(getDocumentHash(db, "c1", "missing.md")).toBeNull();
  });
});

describe("getDocumentContent", () => {
  test("returns body + title for an active doc", async () => {
    await seedDoc("c1", "a.md", { title: "My Title", body: "# My Title\n\ncontent here" });
    const doc = getDocumentContent(db, "c1", "a.md");
    expect(doc?.title).toBe("My Title");
    expect(doc?.body).toBe("# My Title\n\ncontent here");
  });

  test("returns null for a missing doc", () => {
    expect(getDocumentContent(db, "c1", "missing.md")).toBeNull();
  });
});

describe("countDocumentsInCollection", () => {
  test("counts active docs per collection; zero-doc collection returns 0", async () => {
    await seedDoc("c1", "a.md");
    await seedDoc("c1", "b.md");
    await seedDoc("c2", "x.md");
    await seedDoc("c1", "gone.md", { active: 0 });
    expect(countDocumentsInCollection(db, "c1")).toBe(2);
    expect(countDocumentsInCollection(db, "c2")).toBe(1);
    expect(countDocumentsInCollection(db, "empty")).toBe(0);
  });
});

describe("listDocumentsWithMeta", () => {
  test("lists active docs in a collection ordered by path, with title/size/mtime", async () => {
    await seedDoc("c1", "b.md", { title: "B", body: "bbb" });
    await seedDoc("c1", "a.md", { title: "A", body: "a" });
    await seedDoc("c1", "gone.md", { active: 0 });
    await seedDoc("c2", "other.md");
    const rows = listDocumentsWithMeta(db, "c1");
    expect(rows.map(r => r.path)).toEqual(["a.md", "b.md"]);
    expect(rows[0]?.title).toBe("A");
    expect(rows[0]?.size).toBe(1);
    expect(typeof rows[0]?.modified_at).toBe("string");
  });

  test("filters by path prefix", async () => {
    await seedDoc("c1", "sub/x.md");
    await seedDoc("c1", "sub/y.md");
    await seedDoc("c1", "top.md");
    const rows = listDocumentsWithMeta(db, "c1", "sub/");
    expect(rows.map(r => r.path)).toEqual(["sub/x.md", "sub/y.md"]);
  });
});

describe("hasVectorTable", () => {
  test("false before vectors_vec exists, true after ensureVecTable", () => {
    expect(hasVectorTable(db)).toBe(false);
    store.ensureVecTable(3);
    expect(hasVectorTable(db)).toBe(true);
  });
});

describe("sampleEmbeddedChunks", () => {
  test("samples embedded chunks for a model+fingerprint, joined to active doc", async () => {
    const hash = await seedDoc("c1", "a.md", { body: "hello world body" });
    store.ensureVecTable(3);
    insertEmbedding(db, hash, 0, 0, new Float32Array([1, 2, 3]), "m1", new Date().toISOString(), 1, "fp1");

    const rows = sampleEmbeddedChunks(db, "m1", "fp1", 5);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.hash).toBe(hash);
    expect(rows[0]?.seq).toBe(0);
    expect(rows[0]?.body).toBe("hello world body");
    expect(rows[0]?.path).toBe("a.md");

    // A different fingerprint matches nothing.
    expect(sampleEmbeddedChunks(db, "m1", "other", 5)).toHaveLength(0);
  });
});

describe("getStoredEmbedding", () => {
  test("returns stored embedding bytes for hash_seq, null when missing", async () => {
    const hash = await seedDoc("c1", "a.md");
    store.ensureVecTable(3);
    insertEmbedding(db, hash, 0, 0, new Float32Array([1, 2, 3]), "m", new Date().toISOString(), 1);
    const emb = getStoredEmbedding(db, `${hash}_0`);
    expect(emb).toBeInstanceOf(Uint8Array);
    expect(getStoredEmbedding(db, "nope_0")).toBeNull();
  });
});

describe("getSqliteVersion / getVecVersion", () => {
  test("returns version strings", () => {
    expect(getSqliteVersion(db).length).toBeGreaterThan(0);
    expect(getVecVersion(db).length).toBeGreaterThan(0);
  });
});

describe("getEmbeddingFingerprintGroups", () => {
  test("groups content_vectors by model+fingerprint (distinct docs / total chunks)", async () => {
    const h1 = await seedDoc("c1", "a.md");
    const h2 = await seedDoc("c1", "b.md");
    store.ensureVecTable(3);
    const now = new Date().toISOString();
    insertEmbedding(db, h1, 0, 0, new Float32Array([1, 2, 3]), "m1", now, 1, "fpA");
    insertEmbedding(db, h2, 0, 0, new Float32Array([1, 2, 3]), "m1", now, 1, "fpA");
    insertEmbedding(db, h1, 1, 0, new Float32Array([4, 5, 6]), "m1", now, 1, "fpB");

    const rows = getEmbeddingFingerprintGroups(db);
    const fpA = rows.find(r => r.fingerprint === "fpA");
    expect(fpA?.docs).toBe(2);
    expect(fpA?.chunks).toBe(2);
    const fpB = rows.find(r => r.fingerprint === "fpB");
    expect(fpB?.docs).toBe(1);
    expect(fpB?.chunks).toBe(1);
  });
});
