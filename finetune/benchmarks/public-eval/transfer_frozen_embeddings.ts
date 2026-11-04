#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { gunzipSync, gzipSync } from "node:zlib";

import {
  DEFAULT_EMBED_MODEL,
  createStore,
  getStoredEmbedding,
} from "../../../src/store.js";

interface TransferRow {
  hash: string;
  seq: number;
  pos: number;
  model: string;
  embed_fingerprint: string;
  total_chunks: number;
  embedded_at: string;
  embedding_base64: string;
}

interface TransferPayload {
  format: "qmd-frozen-embedding-transfer-v1";
  model: string;
  collections: string[];
  row_count: number;
  rows_sha256: string;
  rows: TransferRow[];
}

const { values } = parseArgs({
  options: {
    mode: { type: "string" },
    db: { type: "string" },
    file: { type: "string" },
    collection: { type: "string", multiple: true },
    model: { type: "string", default: DEFAULT_EMBED_MODEL },
  },
  strict: true,
});

if (!values.mode || !values.db || !values.file) {
  throw new Error(
    "Usage: transfer_frozen_embeddings.ts --mode export|import --db <index.sqlite> "
    + "--file <transfer.json.gz> [--collection <name> ...] [--model <uri>]",
  );
}

const mode = values.mode;
const dbPath = resolve(values.db);
const transferPath = resolve(values.file);
const collections = [...new Set(values.collection ?? [])].sort();
const model = values.model;

if (!existsSync(dbPath)) throw new Error(`Index does not exist: ${dbPath}`);

function canonicalRows(rows: TransferRow[]): string {
  return rows.map(row => JSON.stringify(row)).join("\n");
}

function rowsSha256(rows: TransferRow[]): string {
  return createHash("sha256").update(canonicalRows(rows), "utf8").digest("hex");
}

if (mode === "export") {
  if (collections.length === 0) throw new Error("Export requires at least one --collection");
  const store = createStore(dbPath);
  try {
    const placeholders = collections.map(() => "?").join(", ");
    const metadata = store.db.prepare(`
      SELECT DISTINCT
        cv.hash,
        cv.seq,
        cv.pos,
        cv.model,
        cv.embed_fingerprint,
        cv.total_chunks,
        cv.embedded_at
      FROM content_vectors cv
      JOIN documents d ON d.hash = cv.hash AND d.active = 1
      WHERE d.collection IN (${placeholders}) AND cv.model = ?
      ORDER BY cv.hash, cv.seq
    `).all(...collections, model) as Omit<TransferRow, "embedding_base64">[];

    const rows = metadata.map(row => {
      const embedding = getStoredEmbedding(store.db, `${row.hash}_${row.seq}`);
      if (!embedding) throw new Error(`Missing vector payload for ${row.hash}_${row.seq}`);
      return { ...row, embedding_base64: Buffer.from(embedding).toString("base64") };
    });
    const payload: TransferPayload = {
      format: "qmd-frozen-embedding-transfer-v1",
      model,
      collections,
      row_count: rows.length,
      rows_sha256: rowsSha256(rows),
      rows,
    };
    writeFileSync(transferPath, gzipSync(Buffer.from(JSON.stringify(payload), "utf8")));
    process.stdout.write(`${JSON.stringify({
      status: "exported",
      db: dbPath,
      file: transferPath,
      collections,
      model,
      row_count: rows.length,
      rows_sha256: payload.rows_sha256,
    }, null, 2)}\n`);
  } finally {
    store.close();
  }
} else if (mode === "import") {
  if (!existsSync(transferPath)) throw new Error(`Transfer file does not exist: ${transferPath}`);
  const payload = JSON.parse(gunzipSync(readFileSync(transferPath)).toString("utf8")) as TransferPayload;
  if (payload.format !== "qmd-frozen-embedding-transfer-v1") {
    throw new Error(`Unsupported transfer format: ${String(payload.format)}`);
  }
  if (payload.row_count !== payload.rows.length) throw new Error("Transfer row count mismatch");
  const actualRowsSha256 = rowsSha256(payload.rows);
  if (actualRowsSha256 !== payload.rows_sha256) throw new Error("Transfer payload checksum mismatch");
  if (model !== payload.model) throw new Error(`Transfer model mismatch: expected ${model}, got ${payload.model}`);

  const store = createStore(dbPath);
  try {
    const placeholders = payload.collections.map(() => "?").join(", ");
    const activeHashes = new Set((store.db.prepare(`
      SELECT DISTINCT hash
      FROM documents
      WHERE active = 1 AND collection IN (${placeholders})
    `).all(...payload.collections) as { hash: string }[]).map(row => row.hash));
    const unexpected = payload.rows.find(row => !activeHashes.has(row.hash));
    if (unexpected) {
      throw new Error(
        `Destination has no active document for transferred hash ${unexpected.hash} `
        + `in collections ${payload.collections.join(", ")}`,
      );
    }

    const insertAll = store.db.transaction(() => {
      for (const row of payload.rows) {
        const bytes = Buffer.from(row.embedding_base64, "base64");
        if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
          throw new Error(`Invalid vector byte length for ${row.hash}_${row.seq}`);
        }
        const copy = Uint8Array.from(bytes).buffer;
        store.insertEmbedding(
          row.hash,
          row.seq,
          row.pos,
          new Float32Array(copy),
          row.model,
          row.embedded_at,
          row.total_chunks,
          row.embed_fingerprint,
        );
      }
    });
    insertAll();
    process.stdout.write(`${JSON.stringify({
      status: "imported",
      db: dbPath,
      file: transferPath,
      collections: payload.collections,
      model: payload.model,
      row_count: payload.rows.length,
      rows_sha256: payload.rows_sha256,
    }, null, 2)}\n`);
  } finally {
    store.close();
  }
} else {
  throw new Error(`Unsupported --mode: ${mode}`);
}
