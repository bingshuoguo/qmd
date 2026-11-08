/**
 * probe.ts — one-off probes for the FTS scope-isolation experiment.
 *
 *   tsx probe.ts <mode> <variant> <benchmark-dir> <out.jsonl>
 *
 * modes:
 *   bm25    R1: searchFTS(original query, per_list_limit, collection)
 *   vec     R2: searchVec(original query, per_list_limit, collection)
 *   strong  R4: production strong-signal decision on the unpinned BM25 probe;
 *               triggered queries are re-run through executeHybridRetrieval
 *               with no expansions (what production does when it fires)
 *   gates   G3/G4: FTS row count vs active docs; bm25() of the index's FTS
 *               table vs a freshly built FTS5 table over the same rows
 *
 * Index is opened read-mostly through createStore, like openBenchmarkStore.
 * Paths: INDEX_PATH is not used; the variant name resolves the copy.
 */
import Database from "better-sqlite3";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { LlamaCpp } from "../../../src/llm.js";
import {
  createStore,
  executeHybridRetrieval,
  STRONG_SIGNAL_MIN_GAP,
  STRONG_SIGNAL_MIN_SCORE,
  type SearchResult,
} from "../../../src/store.js";
import { loadBenchmarkV2, loadRetrievalProfile } from "../../../src/bench/qrels.js";

const [mode, variant, benchArg, outArg] = process.argv.slice(2);
if (!mode || !variant || !benchArg || !outArg) {
  console.error("usage: probe.ts <bm25|vec|strong|gates> <variant> <benchmark-dir> <out.jsonl>");
  process.exit(1);
}
const root = resolve(benchArg);
const out = resolve(outArg);
const X = resolve(import.meta.dirname, "../../../artifacts/fts-scope-isolation");
const dbPath = join(X, "index", variant, "index.sqlite");
if (!existsSync(dbPath)) throw new Error(`missing index copy: ${dbPath}`);

const benchmark = loadBenchmarkV2(root);
const profile = loadRetrievalProfile(root, benchmark.manifest.cutoffs);
const collection = profile.collection_name;
const docIdByPath = new Map(benchmark.documents.map(d => [d.path, d.doc_id]));
const prefix = `qmd://${collection}/`;

function docIds(results: { file?: string; filepath?: string }[]): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const r of results) {
    const file = (r.filepath ?? r.file)!;
    if (!file.startsWith(prefix)) throw new Error(`result outside collection: ${file}`);
    const id = docIdByPath.get(file.slice(prefix.length));
    if (!id) throw new Error(`result not in documents.jsonl: ${file}`);
    if (seen.has(id)) continue;
    seen.add(id);
    ids.push(id);
  }
  return ids;
}

function ranking(ids: string[]) {
  return ids.map((doc_id, i) => ({ rank: i + 1, doc_id }));
}

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, "");
const emit = (row: unknown) => appendFileSync(out, `${JSON.stringify(row)}\n`);

const store = createStore(dbPath);
const needsModels = mode === "vec" || mode === "strong";
const llm = needsModels
  ? new LlamaCpp({
    embedModel: profile.embedding_model,
    rerankModel: profile.reranker_model ?? undefined,
    inactivityTimeoutMs: 0,
    disposeModelsOnInactivity: false,
  })
  : null;
if (llm) store.llm = llm;

const started = Date.now();
try {
  if (mode === "bm25") {
    for (const q of benchmark.queries) {
      const t0 = performance.now();
      const results = store.searchFTS(q.query, profile.per_list_limit, collection);
      const latency_ms = performance.now() - t0;
      emit({
        qid: q.qid, retrieval_status: "ok",
        ranking: ranking(docIds(results)),
        scores: results.map(r => r.score),
        latency_ms,
      });
    }
  } else if (mode === "vec") {
    for (const q of benchmark.queries) {
      const results = await store.searchVec(q.query, profile.embedding_model, profile.per_list_limit, collection);
      emit({
        qid: q.qid, retrieval_status: "ok",
        ranking: ranking(docIds(results)),
        scores: results.map(r => r.score),
      });
    }
  } else if (mode === "strong") {
    let fired = 0;
    for (const q of benchmark.queries) {
      const initialFts: SearchResult[] = store.searchFTS(q.query, profile.per_list_limit, collection);
      const top = initialFts[0]?.score ?? 0;
      const second = initialFts[1]?.score ?? 0;
      // Same rule as executeHybridRetrieval; no benchmark query carries an intent.
      const strong = initialFts.length > 0 && top >= STRONG_SIGNAL_MIN_SCORE && (top - second) >= STRONG_SIGNAL_MIN_GAP;
      let rankingOut: ReturnType<typeof ranking> | null = null;
      if (strong) {
        fired++;
        const results = await executeHybridRetrieval(store, {
          originalQuery: q.query,
          originalFts: initialFts,
          expansions: [],
          collection,
          resultLimit: profile.result_limit,
          perListLimit: profile.per_list_limit,
          candidateLimit: profile.candidate_limit,
          rerank: profile.rerank,
        });
        rankingOut = ranking(docIds(results));
      }
      emit({ qid: q.qid, strong, top, second, retrieval_status: "ok", ranking: rankingOut });
    }
    console.error(`${variant} ${collection}: strong-signal fired ${fired}/${benchmark.queries.length}`);
  } else if (mode === "gates") {
    const db = store.db;
    const ftsRows = (db.prepare(`SELECT count(*) AS n FROM documents_fts`).get() as { n: number }).n;
    const activeDocs = (db.prepare(`SELECT count(*) AS n FROM documents WHERE active = 1`).get() as { n: number }).n;
    const targetDocs = (db.prepare(`SELECT count(*) AS n FROM documents WHERE active = 1 AND collection = ?`).get(collection) as { n: number }).n;

    // G4: rebuild an FTS5 table from the stored rows in a scratch in-memory
    // database and compare bm25() for the same MATCH expressions.
    const scratch = new Database(":memory:");
    scratch.exec(`CREATE VIRTUAL TABLE f USING fts5(filepath, title, body, tokenize='porter unicode61')`);
    const insert = scratch.prepare(`INSERT INTO f(rowid, filepath, title, body) VALUES (?, ?, ?, ?)`);
    scratch.transaction(() => {
      for (const row of db.prepare(`SELECT rowid, filepath, title, body FROM documents_fts`).iterate() as Iterable<{ rowid: number; filepath: string; title: string; body: string }>) {
        insert.run(row.rowid, row.filepath, row.title, row.body);
      }
    })();
    const sample = benchmark.queries.slice(0, 50);
    let compared = 0, maxDiff = 0, orderMismatch = 0;
    const sql = (t: string) => `SELECT rowid, bm25(${t}, 1.5, 4.0, 1.0) AS s FROM ${t} WHERE ${t} MATCH ? ORDER BY s, rowid LIMIT 50`;
    const fromIndex = db.prepare(sql("documents_fts"));
    const fromScratch = scratch.prepare(sql("f"));
    for (const q of sample) {
      const tokens = [...new Set(q.query.toLowerCase().match(/[a-z0-9]{3,}/g) ?? [])];
      if (tokens.length === 0) continue;
      const expr = tokens.map(t => `"${t}"`).join(" OR ");
      const a = fromIndex.all(expr) as { rowid: number; s: number }[];
      const b = fromScratch.all(expr) as { rowid: number; s: number }[];
      compared++;
      if (a.length !== b.length || a.some((r, i) => r.rowid !== b[i]!.rowid)) orderMismatch++;
      for (let i = 0; i < Math.min(a.length, b.length); i++) maxDiff = Math.max(maxDiff, Math.abs(a[i]!.s - b[i]!.s));
    }
    emit({ variant, collection, fts_rows: ftsRows, active_docs: activeDocs, target_docs: targetDocs, g4_queries: compared, g4_order_mismatch: orderMismatch, g4_max_abs_diff: maxDiff });
    console.error(JSON.stringify({ variant, collection, ftsRows, activeDocs, targetDocs, compared, orderMismatch, maxDiff }));
  } else {
    throw new Error(`unknown mode ${mode}`);
  }
} finally {
  await llm?.dispose();
  store.close();
}
console.error(`${mode} ${variant} ${collection} done in ${Math.round((Date.now() - started) / 1000)}s`);
