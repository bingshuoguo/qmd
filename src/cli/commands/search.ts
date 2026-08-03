/**
 * commands/search.ts - search / vsearch / query commands: BM25 (search),
 * vector similarity (vectorSearch), and hybrid query with expansion +
 * reranking (querySearch), plus the structured query document parser
 * (lex:/vec:/hyde:/intent: lines) and the -c collection filter.
 * sanitizeFTS5Term/buildFTS5Query/normalizeBM25 are pre-existing dead code
 * (no callers), moved here per spec §8.2 — removal is a separate decision.
 */
import type { Database } from "../../db.js";
import {
  searchFTS,
  getContextForFile,
  getIndexHealth,
  hybridQuery,
  vectorSearchQuery,
  structuredSearch,
  DEFAULT_EMBED_MODEL,
  DEFAULT_RERANK_MODEL,
  type ExpandedQuery,
} from "../../store.js";
import { withLLMSession } from "../../llm.js";
import {
  getCollection as getCollectionFromYaml,
  getDefaultCollectionNames,
} from "../../collections.js";
import { getStore, getDb, closeDb, resolveEmbedModelForCli } from "../context.js";
import { c, progress, formatMs } from "../term.js";
import { printEmptySearchResults, outputResults, type OutputOptions } from "../output.js";

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

export {
  search,
  vectorSearch,
  querySearch,
  resolveCollectionFilter,
  filterByCollections,
  parseStructuredQuery,
  type ParsedStructuredQuery,
};
