/**
 * output.ts - search result rendering (all 6 output formats) and editor
 * link building for the qmd CLI.
 *
 * Not a pure formatter: outputResults resolves the active index and looks up
 * on-disk paths for --full-path, so this module depends on context.ts (store
 * lifecycle) and commands/docs.ts (renderFullPath). Dependency direction
 * stays one-way: qmd.ts → output.ts → {context, docs} — docs.ts must not
 * import this module.
 */
import { existsSync } from "fs";
import {
  buildVirtualPath,
  parseVirtualPath,
  resolveVirtualPath,
  extractSnippet,
  addLineNumbers,
  type HybridQueryExplain,
  type ChunkStrategy,
} from "../store.js";
import { loadConfig } from "../collections.js";
import { escapeCSV, type OutputFormat } from "./formatter.js";
import { getDb, getActiveIndexName } from "./context.js";
import { c, formatScore, formatExplainNumber, highlightTerms } from "./term.js";
import { renderFullPath } from "./commands/docs.js";

type OutputOptions = {
  format: OutputFormat;
  full: boolean;
  limit: number;
  minScore: number;
  all?: boolean;
  collection?: string | string[];  // Filter by collection name(s)
  lineNumbers?: boolean; // Add line numbers to output
  explain?: boolean;     // Include retrieval score traces (query only)
  context?: string;      // Optional context for query expansion
  candidateLimit?: number;  // Max candidates to rerank (default: 40)
  intent?: string;       // Domain intent for disambiguation
  skipRerank?: boolean;  // Skip LLM reranking, use RRF scores only
  chunkStrategy?: ChunkStrategy;  // "auto" (default) or "regex"
  fullPath?: boolean;    // Show realpath instead of qmd:// URI (relative to $PWD when subpath)
};

type EmptySearchReason = "no_results" | "min_score";

// Emit format-safe empty output for search commands.
function printEmptySearchResults(format: OutputFormat, reason: EmptySearchReason = "no_results"): void {
  if (format === "json") {
    console.log("[]");
    return;
  }
  if (format === "csv") {
    console.log("docid,score,file,title,context,line,snippet");
    return;
  }
  if (format === "xml") {
    console.log("<results></results>");
    return;
  }
  if (format === "md" || format === "files") {
    return;
  }

  if (reason === "min_score") {
    console.log("No results found above minimum score threshold.");
    return;
  }
  console.log("No results found.");
}

type OutputRow = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
  context?: string | null;
  chunkPos?: number;
  chunkLen?: number;
  hash?: string;
  docid?: string;
  explain?: HybridQueryExplain;
};

const DEFAULT_EDITOR_URI_TEMPLATE = "vscode://file/{path}:{line}:{col}";

function encodePathForEditorUri(absolutePath: string): string {
  return encodeURI(absolutePath)
    .replace(/\?/g, "%3F")
    .replace(/#/g, "%23");
}

function getEditorUriTemplate(): string {
  const envTemplate = process.env.QMD_EDITOR_URI?.trim();
  if (envTemplate) return envTemplate;

  try {
    const config = loadConfig() as unknown as {
      editor_uri?: string;
      editor_uri_template?: string;
      editorUri?: string;
      [key: string]: unknown;
    };
    const configTemplate = (
      config.editor_uri
      || config.editor_uri_template
      || config.editorUri
      || (typeof config["editor-uri"] === "string" ? config["editor-uri"] : undefined)
    )?.trim();

    if (configTemplate) return configTemplate;
  } catch {
    // Ignore config parsing issues and use default template.
  }

  return DEFAULT_EDITOR_URI_TEMPLATE;
}

function buildEditorUri(template: string, absolutePath: string, line: number, col: number): string {
  const safeLine = Number.isFinite(line) && line > 0 ? Math.floor(line) : 1;
  const safeCol = Number.isFinite(col) && col > 0 ? Math.floor(col) : 1;
  const encodedPath = encodePathForEditorUri(absolutePath);

  return template
    .replace(/\{path\}/g, encodedPath)
    .replace(/\{line\}/g, String(safeLine))
    .replace(/\{col\}/g, String(safeCol))
    .replace(/\{column\}/g, String(safeCol));
}

function termLink(text: string, url: string, isTTY: boolean = !!process.stdout.isTTY): string {
  if (!isTTY) return text;
  return `\x1b]8;;${url}\x07${text}\x1b]8;;\x07`;
}

function outputResults(results: OutputRow[], query: string, opts: OutputOptions): void {
  const filtered = results.filter(r => r.score >= opts.minScore).slice(0, opts.limit);

  if (filtered.length === 0) {
    printEmptySearchResults(opts.format, "min_score");
    return;
  }

  // Helper to create qmd:// URI from displayPath
  const toQmdPath = (displayPath: string) => {
    const [collectionName, ...segments] = displayPath.split("/");
    if (!collectionName || segments.length === 0) {
      return `qmd://${displayPath}`;
    }
    const indexName = getActiveIndexName();
    return buildVirtualPath(
      collectionName,
      segments.join("/"),
      indexName === "index" ? undefined : indexName,
    );
  };

  // Helper to pick the visible path for a result. With --full-path we swap
  // the qmd:// URI for the file's on-disk path via renderFullPath() (./-
  // prefixed relative when under $PWD, absolute realpath otherwise). Falls
  // back to qmd:// if the file is no longer resolvable on disk.
  const linkDbForPaths = opts.fullPath ? getDb() : null;
  const displayPathFor = (row: OutputRow): string => {
    // Always rebuild from displayPath so the active index name is included
    // as ?index=… for non-default indexes. row.file may not carry it.
    const qmdUri = toQmdPath(row.displayPath);
    if (!opts.fullPath || !linkDbForPaths) return qmdUri;
    const absolute = resolveVirtualPath(linkDbForPaths, qmdUri);
    if (!absolute || !existsSync(absolute)) return qmdUri;
    return renderFullPath(absolute);
  };

  if (opts.format === "json") {
    // JSON output for LLM consumption
    const output = filtered.map(row => {
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : undefined);
      const snippetInfo = extractSnippet(row.body, query, 300, row.chunkPos, row.chunkLen, opts.intent);
      let body = opts.full ? row.body : undefined;
      let snippet = !opts.full ? snippetInfo.snippet : undefined;
      if (opts.lineNumbers) {
        if (body) body = addLineNumbers(body);
        if (snippet) snippet = addLineNumbers(snippet);
      }
      // With --full-path, omit docid (the on-disk path is the identifier).
      return {
        ...(docid && !opts.fullPath && { docid: `#${docid}` }),
        score: Math.round(row.score * 100) / 100,
        file: displayPathFor(row),
        line: snippetInfo.line,
        title: row.title,
        ...(row.context && { context: row.context }),
        ...(body && { body }),
        ...(snippet && { snippet }),
        ...(opts.explain && row.explain && { explain: row.explain }),
      };
    });
    console.log(JSON.stringify(output, null, 2));
  } else if (opts.format === "files") {
    // Simple docid,score,filepath,context output
    for (const row of filtered) {
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : "");
      const ctx = row.context ? `,"${row.context.replace(/"/g, '""')}"` : "";
      if (opts.fullPath) {
        // --full-path: drop the docid, the on-disk path is the identifier.
        console.log(`${row.score.toFixed(2)},${displayPathFor(row)}${ctx}`);
      } else {
        console.log(`#${docid},${row.score.toFixed(2)},${displayPathFor(row)}${ctx}`);
      }
    }
  } else if (opts.format === "cli") {
    const editorUriTemplate = getEditorUriTemplate();
    const linkDb = getDb();

    for (let i = 0; i < filtered.length; i++) {
      const row = filtered[i];
      if (!row) continue;
      const { line, snippet } = extractSnippet(row.body, query, 500, row.chunkPos, row.chunkLen, opts.intent);
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : undefined);

      // Line 1: filepath with docid
      // Default: show the full qmd:// URI so the user can see which collection
      // a hit lives in and can pipe the same string straight back into
      // `qmd get`. A bare collection-relative path like `sources/foo.md` is
      // ambiguous: it's not a real filesystem path, not a URI, and not a
      // shell-friendly identifier on its own.
      // With --full-path the visible label is the file's on-disk path
      // ($PWD-relative when in a subfolder; absolute realpath otherwise),
      // and the docid is omitted because the path is the identifier.
      const virtualPath = toQmdPath(row.displayPath);
      const parsed = parseVirtualPath(virtualPath);
      const absolutePath = resolveVirtualPath(linkDb, virtualPath);
      const visiblePath = displayPathFor(row);

      // Only show :line if we actually found a term match in the snippet body (exclude header line).
      const snippetBody = snippet.split("\n").slice(1).join("\n").toLowerCase();
      const hasMatch = query.toLowerCase().split(/\s+/).some(t => t.length > 0 && snippetBody.includes(t));
      const lineInfo = hasMatch ? `:${line}` : "";
      const docidStr = (docid && !opts.fullPath) ? ` ${c.dim}#${docid}${c.reset}` : "";

      if (process.stdout.isTTY && absolutePath && parsed?.path) {
        const linkLine = hasMatch ? line : 1;
        const linkTarget = buildEditorUri(editorUriTemplate, absolutePath, linkLine, 1);
        const clickable = termLink(`${visiblePath}${lineInfo}`, linkTarget);
        console.log(`${c.cyan}${clickable}${c.reset}${docidStr}`);
      } else {
        console.log(`${c.cyan}${visiblePath}${c.dim}${lineInfo}${c.reset}${docidStr}`);
      }

      // Line 2: Title (if available)
      if (row.title) {
        console.log(`${c.bold}Title: ${row.title}${c.reset}`);
      }

      // Line 3: Context (if available)
      if (row.context) {
        console.log(`${c.dim}Context: ${row.context}${c.reset}`);
      }

      // Line 4: Score
      const score = formatScore(row.score);
      console.log(`Score: ${c.bold}${score}${c.reset}`);
      if (opts.explain && row.explain) {
        const explain = row.explain;
        const ftsScores = explain.ftsScores.length > 0
          ? explain.ftsScores.map(formatExplainNumber).join(", ")
          : "none";
        const vecScores = explain.vectorScores.length > 0
          ? explain.vectorScores.map(formatExplainNumber).join(", ")
          : "none";
        const contribSummary = explain.rrf.contributions
          .slice()
          .sort((a, b) => b.rrfContribution - a.rrfContribution)
          .slice(0, 3)
          .map(c => `${c.source}/${c.queryType}#${c.rank}:${formatExplainNumber(c.rrfContribution)}`)
          .join(" | ");

        console.log(`${c.dim}Explain: fts=[${ftsScores}] vec=[${vecScores}]${c.reset}`);
        console.log(`${c.dim}  RRF: total=${formatExplainNumber(explain.rrf.totalScore)} base=${formatExplainNumber(explain.rrf.baseScore)} bonus=${formatExplainNumber(explain.rrf.topRankBonus)} rank=${explain.rrf.rank}${c.reset}`);
        console.log(`${c.dim}  Blend: ${Math.round(explain.rrf.weight * 100)}%*${formatExplainNumber(explain.rrf.positionScore)} + ${Math.round((1 - explain.rrf.weight) * 100)}%*${formatExplainNumber(explain.rerankScore)} = ${formatExplainNumber(explain.blendedScore)}${c.reset}`);
        if (contribSummary.length > 0) {
          console.log(`${c.dim}  Top RRF contributions: ${contribSummary}${c.reset}`);
        }
      }
      console.log();

      // Snippet with highlighting (diff-style header included)
      const content = opts.full ? row.body : snippet;
      const displayContent = opts.lineNumbers ? addLineNumbers(content, opts.full ? 1 : line) : content;
      const highlighted = highlightTerms(displayContent, query);
      console.log(highlighted);

      // Double empty line between results
      if (i < filtered.length - 1) console.log('\n');
    }
  } else if (opts.format === "md") {
    for (let i = 0; i < filtered.length; i++) {
      const row = filtered[i];
      if (!row) continue;
      const visiblePath = displayPathFor(row);
      const heading = row.title || visiblePath;
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : undefined);
      let content = opts.full ? row.body : extractSnippet(row.body, query, 500, row.chunkPos, row.chunkLen, opts.intent).snippet;
      if (opts.lineNumbers) {
        content = addLineNumbers(content);
      }
      const fileLine = `**file:** \`${visiblePath}\`\n`;
      // With --full-path the on-disk path is the identifier; drop the docid line.
      const docidLine = (docid && !opts.fullPath) ? `**docid:** \`#${docid}\`\n` : "";
      const contextLine = row.context ? `**context:** ${row.context}\n` : "";
      console.log(`---\n# ${heading}\n${fileLine}${docidLine}${contextLine}\n${content}\n`);
    }
  } else if (opts.format === "xml") {
    for (const row of filtered) {
      const titleAttr = row.title ? ` title="${row.title.replace(/"/g, '&quot;')}"` : "";
      const contextAttr = row.context ? ` context="${row.context.replace(/"/g, '&quot;')}"` : "";
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : "");
      let content = opts.full ? row.body : extractSnippet(row.body, query, 500, row.chunkPos, row.chunkLen, opts.intent).snippet;
      if (opts.lineNumbers) {
        content = addLineNumbers(content);
      }
      const docidAttr = opts.fullPath ? "" : ` docid="#${docid}"`;
      console.log(`<file${docidAttr} name="${displayPathFor(row)}"${titleAttr}${contextAttr}>\n${content}\n</file>\n`);
    }
  } else {
    // CSV format
    const csvHeader = opts.fullPath
      ? "score,file,title,context,line,snippet"
      : "docid,score,file,title,context,line,snippet";
    console.log(csvHeader);
    for (const row of filtered) {
      const { line, snippet } = extractSnippet(row.body, query, 500, row.chunkPos, row.chunkLen, opts.intent);
      let content = opts.full ? row.body : snippet;
      if (opts.lineNumbers) {
        content = addLineNumbers(content, opts.full ? 1 : line);
      }
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : "");
      const snippetText = content || "";
      const path = escapeCSV(displayPathFor(row));
      const tail = `${path},${escapeCSV(row.title || "")},${escapeCSV(row.context || "")},${line},${escapeCSV(snippetText)}`;
      if (opts.fullPath) {
        console.log(`${row.score.toFixed(4)},${tail}`);
      } else {
        console.log(`#${docid},${row.score.toFixed(4)},${tail}`);
      }
    }
  }
}

export {
  type OutputOptions,
  printEmptySearchResults,
  buildEditorUri,
  termLink,
  outputResults,
};
