/**
 * commands/docs.ts - document retrieval and listing commands
 * (get / multi-get / ls) plus the path-rendering helpers they share with
 * output.ts. renderFullPath lives here (not in output.ts) so the dependency
 * direction output → docs stays one-way (spec §8.3 a1).
 */
import { realpathSync, existsSync } from "fs";
import { relative as relativePath } from "path";
import {
  isVirtualPath,
  parseVirtualPath,
  buildVirtualPath,
  resolveVirtualPath,
  findDocument,
  findDocumentRef,
  matchFilesByGlob,
  getContextForPath,
  getDocumentHash,
  getDocumentContent,
  countDocumentsInCollection,
  listDocumentsWithMeta,
  addLineNumbers,
  DEFAULT_MULTI_GET_MAX_BYTES,
} from "../../store.js";
import {
  getCollection as getCollectionFromYaml,
  listCollections as yamlListCollections,
  setConfigIndexName,
} from "../../collections.js";
import { getDb, closeDb, setIndexName } from "../context.js";
import { c, formatBytes } from "../term.js";
import { escapeXml, type OutputFormat } from "../formatter.js";

function renderFullPath(absolutePath: string, cwd: string = process.cwd()): string {
  let real: string;
  try { real = realpathSync(absolutePath); } catch { real = absolutePath; }
  const cwdReal = (() => { try { return realpathSync(cwd); } catch { return cwd; } })();
  if (real === cwdReal) return "./";
  if (real.startsWith(cwdReal + "/")) {
    const rel = relativePath(cwdReal, real);
    if (rel && !rel.startsWith("..")) return `./${rel}`;
  }
  return real;
}

// Compute unique display path for a document
// Always include at least parent folder + filename, add more parent dirs until unique
function computeDisplayPath(
  filepath: string,
  collectionPath: string,
  existingPaths: Set<string>
): string {
  // Get path relative to collection (include collection dir name)
  const collectionDir = collectionPath.replace(/\/$/, '');
  const collectionName = collectionDir.split('/').pop() || '';

  let relativePath: string;
  if (filepath.startsWith(collectionDir + '/')) {
    // filepath is under collection: use collection name + relative path
    relativePath = collectionName + filepath.slice(collectionDir.length);
  } else {
    // Fallback: just use the filepath
    relativePath = filepath;
  }

  const parts = relativePath.split('/').filter(p => p.length > 0);

  // Always include at least parent folder + filename (minimum 2 parts if available)
  // Then add more parent dirs until unique
  const minParts = Math.min(2, parts.length);
  for (let i = parts.length - minParts; i >= 0; i--) {
    const candidate = parts.slice(i).join('/');
    if (!existingPaths.has(candidate)) {
      return candidate;
    }
  }

  // Absolute fallback: use full path (should be unique)
  return filepath;
}

/**
 * Render an absolute filesystem path for human display under --full-path.
 *
 * If the path is the current working directory or a subpath of it, return a
 * "./"-prefixed relative path so it is unambiguously a filesystem path (not a
 * bare collection-relative string that could be confused for a `qmd://`
 * fragment). Otherwise return the absolute realpath so symlinks resolve
 * consistently. Returns `null` if the path could not be normalized — callers
 * fall back to whatever they had before.
 */
function getDocument(filename: string, fromLine?: number, maxLines?: number, lineNumbers?: boolean, fullPath: boolean = false): void {
  // Parse :line suffix from filename. Two forms:
  //   "file.md:100"     -> start at line 100
  //   "file.md:100:40"  -> start at line 100, read 40 lines
  // The :// in virtual paths is never matched because we anchor digits to $.
  // Explicit --from/-l flags always win over values parsed from the path.
  let inputPath = filename;
  const rangeMatch = inputPath.match(/:(\d+):(\d+)$/);
  if (rangeMatch) {
    if (fromLine === undefined) fromLine = parseInt(rangeMatch[1]!, 10);
    if (maxLines === undefined) maxLines = parseInt(rangeMatch[2]!, 10);
    inputPath = inputPath.slice(0, -rangeMatch[0].length);
  } else {
    const colonMatch = inputPath.match(/:(\d+)$/);
    if (colonMatch) {
      const matched = colonMatch[1];
      if (matched) {
        if (fromLine === undefined) fromLine = parseInt(matched, 10);
        inputPath = inputPath.slice(0, -colonMatch[0].length);
      }
    }
  }
  if (fromLine !== undefined) fromLine = Math.max(1, fromLine);

  const parsedIndexPath = isVirtualPath(inputPath) ? parseVirtualPath(inputPath) : null;
  if (parsedIndexPath) {
    if (parsedIndexPath.indexName) {
      setIndexName(parsedIndexPath.indexName);
      setConfigIndexName(parsedIndexPath.indexName);
    }
    inputPath = buildVirtualPath(parsedIndexPath.collectionName, parsedIndexPath.path);
  }

  const db = getDb();
  const doc = findDocument(db, inputPath, { includeBody: true });
  if ("error" in doc) {
    if (doc.error === "excluded_by_ignore") {
      console.error(`Document is excluded by ignore rule: ${filename}`);
      console.error(`Collection: ${doc.collection}`);
      console.error(`Matched path: ${doc.path}`);
      console.error(`Ignore rule: ${doc.rule}`);
    } else {
      console.error(`Document not found: ${filename}`);
      if (doc.similarFiles.length > 0) {
        console.error("Similar files:");
        for (const file of doc.similarFiles) console.error(`  ${file}`);
      }
    }
    closeDb();
    process.exit(1);
  }

  // `findDocument` already computes the docid (first 6 hash chars) and the
  // canonical display path, so we reuse them here instead of a second lookup.
  const docid = doc.docid;
  const canonicalPath = `qmd://${doc.displayPath}`;

  // --full-path: show the on-disk path instead of the qmd:// URL + docid, when
  // the file actually exists. Fall back to the canonical header otherwise.
  let header: string;
  if (fullPath) {
    const fsPath = resolveVirtualPath(db, canonicalPath);
    if (fsPath && existsSync(fsPath)) {
      header = renderFullPath(fsPath);
    } else {
      header = docid ? `${canonicalPath}  #${docid}` : canonicalPath;
    }
  } else {
    header = docid ? `${canonicalPath}  #${docid}` : canonicalPath;
  }

  let output = doc.body || "";
  const startLine = fromLine || 1;

  // Apply line filtering if specified
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = output.split('\n');
    const start = startLine - 1; // Convert to 0-indexed
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    output = lines.slice(start, end).join('\n');
  }

  // Line numbers are on by default (disable with --no-line-numbers) so the
  // model can cite exact lines and request follow-up ranges via path:from:count.
  if (lineNumbers) {
    output = addLineNumbers(output, startLine);
  }

  // Header: identify the document (path + docid, or the on-disk path with
  // --full-path), then optional context.
  console.log(header);
  if (doc.context) {
    console.log(`Folder Context: ${doc.context}`);
  }
  console.log("---\n");
  console.log(output);
  closeDb();
}

// Multi-get: fetch multiple documents by glob pattern or comma-separated list
function multiGet(pattern: string, maxLines?: number, maxBytes: number = DEFAULT_MULTI_GET_MAX_BYTES, format: OutputFormat = "cli", lineNumbers: boolean = true, fullPath: boolean = false): void {
  const db = getDb();

  // Check if it's a comma-separated list or a glob pattern
  const isCommaSeparated = pattern.includes(',') && !pattern.includes('*') && !pattern.includes('?') && !pattern.includes('{');

  let files: { filepath: string; displayPath: string; bodyLength: number; collection?: string; path?: string }[];

  if (isCommaSeparated) {
    // Comma-separated list of files (can be virtual paths or relative paths)
    const names = pattern.split(',').map(s => s.trim()).filter(Boolean);
    files = [];
    for (const name of names) {
      // Resolve qmd:// exact, path exact, then path suffix (store.findDocumentRef).
      const doc = findDocumentRef(db, name);

      if (doc) {
        files.push({
          filepath: doc.virtual_path,
          displayPath: doc.virtual_path,
          bodyLength: doc.body_length,
          collection: doc.collection,
          path: doc.path
        });
      } else {
        console.error(`File not found: ${name}`);
      }
    }
  } else {
    // Glob pattern - matchFilesByGlob now returns virtual paths
    files = matchFilesByGlob(db, pattern).map(f => ({
      ...f,
      collection: undefined,  // Will be fetched later if needed
      path: undefined
    }));
    if (files.length === 0) {
      console.error(`No files matched pattern: ${pattern}`);
      closeDb();
      process.exit(1);
    }
  }

  // Collect results for structured output
  const results: { file: string; displayPath: string; fsPath?: string; docid?: string; title: string; body: string; context: string | null; skipped: boolean; skipReason?: string }[] = [];

  for (const file of files) {
    // Parse virtual path to get collection info if not already available
    let collection = file.collection;
    let path = file.path;

    if (!collection || !path) {
      const parsed = parseVirtualPath(file.filepath);
      if (parsed) {
        collection = parsed.collectionName;
        path = parsed.path;
      }
    }

    // Get context using collection-scoped function
    const context = collection && path ? getContextForPath(db, collection, path) : null;

    // Resolve docid (first 6 chars of content hash) so every entry can be cited.
    const docHash = collection && path ? getDocumentHash(db, collection, path) : null;
    const docid = docHash ? docHash.slice(0, 6) : undefined;

    // --full-path: resolve the on-disk path when it exists (else fall back).
    // Display as ./-prefixed relative path when under $PWD; absolute realpath
    // otherwise. See renderFullPath() for the policy.
    let fsPath: string | undefined;
    if (fullPath) {
      const resolved = resolveVirtualPath(db, file.filepath);
      if (resolved && existsSync(resolved)) fsPath = renderFullPath(resolved);
    }

    // Check size limit
    if (file.bodyLength > maxBytes) {
      results.push({
        file: file.filepath,
        displayPath: file.displayPath,
        fsPath,
        docid,
        title: file.displayPath.split('/').pop() || file.displayPath,
        body: "",
        context,
        skipped: true,
        skipReason: `File too large (${Math.round(file.bodyLength / 1024)}KB > ${Math.round(maxBytes / 1024)}KB). Use 'qmd get ${file.displayPath}' to retrieve.`,
      });
      continue;
    }

    // Fetch document content using collection and path
    if (!collection || !path) continue;

    const doc = getDocumentContent(db, collection, path);

    if (!doc) continue;

    let body = doc.body;

    // Apply line limit if specified
    if (maxLines !== undefined) {
      const lines = body.split('\n');
      body = lines.slice(0, maxLines).join('\n');
      if (lines.length > maxLines) {
        body += `\n\n[... truncated ${lines.length - maxLines} more lines]`;
      }
    }

    // Line numbers on by default (disable with --no-line-numbers).
    if (lineNumbers) {
      body = addLineNumbers(body);
    }

    results.push({
      file: file.filepath,
      displayPath: file.displayPath,
      fsPath,
      docid,
      title: doc.title || file.displayPath.split('/').pop() || file.displayPath,
      body,
      context,
      skipped: false,
    });
  }

  closeDb();

  // --full-path replaces the qmd:// path + docid with the on-disk path (when it
  // resolved). Per result: pick the identifier and whether to show the docid.
  const identOf = (r: typeof results[number]): string => (fullPath && r.fsPath) ? r.fsPath : r.displayPath;
  const docidOf = (r: typeof results[number]): string | undefined => (fullPath && r.fsPath) ? undefined : r.docid;

  // Output based on format
  if (format === "json") {
    const output = results.map(r => {
      const docidVal = docidOf(r);
      return {
        file: identOf(r),
        ...(docidVal && { docid: `#${docidVal}` }),
        title: r.title,
        ...(r.context && { context: r.context }),
        ...(r.skipped ? { skipped: true, reason: r.skipReason } : { body: r.body }),
      };
    });
    console.log(JSON.stringify(output, null, 2));
  } else if (format === "csv") {
    const escapeField = (val: string | null | undefined): string => {
      if (val === null || val === undefined) return "";
      const str = String(val);
      if (str.includes(",") || str.includes('"') || str.includes("\n")) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    };
    console.log("docid,file,title,context,skipped,body");
    for (const r of results) {
      const docidVal = docidOf(r);
      console.log([docidVal ? `#${docidVal}` : "", identOf(r), r.title, r.context, r.skipped ? "true" : "false", r.skipped ? r.skipReason : r.body].map(escapeField).join(","));
    }
  } else if (format === "files") {
    for (const r of results) {
      const docidVal = docidOf(r);
      const id = docidVal ? `#${docidVal} ` : "";
      const ctx = r.context ? `,"${r.context.replace(/"/g, '""')}"` : "";
      const status = r.skipped ? "[SKIPPED]" : "";
      console.log(`${id}${identOf(r)}${ctx}${status ? `,${status}` : ""}`);
    }
  } else if (format === "md") {
    for (const r of results) {
      const docidVal = docidOf(r);
      console.log(`## ${identOf(r)}\n`);
      if (docidVal) console.log(`**docid:** \`#${docidVal}\`\n`);
      if (r.title && r.title !== r.displayPath) console.log(`**Title:** ${r.title}\n`);
      if (r.context) console.log(`**Context:** ${r.context}\n`);
      if (r.skipped) {
        console.log(`> ${r.skipReason}\n`);
      } else {
        console.log("```");
        console.log(r.body);
        console.log("```\n");
      }
    }
  } else if (format === "xml") {
    console.log('<?xml version="1.0" encoding="UTF-8"?>');
    console.log("<documents>");
    for (const r of results) {
      const docidVal = docidOf(r);
      const docidAttr = docidVal ? ` docid="#${docidVal}"` : "";
      console.log(`  <document${docidAttr}>`);
      console.log(`    <file>${escapeXml(identOf(r))}</file>`);
      console.log(`    <title>${escapeXml(r.title)}</title>`);
      if (r.context) console.log(`    <context>${escapeXml(r.context)}</context>`);
      if (r.skipped) {
        console.log(`    <skipped>true</skipped>`);
        console.log(`    <reason>${escapeXml(r.skipReason || "")}</reason>`);
      } else {
        console.log(`    <body>${escapeXml(r.body)}</body>`);
      }
      console.log("  </document>");
    }
    console.log("</documents>");
  } else {
    // CLI format (default)
    for (const r of results) {
      const docidVal = docidOf(r);
      const id = docidVal ? `  #${docidVal}` : "";
      console.log(`\n${'='.repeat(60)}`);
      console.log(`File: ${identOf(r)}${id}`);
      console.log(`${'='.repeat(60)}\n`);

      if (r.skipped) {
        console.log(`[SKIPPED: ${r.skipReason}]`);
        continue;
      }

      if (r.context) {
        console.log(`Folder Context: ${r.context}\n---\n`);
      }
      console.log(r.body);
    }
  }
}

// List files in virtual file tree
function listFiles(pathArg?: string): void {
  const db = getDb();

  if (!pathArg) {
    // No argument - list all collections
    const yamlCollections = yamlListCollections();

    if (yamlCollections.length === 0) {
      console.log("No collections found. Run 'qmd collection add .' to index files.");
      closeDb();
      return;
    }

    // Get file counts from database for each collection (YAML iteration kept so
    // zero-document collections still show "0 files").
    const collections = yamlCollections.map(coll => {
      return {
        name: coll.name,
        file_count: countDocumentsInCollection(db, coll.name)
      };
    });

    console.log(`${c.bold}Collections:${c.reset}\n`);
    for (const coll of collections) {
      console.log(`  ${c.dim}qmd://${c.reset}${c.cyan}${coll.name}/${c.reset}  ${c.dim}(${coll.file_count} files)${c.reset}`);
    }
    closeDb();
    return;
  }

  // Parse the path argument
  let collectionName: string;
  let pathPrefix: string | null = null;

  const afterScheme = pathArg.startsWith('qmd://') ? pathArg.slice('qmd://'.length) : null;
  if (afterScheme !== null && afterScheme.startsWith('/')) {
    // Absolute-path collection: qmd:///Users/foo/bar — normalizeVirtualPath would corrupt
    // this by stripping all leading slashes, so bypass parseVirtualPath entirely.
    const normalized = afterScheme.replace(/\/$/, '');
    const allColls = yamlListCollections();
    const match = allColls
      .filter(c => normalized === c.name || normalized.startsWith(c.name + '/'))
      .sort((a, b) => b.name.length - a.name.length)[0];
    if (match) {
      collectionName = match.name;
      const rest = normalized.slice(match.name.length).replace(/^\//, '');
      pathPrefix = rest || null;
    } else {
      // Preserve the historical qmd:////collection/path alias behavior for normal
      // collections when no absolute-path collection matches.
      const parsed = parseVirtualPath(pathArg);
      if (!parsed) {
        console.error(`Invalid virtual path: ${pathArg}`);
        closeDb();
        process.exit(1);
      }
      collectionName = parsed.collectionName;
      pathPrefix = parsed.path;
    }
  } else if (afterScheme !== null) {
    // Normal virtual path: qmd://collection-name/path
    const parsed = parseVirtualPath(pathArg);
    if (!parsed) {
      console.error(`Invalid virtual path: ${pathArg}`);
      closeDb();
      process.exit(1);
    }
    collectionName = parsed.collectionName;
    pathPrefix = parsed.path;
  } else if (pathArg.startsWith('/')) {
    // Raw absolute filesystem path — longest-prefix match against collection names
    const normalized = pathArg.replace(/\/$/, '');
    const allColls = yamlListCollections();
    const match = allColls
      .filter(c => normalized === c.name || normalized.startsWith(c.name + '/'))
      .sort((a, b) => b.name.length - a.name.length)[0];
    if (match) {
      collectionName = match.name;
      const rest = normalized.slice(match.name.length).replace(/^\//, '');
      pathPrefix = rest || null;
    } else {
      collectionName = normalized;
    }
  } else {
    // Short collection name or name/path
    const parts = pathArg.split('/');
    collectionName = parts[0] || '';
    if (parts.length > 1) {
      pathPrefix = parts.slice(1).join('/');
    }
  }

  // Get the collection
  const coll = getCollectionFromYaml(collectionName);
  if (!coll) {
    console.error(`Collection not found: ${collectionName}`);
    console.error(`Run 'qmd ls' to see available collections.`);
    closeDb();
    process.exit(1);
  }

  // List files in the collection with size and modification time
  const files = listDocumentsWithMeta(db, coll.name, pathPrefix ?? undefined);

  if (files.length === 0) {
    if (pathPrefix) {
      console.log(`No files found under qmd://${collectionName}/${pathPrefix}`);
    } else {
      console.log(`No files found in collection: ${collectionName}`);
    }
    closeDb();
    return;
  }

  // Calculate max widths for alignment
  const maxSize = Math.max(...files.map(f => formatBytes(f.size).length));

  // Output in ls -l style
  for (const file of files) {
    const sizeStr = formatBytes(file.size).padStart(maxSize);
    const date = new Date(file.modified_at);
    const timeStr = formatLsTime(date);

    // Dim the qmd:// prefix, highlight the filename
    console.log(`${sizeStr}  ${timeStr}  ${c.dim}qmd://${collectionName}/${c.reset}${c.cyan}${file.path}${c.reset}`);
  }

  closeDb();
}

// Format date/time like ls -l
function formatLsTime(date: Date): string {
  const now = new Date();
  const sixMonthsAgo = new Date(now.getTime() - 6 * 30 * 24 * 60 * 60 * 1000);

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const month = months[date.getMonth()];
  const day = date.getDate().toString().padStart(2, ' ');

  // If file is older than 6 months, show year instead of time
  if (date < sixMonthsAgo) {
    const year = date.getFullYear();
    return `${month} ${day}  ${year}`;
  } else {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${month} ${day} ${hours}:${minutes}`;
  }
}

export {
  renderFullPath,
  getDocument,
  multiGet,
  listFiles,
};
