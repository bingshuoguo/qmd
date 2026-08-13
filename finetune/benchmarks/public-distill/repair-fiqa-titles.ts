#!/usr/bin/env node

import { readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { renderPublicCorpusMarkdown } from "../lib/public-corpus.js";

const corpusRoot = resolve("finetune/data/public-distill-v0/prepared/fiqa-train/corpus");
const documentsPath = resolve("finetune/data/public-distill-v0/prepared/fiqa-train/documents.jsonl");
const documents = readFileSync(documentsPath, "utf8")
  .trim()
  .split("\n")
  .filter(Boolean)
  .map(line => JSON.parse(line) as { doc_id: string; path: string });

let repaired = 0;
let unchanged = 0;
for (const document of documents) {
  const path = join(corpusRoot, document.path);
  const markdown = readFileSync(path, "utf8");
  if (!markdown.startsWith("# \n\n")) {
    unchanged++;
    continue;
  }

  const text = markdown.slice(4, markdown.endsWith("\n") ? -1 : undefined);
  const replacement = renderPublicCorpusMarkdown(document.doc_id, "", text);
  const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.tmp`);
  try {
    writeFileSync(temporary, replacement, "utf8");
    renameSync(temporary, path);
  } finally {
    rmSync(temporary, { force: true });
  }
  repaired++;
}

process.stdout.write(`${JSON.stringify({ corpus_root: corpusRoot, repaired, unchanged }, null, 2)}\n`);
