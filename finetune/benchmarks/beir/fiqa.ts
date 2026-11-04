#!/usr/bin/env node

/**
 * FiQA-2018 benchmark converter. All machinery lives in ../lib/beir.ts; this
 * file only supplies the FiQA dataset config and a CLI bootstrap.
 *
 * FiQA qrels are GRADED (relevance scores 2 and 3). relevantThreshold 2 folds
 * every graded relevant pair down to the binary 1 the QMD bench contract
 * requires; source-qrels.tsv keeps the raw graded scores for provenance.
 */

import {
  isMainModule,
  runBeirCli,
  type BeirDatasetConfig,
} from "../lib/beir.js";

export const FIQA_CONFIG: BeirDatasetConfig = {
  benchmarkId: "qmd-expansion-fiqa-v1",
  sourceUrl: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
  archiveMd5: "17918ed23cd04fb15047f73e6c3bd9d9",
  relevantThreshold: 2,
};

if (isMainModule(import.meta.url)) {
  runBeirCli(FIQA_CONFIG, process.argv.slice(2));
}
