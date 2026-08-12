#!/usr/bin/env node

/**
 * SciFact benchmark converter. All machinery lives in ../lib/beir.ts; this
 * file only supplies the SciFact dataset config and a CLI bootstrap.
 *
 * SciFact qrels are already binary (score 1), so relevantThreshold 1 makes the
 * binarization an identity and the emitted bytes match the frozen pipeline.
 */

import {
  isMainModule,
  runBeirCli,
  type BeirDatasetConfig,
} from "../lib/beir.js";

export const SCIFACT_CONFIG: BeirDatasetConfig = {
  benchmarkId: "qmd-expansion-scifact-v1",
  sourceUrl: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
  archiveMd5: "5f7d1de60b170fc8027bb7898e2efca1",
  relevantThreshold: 1,
};

if (isMainModule(import.meta.url)) {
  runBeirCli(SCIFACT_CONFIG, process.argv.slice(2));
}
