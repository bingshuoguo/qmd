import { describe, expect, test } from "vitest";
import {
  parseDocumentsJsonl,
  parseExpansionsJsonl,
  parseQrelsTsv,
  parseQueriesJsonl,
  validateBenchmarkData,
  validateBenchmarkManifest,
  validateRetrievalProfile,
} from "../src/bench/qrels.js";
import {
  computeConvertedDataSha256,
  normalizeLeakageQuery,
  renderSciFactMarkdown,
  safeExtractZip,
  verifyArchiveMd5,
} from "../finetune/benchmarks/prepare-scifact.js";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const queriesText = [
  JSON.stringify({ qid: "q1", query: "first claim" }),
  JSON.stringify({ qid: "q2", query: "second claim" }),
  "",
].join("\n");
const documentsText = [
  JSON.stringify({ doc_id: "d1", path: "d1.md" }),
  JSON.stringify({ doc_id: "d2", path: "d2.md" }),
  "",
].join("\n");
const qrelsText = [
  "query-id\tcorpus-id\tscore",
  "q1\td1\t1",
  "q2\td2\t1",
  "",
].join("\n");

describe("v2 qrels data", () => {
  test("loads valid queries, qrels, and one-to-one document mapping", () => {
    const queries = parseQueriesJsonl(queriesText);
    const documents = parseDocumentsJsonl(documentsText);
    const qrels = parseQrelsTsv(qrelsText);
    const loaded = validateBenchmarkData(queries, qrels, documents);
    expect(loaded.queryById.get("q1")?.query).toBe("first claim");
    expect(loaded.documentById.get("d1")?.path).toBe("d1.md");
    expect(loaded.relevanceByQuery.get("q2")?.get("d2")).toBe(1);
  });

  test("rejects duplicate query and document IDs", () => {
    expect(() => parseQueriesJsonl(`${queriesText}${JSON.stringify({ qid: "q1", query: "again" })}\n`))
      .toThrow('duplicate qid "q1"');
    expect(() => parseDocumentsJsonl(`${documentsText}${JSON.stringify({ doc_id: "d1", path: "other.md" })}\n`))
      .toThrow('duplicate doc_id "d1"');
  });

  test("rejects non-unique document paths", () => {
    expect(() => parseDocumentsJsonl([
      JSON.stringify({ doc_id: "d1", path: "same.md" }),
      JSON.stringify({ doc_id: "d2", path: "same.md" }),
    ].join("\n"))).toThrow('duplicate path "same.md"');
  });

  test("rejects duplicate and conflicting qrels", () => {
    expect(() => parseQrelsTsv(`${qrelsText}q1\td1\t1\n`))
      .toThrow("duplicate qrel");
    expect(() => parseQrelsTsv(`${qrelsText}q1\td1\t0\n`))
      .toThrow("conflicting qrel");
  });

  test("rejects invalid relevance", () => {
    expect(() => parseQrelsTsv("query-id\tcorpus-id\tscore\nq1\td1\t2\n"))
      .toThrow("relevance must be 0 or 1");
  });

  test("rejects unknown qid and doc_id", () => {
    const queries = parseQueriesJsonl(queriesText);
    const documents = parseDocumentsJsonl(documentsText);
    expect(() => validateBenchmarkData(
      queries,
      parseQrelsTsv("query-id\tcorpus-id\tscore\nmissing\td1\t1\n"),
      documents,
    )).toThrow('unknown qid "missing"');
    expect(() => validateBenchmarkData(
      queries,
      parseQrelsTsv("query-id\tcorpus-id\tscore\nq1\tmissing\t1\n"),
      documents,
    )).toThrow('unknown doc_id "missing"');
  });

  test("rejects a query with no positive qrel", () => {
    const queries = parseQueriesJsonl(queriesText);
    const documents = parseDocumentsJsonl(documentsText);
    const qrels = parseQrelsTsv([
      "query-id\tcorpus-id\tscore",
      "q1\td1\t1",
      "q2\td2\t0",
    ].join("\n"));
    expect(() => validateBenchmarkData(queries, qrels, documents))
      .toThrow('query "q2" has no relevant documents');
  });
});

describe("v2 expansion validation", () => {
  const queries = parseQueriesJsonl(queriesText);
  const valid = [
    JSON.stringify({
      qid: "q1",
      query: "first claim",
      status: "ok",
      raw_output: "lex: first",
      output: [["lex", "first"]],
      fallback_used: false,
      error: null,
    }),
    JSON.stringify({
      qid: "q2",
      query: "second claim",
      status: "format_error",
      raw_output: "bad",
      output: [],
      fallback_used: false,
      error: "invalid format",
    }),
  ].join("\n");

  test("accepts exact coverage and preserves output order", () => {
    const parsed = parseExpansionsJsonl(valid, queries);
    expect(parsed.map(item => item.qid)).toEqual(["q1", "q2"]);
    expect(parsed[0]?.output).toEqual([["lex", "first"]]);
  });

  test("rejects missing, extra, duplicate, and mismatched qids", () => {
    expect(() => parseExpansionsJsonl(valid.split("\n")[0]!, queries))
      .toThrow('missing qid "q2"');
    expect(() => parseExpansionsJsonl(`${valid}\n${valid.split("\n")[0]}`, queries))
      .toThrow('duplicate qid "q1"');
    expect(() => parseExpansionsJsonl(valid.replace('"qid":"q2"', '"qid":"extra"'), queries))
      .toThrow('extra qid "extra"');
    expect(() => parseExpansionsJsonl(valid.replace("second claim", "changed claim"), queries))
      .toThrow('query mismatch for qid "q2"');
  });

  test("rejects illegal type, empty/multiline text, and non-ok output", () => {
    expect(() => parseExpansionsJsonl(valid.replace('"lex","first"', '"intent","first"'), queries))
      .toThrow("invalid expansion type");
    expect(() => parseExpansionsJsonl(valid.replace('"lex","first"', '"lex",""'), queries))
      .toThrow("text must be non-empty");
    expect(() => parseExpansionsJsonl(valid.replace('"lex","first"', '"lex","first\\nsecond"'), queries))
      .toThrow("text must be single-line");
    expect(() => parseExpansionsJsonl(valid.replace('"status":"ok"', '"status":"generation_error"'), queries))
      .toThrow("non-ok expansion must use empty output");
  });

  test("rejects fallback_used on format and generation errors", () => {
    for (const status of ["format_error", "generation_error"]) {
      const invalid = valid
        .replace('"status":"format_error"', `"status":"${status}"`)
        .replace('"fallback_used":false,"error":"invalid format"', '"fallback_used":true,"error":"invalid format"');
      expect(() => parseExpansionsJsonl(invalid, queries))
        .toThrow("non-ok expansion must set fallback_used to false");
    }
  });
});

describe("v2 manifest validation", () => {
  const manifest = {
    benchmark_id: "qmd-expansion-scifact-v1",
    source: {
      url: "https://example.test/scifact.zip",
      archive_md5: "0".repeat(32),
      split: "test",
    },
    source_qrels_sha256: "1".repeat(64),
    excluded_qids_sha256: "2".repeat(64),
    leakage_report_sha256: "3".repeat(64),
    converted_data_sha256: "4".repeat(64),
    qrels: {
      relevant_threshold: 1,
      unjudged: "nonrelevant",
      graded: false,
    },
    cutoffs: [1, 3, 5, 10, 20, 30],
    metrics: ["recall_at_cutoffs", "mrr_at_10", "ndcg_at_10"],
  };

  test("accepts the frozen benchmark ID and metrics", () => {
    expect(validateBenchmarkManifest(manifest).benchmark_id)
      .toBe("qmd-expansion-scifact-v1");
  });

  test("rejects benchmark ID mismatch, unknown metrics, and invalid cutoffs", () => {
    expect(() => validateBenchmarkManifest({ ...manifest, benchmark_id: "other" }))
      .toThrow("benchmark_id");
    expect(() => validateBenchmarkManifest({ ...manifest, metrics: ["precision_at_10"] }))
      .toThrow("unknown metric");
    expect(() => validateBenchmarkManifest({ ...manifest, cutoffs: [0, 10] }))
      .toThrow("invalid cutoff");
    expect(() => validateBenchmarkManifest({ ...manifest, cutoffs: [10, 10] }))
      .toThrow("duplicate cutoff");
  });
});

describe("retrieval profile validation", () => {
  const profile = {
    profile_id: "qmd-scifact-controlled-v1",
    collection_name: "qmd-expansion-scifact-v1",
    collection_root: "corpus",
    embedding_model: "embedding",
    reranker_model: "reranker",
    result_limit: 30,
    per_list_limit: 30,
    candidate_limit: 40,
    rerank: true,
    auto_expand: false,
    strong_signal_bypass: false,
  };

  test("accepts the frozen controlled retrieval profile", () => {
    expect(validateRetrievalProfile(profile, [1, 3, 30]).profile_id)
      .toBe("qmd-scifact-controlled-v1");
  });

  test("rejects insufficient limits and uncontrolled expansion behavior", () => {
    expect(() => validateRetrievalProfile({ ...profile, result_limit: 20 }, [30]))
      .toThrow("result_limit");
    expect(() => validateRetrievalProfile({ ...profile, per_list_limit: 29 }, [30]))
      .toThrow("per_list_limit");
    expect(() => validateRetrievalProfile({ ...profile, candidate_limit: 29 }, [30]))
      .toThrow("candidate_limit");
    expect(() => validateRetrievalProfile({ ...profile, auto_expand: true }, [30]))
      .toThrow("auto_expand");
    expect(() => validateRetrievalProfile({ ...profile, strong_signal_bypass: true }, [30]))
      .toThrow("strong_signal_bypass");
  });
});

function storedZip(name: string, content = "content"): Buffer {
  const nameBytes = Buffer.from(name);
  const contentBytes = Buffer.from(content);
  const local = Buffer.alloc(30);
  local.writeUInt32LE(0x04034b50, 0);
  local.writeUInt16LE(20, 4);
  local.writeUInt32LE(contentBytes.length, 18);
  local.writeUInt32LE(contentBytes.length, 22);
  local.writeUInt16LE(nameBytes.length, 26);
  const central = Buffer.alloc(46);
  central.writeUInt32LE(0x02014b50, 0);
  central.writeUInt16LE(20, 4);
  central.writeUInt16LE(20, 6);
  central.writeUInt32LE(contentBytes.length, 20);
  central.writeUInt32LE(contentBytes.length, 24);
  central.writeUInt16LE(nameBytes.length, 28);
  const centralOffset = local.length + nameBytes.length + contentBytes.length;
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(1, 8);
  eocd.writeUInt16LE(1, 10);
  eocd.writeUInt32LE(central.length + nameBytes.length, 12);
  eocd.writeUInt32LE(centralOffset, 16);
  return Buffer.concat([local, nameBytes, contentBytes, central, nameBytes, eocd]);
}

describe("SciFact preparation primitives", () => {
  test("rejects archive MD5 mismatch", () => {
    expect(() => verifyArchiveMd5(Buffer.from("not-scifact")))
      .toThrow("archive MD5 mismatch");
  });

  test("rejects ZIP path traversal", () => {
    const root = mkdtempSync(join(tmpdir(), "qmd-scifact-zip-"));
    try {
      expect(() => safeExtractZip(storedZip("../escape.txt"), root))
        .toThrow("path traversal");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("extracts a safe ZIP entry", () => {
    const root = mkdtempSync(join(tmpdir(), "qmd-scifact-zip-"));
    try {
      safeExtractZip(storedZip("scifact/corpus.jsonl", "safe"), root);
      expect(readFileSync(join(root, "scifact", "corpus.jsonl"), "utf8")).toBe("safe");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("renders the frozen Markdown template with LF endings", () => {
    expect(renderSciFactMarkdown("Title\r\nLine", "Text\r\nbody"))
      .toBe("# Title\nLine\n\nText\nbody\n");
  });

  test("normalizes leakage queries and removes only prefixes", () => {
    expect(normalizeLeakageQuery("  /only:HyDe  Café   FACTS  "))
      .toBe("café facts");
  });

  test("aggregate hash is deterministic and content-sensitive", () => {
    const createFixture = (): string => {
      const root = mkdtempSync(join(tmpdir(), "qmd-scifact-hash-"));
      mkdirSync(join(root, "corpus"));
      writeFileSync(join(root, "corpus", "2.md"), "# Two\n\nText\n");
      writeFileSync(join(root, "corpus", "1.md"), "# One\n\nText\n");
      writeFileSync(join(root, "queries.jsonl"), '{"qid":"q","query":"claim"}\n');
      writeFileSync(join(root, "qrels.tsv"), "query-id\tcorpus-id\tscore\nq\t1\t1\n");
      writeFileSync(join(root, "documents.jsonl"), '{"doc_id":"1","path":"1.md"}\n');
      return root;
    };
    const first = createFixture();
    const second = createFixture();
    try {
      expect(computeConvertedDataSha256(first)).toBe(computeConvertedDataSha256(second));
      writeFileSync(join(second, "corpus", "1.md"), "# One\n\nChanged\n");
      expect(computeConvertedDataSha256(first)).not.toBe(computeConvertedDataSha256(second));
    } finally {
      rmSync(first, { recursive: true, force: true });
      rmSync(second, { recursive: true, force: true });
    }
  });
});
