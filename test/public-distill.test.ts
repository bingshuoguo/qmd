import { describe, expect, test } from "vitest";
import { comparePublicMetrics, publicTieIndex, selectPublicCandidate } from "../finetune/benchmarks/lib/public-distill.js";
import { renderPublicCorpusMarkdown } from "../finetune/benchmarks/lib/public-corpus.js";

const metrics = (recall: number, ndcg: number, mrr: number) => ({
  recall_at_10: recall,
  recall_at_20: recall,
  recall_at_30: recall,
  ndcg_at_10: ndcg,
  mrr_at_10: mrr,
});

describe("public distillation selection", () => {
  test("orders Recall@10 before nDCG@10 and MRR@10", () => {
    expect(comparePublicMetrics(metrics(1, 0, 0), metrics(0, 1, 1))).toBe(1);
    expect(comparePublicMetrics(metrics(1, 0.5, 0), metrics(1, 0.4, 1))).toBe(1);
    expect(comparePublicMetrics(metrics(1, 0.5, 0.5), metrics(1, 0.5, 0.4))).toBe(1);
  });

  test("retains qualified ties and uses stable hash tie break", () => {
    const raw = metrics(1, 1, 1);
    const candidates = [0, 1, 2, 3].map(candidate_index => ({ candidate_index, metrics: raw }));
    const result = selectPublicCandidate({ sourceId: "fiqa-train", qid: "42", raw, candidates });
    expect(result.selection_status).toBe("qualified_tie");
    expect(result.selected_candidate_index).toBe(publicTieIndex("fiqa-train", "42", 4));
  });

  test("distinguishes winner, no winner, and no valid candidate", () => {
    expect(selectPublicCandidate({
      sourceId: "s", qid: "q", raw: metrics(0, 0, 0),
      candidates: [{ candidate_index: 0, metrics: metrics(1, 0, 0) }],
    }).selection_status).toBe("winner");
    expect(selectPublicCandidate({
      sourceId: "s", qid: "q", raw: metrics(1, 1, 1),
      candidates: [{ candidate_index: 0, metrics: metrics(0, 0, 0) }],
    }).selection_status).toBe("no_winner");
    expect(selectPublicCandidate({
      sourceId: "s", qid: "q", raw: metrics(1, 1, 1), candidates: [],
    }).selection_status).toBe("no_valid_candidate");
  });
});

describe("public corpus conversion", () => {
  test("uses the document id when the source title is empty", () => {
    expect(renderPublicCorpusMarkdown("418057", "", "Body text"))
      .toBe("# 418057\n\nBody text\n");
  });

  test("normalizes source titles to one line", () => {
    expect(renderPublicCorpusMarkdown("42", " First\n second ", "Body\r\ntext"))
      .toBe("# First second\n\nBody\ntext\n");
  });
});
