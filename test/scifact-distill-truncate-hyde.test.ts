import { describe, expect, test } from "vitest";

import { truncateHydeAtSentenceBoundary } from "../finetune/benchmarks/distill/hyde-truncation.js";

describe("truncateHydeAtSentenceBoundary", () => {
  test("keeps the longest complete-sentence prefix within the English-word limit", () => {
    const hyde = [
      "Alpha beta gamma.",
      "Delta epsilon zeta eta.",
      "Theta iota kappa lambda mu.",
    ].join(" ");

    expect(truncateHydeAtSentenceBoundary(hyde, 7)).toBe(
      "Alpha beta gamma. Delta epsilon zeta eta.",
    );
  });

  test("does not change a passage already within the limit", () => {
    const hyde = "Alpha beta gamma. Delta epsilon.";

    expect(truncateHydeAtSentenceBoundary(hyde, 70)).toBe(hyde);
  });
});
