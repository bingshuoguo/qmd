const ENGLISH_WORD = /[A-Za-z]+(?:['-][A-Za-z]+)*/g;
const sentenceSegmenter = new Intl.Segmenter("en", { granularity: "sentence" });

function englishWordCount(text: string): number {
  return text.match(ENGLISH_WORD)?.length ?? 0;
}

/** Return the longest complete-sentence prefix that has at most maxWords English words. */
export function truncateHydeAtSentenceBoundary(text: string, maxWords: number): string {
  if (!Number.isSafeInteger(maxWords) || maxWords <= 0) {
    throw new Error("maxWords must be a positive integer");
  }
  if (englishWordCount(text) <= maxWords) return text;

  const retained: string[] = [];
  let wordCount = 0;
  for (const { segment } of sentenceSegmenter.segment(text)) {
    const sentence = segment.trim();
    if (!sentence) continue;
    const sentenceWords = englishWordCount(sentence);
    if (wordCount + sentenceWords > maxWords) break;
    retained.push(sentence);
    wordCount += sentenceWords;
  }
  if (retained.length === 0) {
    throw new Error(`no complete sentence fits within ${maxWords} English words`);
  }
  return retained.join(" ");
}
