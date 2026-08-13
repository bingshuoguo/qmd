export function renderPublicCorpusMarkdown(
  docId: string,
  title: string,
  text: string,
): string {
  const normalizedTitle = title.replace(/\s+/gu, " ").trim() || docId;
  const normalizedText = text.replace(/\r\n?/gu, "\n");
  return `# ${normalizedTitle}\n\n${normalizedText}\n`;
}
