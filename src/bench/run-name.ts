export function validateBenchmarkRunName(value: string): string {
  if (!/^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$/.test(value)) {
    throw new Error(
      `Benchmark run name must be a lowercase safe slug, got ${JSON.stringify(value)}`,
    );
  }
  return value;
}
