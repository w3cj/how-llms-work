/**
 * Cosine similarity — measures the angle between two vectors in embedding space.
 *
 * Returns a value from -1 to 1: 1 means identical direction, 0 means perpendicular
 * (unrelated), -1 means opposite. This is how we measure "closeness" in embedding
 * space — words with similar meanings have vectors pointing in similar directions.
 *
 * @example
 * cosineSimilarity([1, 0], [1, 0])   // => 1.0 (identical)
 * cosineSimilarity([1, 0], [0, 1])   // => 0.0 (perpendicular / unrelated)
 * cosineSimilarity([1, 0], [-1, 0])  // => -1.0 (opposite)
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}
