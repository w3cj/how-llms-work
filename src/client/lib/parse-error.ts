/**
 * Extracts a human-readable error message from a failed HTTP response.
 * Handles nested JSON error structures (like Zod validation errors from Hono)
 * and falls back to raw response text.
 */
export async function parseError(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const json = JSON.parse(text);
    if (json.error?.message) {
      const parsed = JSON.parse(json.error.message);
      if (Array.isArray(parsed)) {
        return parsed.map((e: { message?: string }) => e.message).filter(Boolean).join(", ");
      }
      return json.error.message;
    }
    return text;
  }
  catch {
    return text;
  }
}
