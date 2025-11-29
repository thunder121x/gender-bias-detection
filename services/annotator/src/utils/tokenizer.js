export const normalizeTokens = (input) => {
  if (Array.isArray(input)) {
    return input.map((t) => String(t));
  }
  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (!trimmed) return [];
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        return parsed.map((t) => String(t));
      }
    } catch (err) {
      // Fall through to whitespace split.
    }
    return trimmed.split(/\s+/).map((t) => String(t));
  }
  return [];
};
