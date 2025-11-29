export const normalizeTokens = (input) => {
  const cleanToken = (token) => {
    let t = String(token || '').trim();
    if (t.startsWith('[') && t.length > 1) t = t.slice(1);
    if (t.endsWith(']') && t.length > 1) t = t.slice(0, -1);
    t = t.replace(/(^,+|,+$)/g, '');
    return t;
  };

  const tryParseJSON = (value) => {
    try {
      return JSON.parse(value);
    } catch (err) {
      return null;
    }
  };

  const parseLooseList = (value) => {
    const stripped = value.replace(/^\s*\[+/, "").replace(/\]+$/, "");
    return stripped
      .split(",")
      .map((t) => t.replace(/"/g, "").trim()) // remove quotes + trim
      .map((t) => t.replace(/,$/, "")) // <-- remove trailing comma from token
      .filter((t) => t.length > 0);
  };

  const mapClean = (tokens) => tokens.map((t) => cleanToken(t));

  if (Array.isArray(input)) {
    return mapClean(input);
  }
  if (typeof input === 'string') {
    const trimmed = input.trim();
    if (!trimmed) return [];
    const parsed =
      tryParseJSON(trimmed) ||
      tryParseJSON(trimmed.replace(/""/g, '"')) ||
      tryParseJSON(trimmed.replace(/\\"/g, '"'));
    if (Array.isArray(parsed)) {
      return mapClean(parsed);
    }
    const loose = parseLooseList(trimmed);
    if (loose.length) return mapClean(loose);
    return trimmed.split(/\s+/).map((t) => cleanToken(t));
  }
  return [];
};
