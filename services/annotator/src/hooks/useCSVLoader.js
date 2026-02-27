import { useState } from 'react';
import { parseCSV } from '../utils/csvParser';
import { normalizeTokens } from '../utils/tokenizer';

const safeJSONParse = (value, fallback = []) => {
  if (value == null) return fallback;
  if (Array.isArray(value)) return value;
  const text = String(value).trim();
  if (!text) return fallback;
  const attempts = [
    text,
    text.replace(/""/g, '"'),
    text.replace(/\\"/g, '"'),
    text.replace(/'/g, '"'),
    text.replace(/\]\]$/, ']'),
    text.replace(/\]\]\]$/, ']]')
  ];
  for (const attempt of attempts) {
    try {
      const parsed = JSON.parse(attempt);
      return parsed;
    } catch (err) {
      // continue
    }
  }
  return fallback;
};

const toIndexList = (entry) => {
  if (!Array.isArray(entry)) return [];
  const normalized = entry.map((n) => Number(n)).filter((n) => Number.isFinite(n));
  if (normalized.length === 2) {
    const [a, b] = normalized;
    const start = Math.min(a, b);
    const end = Math.max(a, b);
    const list = [];
    for (let i = start; i <= end; i += 1) list.push(i);
    return list;
  }
  return Array.from(new Set(normalized)).sort((a, b) => a - b);
};

const normalizeDecisionRule = (value) => {
  if (!value) return [];
  if (Array.isArray(value)) return value.map((x) => String(x));
  return [String(value)];
};

const mapExistingRationales = (row) => {
  const rationalesRaw = safeJSONParse(row.wa_rationales ?? row.rationales, []);
  const triggersRaw = safeJSONParse(row.wa_triggers ?? row.triggers, []);
  const labelTypeRaw = safeJSONParse(row.wa_label_type ?? row.label_type, []);
  const decisionRuleRaw = safeJSONParse(
    row.wa_decision_rule ?? row.decision_rule,
    []
  );

  const rationaleLists = Array.isArray(rationalesRaw) ? rationalesRaw : [];
  const triggerLists = Array.isArray(triggersRaw) ? triggersRaw : [];
  const labelList = Array.isArray(labelTypeRaw) ? labelTypeRaw : [];
  const ruleList = Array.isArray(decisionRuleRaw) ? decisionRuleRaw : [];

  const count = Math.max(
    rationaleLists.length,
    triggerLists.length,
    labelList.length,
    ruleList.length
  );

  if (count === 0) return [];

  return Array.from({ length: count }, (_, idx) => ({
    id: `R${idx}`,
    label_type: labelList[idx] ?? null,
    spans: [toIndexList(rationaleLists[idx])],
    triggers: [toIndexList(triggerLists[idx])],
    decision_rule: normalizeDecisionRule(ruleList[idx])
  })).filter(
    (r) =>
      r.label_type ||
      (r.spans[0] && r.spans[0].length > 0) ||
      (r.triggers[0] && r.triggers[0].length > 0) ||
      r.decision_rule.length > 0
  );
};

export const useCSVLoader = () => {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const parseAndStore = (text) => {
    try {
      const parsed = parseCSV(text);
      if (!parsed.length) {
        throw new Error('No rows found in CSV.');
      }
      const findTextColumn = (row) => {
        return (
          row.tokens ||
          row.text ||
          row.raw_text ||
          row.content ||
          row.message ||
          row.body ||
          row.Text || // uppercase
          row["raw text"] ||
          row["text "] || // trimmed
          null
        );
      };

      if (!parsed[0].id) {
        throw new Error('CSV must include "id" column.');
      }

      if (!findTextColumn(parsed[0])) {
        throw new Error(
          'CSV must contain a text column such as "text", "raw_text", or "tokens".'
        );
      }
      const normalized = parsed.map((row, idx) => {
        const tokens = normalizeTokens(row.tokens ?? row.text ?? '');
        return {
          id: row.id ?? `row-${idx + 1}`,
          tokens,
          text: row.text ?? tokens.join(' '),
          rationales: mapExistingRationales(row)
        };
      });
      setRows(normalized);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to parse CSV.');
      setRows([]);
    }
  };

  const loadFile = (file) => {
    if (!file) return;
    setLoading(true);
    const reader = new FileReader();
    reader.onload = (evt) => {
      parseAndStore(evt.target.result);
      setLoading(false);
    };
    reader.onerror = () => {
      setError('Failed to read file.');
      setLoading(false);
    };
    reader.readAsText(file);
  };

  const reset = () => {
    setRows([]);
    setError(null);
  };

  return { rows, error, loading, loadFile, reset };
};
