import { useState } from 'react';
import { parseCSV } from '../utils/csvParser';
import { normalizeTokens } from '../utils/tokenizer';

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
          text: row.text ?? tokens.join(' ')
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
