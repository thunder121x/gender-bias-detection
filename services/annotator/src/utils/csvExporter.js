const escapeCell = (value) => {
  const str = String(value ?? '');
  if (str.includes(',') || str.includes('"') || /\r?\n/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
};

export const buildCSV = (sentences) => {
  const header = ['id', 'text', 'tokens', 'rationales', 'triggers', 'bias_type'];
  const lines = [header.join(',')];

  sentences.forEach((sentence) => {
    const tokens = sentence.tokens || [];
    const flattenedRationales = (sentence.rationales || []).flatMap(
      (r) => r.spans || []
    );
    const flattenedTriggers = (sentence.rationales || []).flatMap((r) =>
      (r.triggers || []).map((t) => ({ span: t.span }))
    );
    const biasTypes = (sentence.rationales || [])
      .map((r) => r.bias_type)
      .filter(Boolean);

    const row = [
      escapeCell(sentence.id),
      escapeCell(sentence.text ?? tokens.join(' ')),
      escapeCell(JSON.stringify(tokens)),
      escapeCell(JSON.stringify(flattenedRationales)),
      escapeCell(JSON.stringify(flattenedTriggers)),
      escapeCell(JSON.stringify(biasTypes))
    ];
    lines.push(row.join(','));
  });

  return lines.join('\n');
};

export const downloadCSV = (filename, csvString) => {
  const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
