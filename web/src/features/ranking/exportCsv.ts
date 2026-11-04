import type { BatchItem } from "../../api/types";
import type { Demo } from "./parseCsv";

const cell = (v: string | number | boolean) => {
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** Lote ordenado em CSV (rank, nome, artista, gênero, percentil, probabilidade, badge). */
export function rankingToCsv(results: BatchItem[], demos: Demo[]): string {
  const rows = results.map((r) => {
    const d = demos[r.input_index];
    return [
      r.rank,
      d.name,
      d.artist,
      d.features.track_genre,
      r.mode,
      r.percentile_score.toFixed(2),
      r.expected_popularity.toFixed(1),
      r.probability.toFixed(4),
      r.is_hit,
    ];
  });
  return [
    [
      "rank",
      "name",
      "artist",
      "track_genre",
      "mode",
      "percentile_score",
      "expected_popularity",
      "probability",
      "is_hit",
    ],
    ...rows,
  ]
    .map((row) => row.map(cell).join(","))
    .join("\n");
}

export function downloadCsv(filename: string, csv: string) {
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}
