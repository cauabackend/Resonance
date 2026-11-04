import type { TrackFeatures } from "../../api/types";

export const MAX_ROWS = 1000;

const NUMERIC = [
  "danceability",
  "energy",
  "valence",
  "tempo",
  "loudness",
  "speechiness",
  "acousticness",
  "instrumentalness",
  "duration_ms",
] as const;
// Opcionais: demais features de áudio e o histórico do artista (liga a modalidade "com histórico").
export const OPTIONAL_COLUMNS = [
  "liveness",
  "key",
  "mode",
  "time_signature",
  "explicit",
  "artist_n_tracks",
  "artist_mean_popularity",
  "artist_hit_share",
];
const INTEGERS = new Set(["key", "mode", "time_signature", "artist_n_tracks"]);
export const REQUIRED_COLUMNS = [...NUMERIC, "track_genre"];

export interface Demo {
  name: string;
  artist: string;
  features: TrackFeatures;
}

/** Parser CSV mínimo (RFC 4180): campos entre aspas, "" escapado e quebras de linha dentro de aspas. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") {
      row.push(cell);
      cell = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      cell = "";
      if (row.some((v) => v !== "")) rows.push(row);
      row = [];
    } else cell += c;
  }
  row.push(cell);
  if (row.some((v) => v !== "")) rows.push(row);
  return rows;
}

export type ParseResult = { ok: true; demos: Demo[] } | { ok: false; error: string };

/** Converte o CSV em faixas; falha com mensagem clara (coluna ausente, valor inválido, lote grande demais). */
export function parseDemos(text: string): ParseResult {
  const [header, ...body] = parseCsv(text.charCodeAt(0) === 0xfeff ? text.slice(1) : text); // remove BOM
  if (!header) return { ok: false, error: "O arquivo está vazio." };
  const cols = header.map((h) => h.trim());
  const missing = REQUIRED_COLUMNS.filter((c) => !cols.includes(c));
  if (missing.length) return { ok: false, error: `Colunas ausentes: ${missing.join(", ")}.` };
  if (body.length === 0) return { ok: false, error: "O arquivo não tem nenhuma faixa." };
  if (body.length > MAX_ROWS)
    return { ok: false, error: `Máximo de ${MAX_ROWS} faixas por lote (o arquivo tem ${body.length}).` };

  const idx = (c: string) => cols.indexOf(c);
  const demos: Demo[] = [];
  for (const [n, r] of body.entries()) {
    const num: Record<string, number> = {};
    for (const c of NUMERIC) {
      const v = Number(r[idx(c)]);
      if (r[idx(c)] === undefined || r[idx(c)].trim() === "" || !Number.isFinite(v))
        return { ok: false, error: `Linha ${n + 2}: valor inválido em "${c}".` };
      num[c] = v;
    }
    const genre = (r[idx("track_genre")] ?? "").trim();
    if (!genre) return { ok: false, error: `Linha ${n + 2}: "track_genre" vazio.` };
    const features = {
      ...num,
      track_genre: genre,
      duration_ms: Math.round(num.duration_ms),
    } as unknown as TrackFeatures;
    const extra = features as unknown as Record<string, unknown>;
    for (const c of OPTIONAL_COLUMNS) {
      const raw = idx(c) >= 0 ? (r[idx(c)] ?? "").trim() : "";
      if (raw === "") continue; // célula vazia = sem a informação (o modelo trata como ausente)
      if (c === "explicit") {
        extra[c] = ["true", "1", "sim", "yes"].includes(raw.toLowerCase());
        continue;
      }
      const v = Number(raw);
      if (!Number.isFinite(v)) return { ok: false, error: `Linha ${n + 2}: valor inválido em "${c}".` };
      extra[c] = INTEGERS.has(c) ? Math.round(v) : v;
    }
    demos.push({
      name: (idx("name") >= 0 ? r[idx("name")] : "")?.trim() || `Faixa ${n + 1}`,
      artist: (idx("artist") >= 0 ? r[idx("artist")] : "")?.trim() ?? "",
      features,
    });
  }
  return { ok: true, demos };
}
