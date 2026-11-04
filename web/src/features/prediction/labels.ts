export const FEATURE_LABELS: Record<string, string> = {
  danceability: "Dançabilidade",
  energy: "Energia",
  valence: "Positividade",
  tempo: "BPM",
  loudness: "Volume (dB)",
  speechiness: "Presença vocal",
  acousticness: "Acústico",
  instrumentalness: "Instrumental",
  duration_ms: "Duração",
  track_genre: "Gênero",
  liveness: "Ao vivo",
  key: "Tom",
  mode: "Modo",
  time_signature: "Compasso",
  explicit: "Letra explícita",
  energy_x_danceability: "Energia × dançabilidade",
  artist_context: "Histórico do artista",
  artist_n_tracks: "Outras faixas do artista",
  artist_mean_popularity: "Popularidade média do artista",
  artist_hit_share: "Fração de hits do artista",
};

export const label = (f: string) => FEATURE_LABELS[f] ?? f;

// Features em escala 0-1: as únicas comparáveis lado a lado no radar e nas barras.
export const UNIT_FEATURES = ["danceability", "energy", "valence", "speechiness", "acousticness", "instrumentalness"];

export type Value = number | string | boolean | null | undefined;

/** Só as entradas que existem (opcionais ausentes chegam como null/undefined). */
export function present(features: Record<string, Value>): Record<string, number | string | boolean> {
  return Object.fromEntries(Object.entries(features).filter(([, v]) => v !== null && v !== undefined)) as Record<
    string,
    number | string | boolean
  >;
}

export function formatValue(feature: string, value: number | string | boolean): string {
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "Sim" : "Não";
  if (feature === "duration_ms") {
    const s = Math.round(value / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
  if (feature === "tempo") return `${value.toFixed(0)} BPM`;
  if (feature === "loudness") return `${value.toFixed(1)} dB`;
  if (feature === "mode") return value === 1 ? "Maior" : "Menor";
  if (["key", "time_signature", "artist_n_tracks"].includes(feature)) return String(Math.round(value));
  if (feature === "artist_mean_popularity" || feature === "artist_context") return `${value.toFixed(0)} de 100`;
  if (feature === "artist_hit_share") return `${(value * 100).toFixed(0)}%`;
  return value.toFixed(2);
}
