import type { BatchPredictionResponse, ModelMetadata, PredictionResponse, TrackFeatures, TrackOut } from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "Não foi possível falar com a API.");
  }
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json() as Promise<T>;
}

export const searchTracks = (q: string, signal?: AbortSignal) =>
  request<TrackOut[]>(`/v1/tracks/search?q=${encodeURIComponent(q)}&limit=12`, { signal });

export const getTrack = (id: string, signal?: AbortSignal) =>
  request<TrackOut>(`/v1/tracks/${encodeURIComponent(id)}`, { signal });

export const getModelMetadata = (signal?: AbortSignal) => request<ModelMetadata>("/v1/model/metadata", { signal });

// source "catalog": faixa do dataset de treino (navegação); a API não a registra no log de drift.
// useArtistContext=false ignora o histórico do artista: "e se o artista fosse novo?"
export const predictTrack = (
  features: TrackFeatures,
  source: "user" | "catalog" = "user",
  signal?: AbortSignal,
  useArtistContext = true,
) =>
  request<PredictionResponse>("/v1/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features, source, use_artist_context: useArtistContext }),
    signal,
  });

export const predictBatch = (tracks: TrackFeatures[], useArtistContext = true, signal?: AbortSignal) =>
  request<BatchPredictionResponse>("/v1/predict/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tracks, use_artist_context: useArtistContext }),
    signal,
  });
