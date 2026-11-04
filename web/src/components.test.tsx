import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PredictionResponse, TrackFeatures } from "./api/types";
import ContrastToggle from "./components/ContrastToggle";
import ErrorPanel from "./components/ui/ErrorPanel";
import WhatIfSimulator from "./features/prediction/WhatIfSimulator";

const features: TrackFeatures = {
  danceability: 0.5,
  energy: 0.5,
  valence: 0.5,
  tempo: 100,
  loudness: -5,
  speechiness: 0.1,
  acousticness: 0.1,
  instrumentalness: 0,
  duration_ms: 200000,
  track_genre: "pop",
};

const prediction = (score: number): PredictionResponse => ({
  percentile_score: score,
  mode: "audio_only",
  expected_popularity: 40,
  is_hit: false,
  probability: 0.1,
  threshold: 0.17,
  model_version: "v1",
  top_factors: [],
  factors: [],
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("ErrorPanel", () => {
  it("oferece nova tentativa só quando há o que repetir", () => {
    const retry = vi.fn();
    const { rerender } = render(<ErrorPanel message="API fora do ar" onRetry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("API fora do ar");
    fireEvent.click(screen.getByRole("button", { name: "Tentar de novo" }));
    expect(retry).toHaveBeenCalledOnce();
    rerender(<ErrorPanel message="Não encontrada" />);
    expect(screen.queryByRole("button")).toBeNull();
  });
});

describe("ContrastToggle", () => {
  it("alterna o atributo no <html> e persiste a escolha", () => {
    render(<ContrastToggle />);
    const btn = screen.getByRole("button", { name: "Alto contraste" });
    expect(btn).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(btn);
    expect(btn).toHaveAttribute("aria-pressed", "true");
    expect(document.documentElement.dataset.contrast).toBe("high");
    expect(localStorage.getItem("resonance-contrast")).toBe("high");
    fireEvent.click(btn);
    expect(document.documentElement.dataset.contrast).toBe("normal");
  });
});

describe("WhatIfSimulator", () => {
  it("recalcula na API ao mexer num slider e mostra a diferença para a faixa original", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => prediction(58) });
    vi.stubGlobal("fetch", fetchMock);
    render(<WhatIfSimulator features={features} base={prediction(50)} />);
    expect(screen.getByRole("button", { name: /voltar aos valores originais/i })).toBeDisabled();
    expect(screen.getByText("Valores originais da faixa")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Dançabilidade/), { target: { value: "0.9" } });
    await waitFor(() => expect(screen.getByText(/\+8,0|\+8\.0/)).toBeInTheDocument(), { timeout: 2000 });
    const body = JSON.parse(fetchMock.mock.calls.at(-1)![1].body);
    expect(body.features.danceability).toBe(0.9);
    expect(body.source).toBe("catalog"); // simulação não polui o log de drift
    expect(screen.getByRole("button", { name: /voltar aos valores originais/i })).toBeEnabled();
  });
});
