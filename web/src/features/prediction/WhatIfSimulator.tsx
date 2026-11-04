import { useEffect, useState } from "react";
import { predictTrack } from "../../api/client";
import type { PredictionResponse, TrackFeatures } from "../../api/types";
import { formatValue, label } from "./labels";

interface Slider {
  key: "energy" | "danceability" | "valence" | "tempo";
  min: number;
  max: number;
  step: number;
}

const SLIDERS: Slider[] = [
  { key: "danceability", min: 0, max: 1, step: 0.01 },
  { key: "energy", min: 0, max: 1, step: 0.01 },
  { key: "valence", min: 0, max: 1, step: 0.01 },
  { key: "tempo", min: 30, max: 300, step: 1 },
];

interface Props {
  features: TrackFeatures;
  base: PredictionResponse;
  useArtist?: boolean; // mantém a modalidade escolhida na página
}

/** "E se": ajusta features e recalcula o score na API (debounce), mostrando a diferença para a faixa original. */
export default function WhatIfSimulator({ features, base, useArtist = true }: Props) {
  const [values, setValues] = useState<Record<string, number>>(() =>
    Object.fromEntries(SLIDERS.map((s) => [s.key, features[s.key]])),
  );
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState(false);

  const changed = SLIDERS.some((s) => values[s.key] !== features[s.key]);

  useEffect(() => {
    if (!changed) return; // sem alteração não há o que recalcular (e evita um POST ao montar)
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      predictTrack({ ...features, ...values }, "catalog", ctrl.signal, useArtist)
        .then((r) => {
          setResult(r);
          setError(false);
        })
        .catch((e: unknown) => {
          if (!(e instanceof DOMException && e.name === "AbortError")) setError(true);
        });
    }, 250);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [values, features, useArtist, changed]);

  const current = changed ? (result ?? base) : base;
  const delta = current.percentile_score - base.percentile_score;

  return (
    <div>
      <p className="text-muted-r">
        Mexa nas características e veja o percentil previsto mudar. As mudanças não alteram a faixa do catálogo.
      </p>
      <div className="row g-4 align-items-center">
        <div className="col-md-7 d-grid gap-3">
          {SLIDERS.map((s) => (
            <div key={s.key}>
              <label htmlFor={`whatif-${s.key}`} className="d-flex justify-content-between mb-1">
                <span>{label(s.key)}</span>
                <span className="text-muted-r">{formatValue(s.key, values[s.key])}</span>
              </label>
              <input
                id={`whatif-${s.key}`}
                type="range"
                className="form-range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={values[s.key]}
                onChange={(e) => setValues((v) => ({ ...v, [s.key]: Number(e.target.value) }))}
              />
            </div>
          ))}
          <button
            type="button"
            className="btn btn-outline-light btn-sm align-self-start"
            disabled={!changed}
            onClick={() => setValues(Object.fromEntries(SLIDERS.map((s) => [s.key, features[s.key]])))}
          >
            Voltar aos valores originais
          </button>
        </div>
        <div className="col-md-5" aria-live="polite">
          <p className="mb-0">
            <span className="display-5 fw-semibold">{current.percentile_score.toFixed(1)}</span>
            <span className="text-muted-r"> / 100</span>
          </p>
          <p className="text-muted-r mb-1">
            {changed
              ? `${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(1)} pontos vs. a faixa original (${base.percentile_score.toFixed(1)})`
              : "Valores originais da faixa"}
          </p>
          {error && <p className="small mb-0">Não foi possível recalcular agora.</p>}
        </div>
      </div>
    </div>
  );
}
