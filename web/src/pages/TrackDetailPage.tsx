import { lazy, Suspense, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, getModelMetadata, getTrack, predictTrack } from "../api/client";
import type { ModelMetadata, PredictionResponse, TrackOut } from "../api/types";
import { Chip, VerdictBadge } from "../components/ui/Badge";
import Card from "../components/ui/Card";
import ErrorPanel from "../components/ui/ErrorPanel";
import Tabs from "../components/ui/Tabs";
import FeatureTable from "../features/prediction/FeatureTable";
import { formatValue, label, present, UNIT_FEATURES } from "../features/prediction/labels";
import WhatIfSimulator from "../features/prediction/WhatIfSimulator";

// Plotly é pesado: cada gráfico vira um chunk carregado só nesta tela.
const HitGauge = lazy(() => import("../features/prediction/HitGauge"));
const RadarComparison = lazy(() => import("../features/prediction/RadarComparison"));
const FeatureBars = lazy(() => import("../features/prediction/FeatureBars"));
const ShapWaterfall = lazy(() => import("../features/prediction/ShapWaterfall"));

type Mode = "artist_aware" | "audio_only";

interface Data {
  track: TrackOut;
  aware: PredictionResponse | null; // null quando o artista não tem outras faixas no catálogo
  audio: PredictionResponse;
  meta: ModelMetadata;
}

const ChartFallback = () => <div className="skeleton" style={{ height: 300 }} aria-label="Carregando gráfico" />;

export default function TrackDetailPage() {
  const { id = "" } = useParams();
  return <TrackDetail key={id} id={id} />; // key: estado novo a cada música, sem reset em effect
}

function TrackDetail({ id }: { id: string }) {
  const [data, setData] = useState<Data | null>(null);
  const [mode, setMode] = useState<Mode>("artist_aware");
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    (async () => {
      const track = await getTrack(id, ctrl.signal);
      const hasHistory = track.features.artist_mean_popularity != null;
      const [audio, aware, meta] = await Promise.all([
        predictTrack(track.features, "catalog", ctrl.signal, false),
        hasHistory ? predictTrack(track.features, "catalog", ctrl.signal, true) : Promise.resolve(null),
        getModelMetadata(ctrl.signal),
      ]);
      setData({ track, aware, audio, meta });
      setMode(hasHistory ? "artist_aware" : "audio_only");
      setError(null);
    })().catch((e: unknown) => {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(
        e instanceof ApiError && e.status === 404 ? "Música não encontrada." : "Não foi possível carregar a análise.",
      );
    });
    return () => ctrl.abort();
  }, [id, attempt]);

  if (error)
    return (
      <>
        <ErrorPanel
          message={error}
          onRetry={error === "Música não encontrada." ? undefined : () => setAttempt((a) => a + 1)}
        />
        <Link to="/" className="d-inline-block mt-3">
          Voltar à busca
        </Link>
      </>
    );
  if (!data)
    return (
      <div className="d-grid gap-3" aria-label="Carregando análise">
        <div className="skeleton" style={{ height: 64, maxWidth: 480 }} />
        <div className="skeleton" style={{ height: 320 }} />
      </div>
    );

  const { track, aware, audio, meta } = data;
  const prediction = mode === "artist_aware" && aware ? aware : audio;
  const useArtist = prediction.mode === "artist_aware";
  const f = track.features;
  const genre = f.track_genre;
  const numericRecord: Record<string, number> = Object.fromEntries(
    UNIT_FEATURES.map((k) => [k, f[k as keyof typeof f]]),
  ) as Record<string, number>;

  return (
    <>
      <Link to="/" className="text-muted-r">
        ← Nova busca
      </Link>
      <header className="mt-3 mb-4">
        <h1 className="page-title mb-2" style={{ maxWidth: "24ch" }}>
          {track.name}
        </h1>
        <p className="lede mb-2">{track.artist}</p>
        <div className="d-flex flex-wrap gap-2 align-items-center">
          <Chip>{genre}</Chip>
          <span className="text-muted-r">Popularidade real no Spotify: {track.popularity}</span>
        </div>
        <p className="text-muted-r small mt-2 mb-0" style={{ maxWidth: "60ch" }}>
          Faixa do catálogo do dataset (a maior parte dele foi usada no treino): a previsão aqui tende a ser mais
          precisa do que seria numa demo nova.
        </p>
      </header>

      <div className="mode-switch mb-4" role="group" aria-label="Modalidade da previsão">
        <button
          type="button"
          aria-pressed={useArtist}
          disabled={!aware}
          title={aware ? undefined : "Este artista não tem outras faixas no catálogo"}
          onClick={() => setMode("artist_aware")}
        >
          Com histórico do artista
        </button>
        <button type="button" aria-pressed={!useArtist} onClick={() => setMode("audio_only")}>
          Só áudio (artista novo)
        </button>
      </div>

      <div className="row g-4">
        <div className="col-lg-4">
          <Card>
            <VerdictBadge isHit={prediction.is_hit} />
            <p className="mt-3 mb-0">
              <span className="display-6 fw-semibold">{prediction.percentile_score.toFixed(0)}</span>
              <span className="text-muted-r"> / 100</span>
            </p>
            <p className="text-muted-r small mb-2">
              Percentil previsto dentro de {genre}: o modelo espera que a faixa fique acima de{" "}
              {prediction.percentile_score.toFixed(0)}% das faixas do gênero, o que equivale a uma popularidade perto de{" "}
              <strong className="text-body">{prediction.expected_popularity.toFixed(0)}</strong> (a real é{" "}
              {track.popularity}).
            </p>
            <Suspense fallback={<ChartFallback />}>
              <HitGauge probability={prediction.probability} threshold={prediction.threshold} />
            </Suspense>
            <p className="text-muted-r small mb-0">
              {useArtist
                ? `Usa o histórico do artista (${f.artist_n_tracks ?? "?"} outras faixas, popularidade média ${f.artist_mean_popularity?.toFixed(0)}).`
                : "Usa só as características da faixa, como numa demo de artista novo."}{" "}
              Probabilidade de hit global (popularidade ≥ 60), calibrada para esta modalidade e para o gênero. Corte do
              badge em {(prediction.threshold * 100).toFixed(0)}%. Modelo{" "}
              <span className="text-nowrap">{prediction.model_version}</span>.
            </p>
          </Card>
        </div>
        <div className="col-lg-8">
          <Tabs
            tabs={[
              {
                id: "radar",
                label: "Radar",
                content: (
                  <Suspense fallback={<ChartFallback />}>
                    <RadarComparison track={numericRecord} hitMeans={meta.hit_means} />
                  </Suspense>
                ),
              },
              {
                id: "comparacao",
                label: "Comparação",
                content: (
                  <Suspense fallback={<ChartFallback />}>
                    <FeatureBars track={numericRecord} hitMeans={meta.hit_means} />
                  </Suspense>
                ),
              },
              {
                id: "valores",
                label: "Valores",
                content: <FeatureTable features={present({ ...f })} hitMeans={meta.hit_means} />,
              },
              {
                id: "ease",
                label: "E se…",
                content: <WhatIfSimulator features={f} base={prediction} useArtist={useArtist} />,
              },
              {
                id: "porque",
                label: "Por que essa decisão",
                content: (
                  <>
                    <ul className="factor-list mb-3">
                      {prediction.top_factors.map((fac) => (
                        <li key={fac.feature}>
                          <strong>{label(fac.feature)}</strong> ({formatValue(fac.feature, fac.value)}){" "}
                          {fac.direction === "increases" ? "aumenta" : "diminui"} o percentil previsto em{" "}
                          {Math.abs(fac.shap_value).toFixed(1)} pontos.
                        </li>
                      ))}
                    </ul>
                    <Suspense fallback={<ChartFallback />}>
                      <ShapWaterfall factors={prediction.factors} />
                    </Suspense>
                  </>
                ),
              },
            ]}
          />
        </div>
      </div>
    </>
  );
}
