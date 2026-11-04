import { useRef, useState, type ChangeEvent } from "react";
import { predictBatch } from "../api/client";
import type { BatchPredictionResponse } from "../api/types";
import ErrorPanel from "../components/ui/ErrorPanel";
import { downloadCsv, rankingToCsv } from "../features/ranking/exportCsv";
import { OPTIONAL_COLUMNS, parseDemos, REQUIRED_COLUMNS, type Demo } from "../features/ranking/parseCsv";
import RankingTable from "../features/ranking/RankingTable";

const TOP_K = 50;

type State =
  | { kind: "idle" }
  | { kind: "loading"; file: string }
  | { kind: "error"; message: string; retry?: () => void }
  | { kind: "ok"; file: string; demos: Demo[]; response: BatchPredictionResponse };

export default function BatchRankingPage() {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [ignoreHistory, setIgnoreHistory] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const latest = useRef(0); // só a resposta da requisição mais recente pode atualizar a tela

  async function rank(file: string, demos: Demo[], useArtist: boolean) {
    const id = ++latest.current;
    setState({ kind: "loading", file });
    try {
      const response = await predictBatch(
        demos.map((d) => d.features),
        useArtist,
      );
      if (id === latest.current) setState({ kind: "ok", file, demos, response });
    } catch (err) {
      if (id !== latest.current) return;
      // o CSV já foi lido e validado: tentar de novo não pede o arquivo outra vez, e repete a MESMA modalidade
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : "Erro inesperado.",
        retry: () => void rank(file, demos, useArtist),
      });
    }
  }

  async function onFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const parsed = parseDemos(await file.text());
    if (input.current) input.current.value = ""; // permite reenviar o mesmo arquivo
    if (!parsed.ok) return setState({ kind: "error", message: parsed.error });
    await rank(file.name, parsed.demos, !ignoreHistory);
  }

  const ok = state.kind === "ok" ? state : null;
  const top = ok ? Math.min(TOP_K, ok.response.results.length) : 0;
  // Lote misto: o score tem escalas diferentes nas duas modalidades (com histórico é bem mais espalhado).
  const mixed = ok ? new Set(ok.response.results.map((r) => r.mode)).size > 1 : false;

  return (
    <>
      <h1 className="page-title mb-3">Quais demos merecem uma escuta?</h1>
      <p className="lede mb-4">
        Envie um CSV com as faixas e receba a lista ordenada pelo percentil previsto dentro do gênero. As {TOP_K}{" "}
        primeiras são as candidatas à escuta atenta.
      </p>

      <div className="panel mb-4" style={{ maxWidth: "40rem" }}>
        <label htmlFor="csv" className="form-label text-muted-r">
          Arquivo CSV (até 1000 faixas)
        </label>
        <input id="csv" ref={input} type="file" accept=".csv,text/csv" className="form-control" onChange={onFile} />
        <p className="text-muted-r small mt-3 mb-0">
          Colunas: <code>{REQUIRED_COLUMNS.join(", ")}</code>. Opcionais: <code>name</code>, <code>artist</code> e, para
          melhorar a previsão, <code>{OPTIONAL_COLUMNS.join(", ")}</code>. As colunas <code>artist_*</code> (histórico
          das outras faixas do artista) ligam a modalidade "com histórico"; célula vazia = artista novo.{" "}
          <a href="/exemplo-demos.csv" download>
            Baixar exemplo com 60 faixas (metade com histórico)
          </a>
          .
        </p>
      </div>

      <div className="form-check mb-4">
        <input
          id="ignore-history"
          type="checkbox"
          className="form-check-input"
          checked={ignoreHistory}
          onChange={(e) => {
            setIgnoreHistory(e.target.checked);
            if (ok) void rank(ok.file, ok.demos, !e.target.checked);
          }}
        />
        <label htmlFor="ignore-history" className="form-check-label">
          Ignorar o histórico do artista (avaliar todas como se fossem artistas novos)
        </label>
      </div>

      <div aria-live="polite">
        {state.kind === "loading" && (
          <div className="skeleton" style={{ height: 240 }} aria-label={`Ordenando ${state.file}`} />
        )}
        {state.kind === "error" && <ErrorPanel message={state.message} onRetry={state.retry} />}
        {ok && (
          <>
            <div className="d-flex flex-wrap align-items-center justify-content-between gap-3 mb-3">
              <p className="mb-0">
                <strong>{ok.response.results.length} faixas</strong> ordenadas de <em>{ok.file}</em>.{" "}
                <span className="text-muted-r">
                  Modelo {ok.response.model_version}. O corte abaixo marca o fim do top {top}.
                </span>
              </p>
              <button
                type="button"
                className="btn btn-outline-light btn-sm"
                onClick={() =>
                  downloadCsv(
                    `ranking-${ok.file.replace(/\.csv$/i, "")}.csv`,
                    rankingToCsv(ok.response.results, ok.demos),
                  )
                }
              >
                Exportar CSV
              </button>
            </div>
            {mixed && (
              <p role="note" className="panel small mb-3">
                Este lote mistura faixas <strong>com histórico</strong> e <strong>só áudio</strong>. O score das faixas
                com histórico é bem mais espalhado, então elas tendem a ocupar o topo e o fundo da lista. Para comparar
                todas em pé de igualdade, marque "Ignorar o histórico do artista".
              </p>
            )}
            <RankingTable results={ok.response.results} demos={ok.demos} topK={top} />
          </>
        )}
      </div>
    </>
  );
}
