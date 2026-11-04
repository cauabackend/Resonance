import { useEffect, useState } from "react";
import { getModelMetadata } from "../api/client";
import type { ModelMetadata } from "../api/types";
import ErrorPanel from "../components/ui/ErrorPanel";

const f = (n: number | undefined, d = 3) => (n === undefined ? "—" : n.toFixed(d).replace(".", ","));

export default function ModelCardPage() {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    getModelMetadata(ctrl.signal)
      .then((m) => {
        setMeta(m);
        setFailed(false);
      })
      .catch((e: unknown) => {
        if (!(e instanceof DOMException && e.name === "AbortError")) setFailed(true);
      });
    return () => ctrl.abort();
  }, [attempt]);

  const m = meta?.metrics ?? {};
  const k = m.ndcg_k ?? 50;
  return (
    <>
      <h1 className="page-title mb-3">Model card</h1>
      <p className="lede mb-5">
        O que este modelo faz, como foi medido e onde ele não deve ser usado. As métricas vêm do modelo em produção, não
        deste texto.
      </p>

      {failed && (
        <ErrorPanel
          message="Não foi possível carregar as métricas do modelo."
          onRetry={() => setAttempt((a) => a + 1)}
        />
      )}
      {!meta && !failed && <div className="skeleton" style={{ height: 240 }} aria-label="Carregando métricas" />}

      {meta && (
        <section className="panel mb-4" aria-labelledby="mc-metrics">
          <h2 id="mc-metrics" className="h5 mb-3">
            Modelo em produção
          </h2>
          <dl className="row mb-0">
            <dt className="col-sm-4 fw-normal text-muted-r">Versão</dt>
            <dd className="col-sm-8">{meta.version}</dd>
            <dt className="col-sm-4 fw-normal text-muted-r">Treinado em</dt>
            <dd className="col-sm-8">{meta.created_at ? new Date(meta.created_at).toLocaleString("pt-BR") : "—"}</dd>
            <dt className="col-sm-4 fw-normal text-muted-r">Dataset (hash DVC)</dt>
            <dd className="col-sm-8 text-break">{meta.dataset_md5}</dd>
          </dl>

          <h3 className="h6 mt-4 mb-2">Ranking (holdout separado por artista)</h3>
          <div className="table-responsive">
            <table className="table align-middle mb-0">
              <thead>
                <tr className="text-muted-r">
                  <th scope="col" className="fw-medium">
                    Modalidade
                  </th>
                  <th scope="col" className="fw-medium text-end">
                    Spearman
                  </th>
                  <th scope="col" className="fw-medium text-end">
                    NDCG@{k}
                  </th>
                  <th scope="col" className="fw-medium text-end">
                    Piso (baseline)
                  </th>
                  <th scope="col" className="fw-medium text-end">
                    ROC-AUC do badge
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row" className="fw-normal">
                    Com histórico do artista
                  </th>
                  <td className="text-end">{f(m.spearman)}</td>
                  <td className="text-end">{f(m.ndcg_at_k)}</td>
                  <td className="text-end text-muted-r">
                    {f(meta.baseline_artist_mean_spearman)} · {f(meta.baseline_artist_mean_ndcg_at_k)}
                  </td>
                  <td className="text-end">{f(m.hit_roc_auc)}</td>
                </tr>
                <tr>
                  <th scope="row" className="fw-normal">
                    Só áudio (artista novo)
                  </th>
                  <td className="text-end">{f(m.audio_only_spearman)}</td>
                  <td className="text-end">{f(m.audio_only_ndcg_at_k)}</td>
                  <td className="text-end text-muted-r">
                    {f(meta.baseline_ridge_spearman)} · {f(meta.baseline_ridge_ndcg_at_k)}
                  </td>
                  <td className="text-end">{f(m.audio_only_hit_roc_auc)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-muted-r small mt-2 mb-0">
            Piso: ordenar só pela popularidade média do artista (com histórico) ou uma regressão linear (só áudio).
            Badge "é hit" com corte de {f(m.badge_precision, 2)} de precisão e {f(m.badge_recall, 2)} de recall na
            modalidade com histórico; calibração: ECE {f(m.ece)}, Brier {f(m.brier)} (só a taxa-base:{" "}
            {f(m.brier_base_rate)}).
          </p>
        </section>
      )}

      <section className="panel mb-4" aria-labelledby="mc-use">
        <h2 id="mc-use" className="h5 mb-3">
          Uso pretendido e limites
        </h2>
        <ul className="mb-0 ps-3 d-grid gap-2">
          <li>
            <strong>Serve para ordenar demos</strong> dentro de um lote: o percentil previsto compara a faixa com o
            próprio gênero. Não serve como decisão final de contratação.
          </li>
          <li>
            <strong>Duas modalidades.</strong> Com o histórico do artista (outras faixas já publicadas) o ranking é
            muito melhor; sem ele, só as características da faixa, o sinal é fraco, porque o áudio explica pouco de quem
            estoura. Para um artista realmente novo, use a modalidade só áudio e leia o resultado como "vale ouvir", não
            como "vai estourar".
          </li>
          <li>
            <strong>Os scores da modalidade só áudio ficam perto de 50.</strong> Com pouco sinal o modelo raramente
            arrisca valores extremos. O que importa é a ordem, não o valor absoluto.
          </li>
          <li>
            <strong>O histórico do artista é informação de fama,</strong> não de áudio: ele deve vir de dados reais do
            artista (popularidade das outras faixas). Se estiver errado ou desatualizado, a previsão herda o erro.
          </li>
          <li>
            <strong>Viés de seleção.</strong> O dataset do Kaggle tende a ser montado por sementes de gênero; a
            proporção de hits por gênero pode não refletir o mercado, e a popularidade é um retrato de um momento, sem
            data de lançamento.
          </li>
          <li>
            <strong>Gêneros de nicho</strong> (grindcore, black-metal) têm ranking mais fraco que o agregado.
          </li>
        </ul>
      </section>

      {meta && (
        <p className="text-muted-r small">
          {meta.known_genres.length} gêneros com amostra suficiente; os demais caem numa categoria agrupada.
        </p>
      )}
    </>
  );
}
