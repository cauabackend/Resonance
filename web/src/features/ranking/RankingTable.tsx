import type { CSSProperties } from "react";
import type { BatchItem } from "../../api/types";
import { VerdictBadge } from "../../components/ui/Badge";
import type { Demo } from "./parseCsv";

interface Props {
  results: BatchItem[]; // já ordenado por percentile_score pela API
  demos: Demo[];
  topK: number;
}

export default function RankingTable({ results, demos, topK }: Props) {
  return (
    <div className="table-responsive">
      <table className="table align-middle mb-0 ranking-table">
        <caption className="visually-hidden">Faixas ordenadas pelo percentil previsto no gênero</caption>
        <thead>
          <tr className="text-muted-r">
            <th scope="col" className="fw-medium text-end">
              #
            </th>
            <th scope="col" className="fw-medium">
              Faixa
            </th>
            <th scope="col" className="fw-medium">
              Gênero
            </th>
            <th scope="col" className="fw-medium">
              Modalidade
            </th>
            <th scope="col" className="fw-medium text-end">
              Percentil previsto
            </th>
            <th scope="col" className="fw-medium score-col">
              <span className="visually-hidden">Barra do percentil (0 a 100)</span>
            </th>
            <th scope="col" className="fw-medium text-end">
              P(hit)
            </th>
            <th scope="col" className="fw-medium">
              Badge
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => {
            const demo = demos[r.input_index];
            return (
              <tr
                key={r.input_index}
                className={r.rank === topK ? "ranking-cut" : undefined}
                style={{ "--i": Math.min(i, 20) } as CSSProperties}
              >
                <td className="text-end pop-score">{r.rank}</td>
                <td>
                  <span className="d-block fw-semibold">{demo.name}</span>
                  {demo.artist && <span className="d-block text-muted-r small">{demo.artist}</span>}
                </td>
                <td className="text-muted-r">{demo.features.track_genre}</td>
                <td className="text-muted-r small">{r.mode === "artist_aware" ? "com histórico" : "só áudio"}</td>
                <td className="text-end" title={`Popularidade esperada no gênero: ${r.expected_popularity.toFixed(0)}`}>
                  {r.percentile_score.toFixed(1)}
                </td>
                <td className="score-col" aria-hidden="true">
                  <span className="score-bar">
                    <span style={{ width: `${r.percentile_score}%` }} />
                  </span>
                </td>
                <td className="text-end text-muted-r">{(r.probability * 100).toFixed(0)}%</td>
                <td>
                  <VerdictBadge isHit={r.is_hit} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
