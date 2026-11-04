import { formatValue, label } from "./labels";

interface Props {
  features: Record<string, number | string | boolean>;
  hitMeans: Record<string, number>;
}

export default function FeatureTable({ features, hitMeans }: Props) {
  return (
    <div className="table-responsive">
      <table className="table align-middle mb-0">
        <thead>
          <tr className="text-muted-r">
            <th scope="col" className="fw-medium">
              Característica
            </th>
            <th scope="col" className="fw-medium text-end">
              Esta música
            </th>
            <th scope="col" className="fw-medium text-end">
              Média dos hits
            </th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(features).map(([f, v]) => (
            <tr key={f}>
              <th scope="row" className="fw-normal">
                {label(f)}
              </th>
              <td className="text-end">{formatValue(f, v)}</td>
              <td className="text-end text-muted-r">{f in hitMeans ? formatValue(f, hitMeans[f]) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
