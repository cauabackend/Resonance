import type { ShapFactor } from "../../api/types";
import { label } from "./labels";
import Plot from "./Plot";
import { baseLayout, GRAYS, plotConfig } from "./plotlyTheme";

export default function ShapWaterfall({ factors }: { factors: ShapFactor[] }) {
  // Menor impacto no topo, maior embaixo.
  const sorted = [...factors].sort((a, b) => Math.abs(a.shap_value) - Math.abs(b.shap_value));
  return (
    <Plot
      data={[
        {
          type: "waterfall",
          orientation: "h",
          y: sorted.map((f) => label(f.feature)),
          x: sorted.map((f) => f.shap_value),
          measure: sorted.map(() => "relative"),
          increasing: { marker: { color: GRAYS.strong } },
          decreasing: { marker: { color: GRAYS.muted } },
          connector: { line: { color: GRAYS.grid } },
          hovertemplate: "%{y}: %{x:+.3f}<extra></extra>",
        },
      ]}
      layout={{
        ...baseLayout,
        height: 380,
        margin: { t: 16, r: 24, b: 48, l: 24 },
        xaxis: { title: { text: "Impacto no percentil previsto (pontos)" }, gridcolor: GRAYS.grid, zeroline: false },
        yaxis: { automargin: true },
      }}
      config={plotConfig}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
