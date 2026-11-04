import { label, UNIT_FEATURES } from "./labels";
import Plot from "./Plot";
import { baseLayout, GRAYS, plotConfig } from "./plotlyTheme";

interface Props {
  track: Record<string, number>;
  hitMeans: Record<string, number>;
}

export default function FeatureBars({ track, hitMeans }: Props) {
  const y = UNIT_FEATURES.map(label);
  return (
    <Plot
      data={[
        {
          type: "bar",
          orientation: "h",
          y,
          x: UNIT_FEATURES.map((f) => hitMeans[f]),
          name: "Média dos hits",
          marker: { color: GRAYS.grid },
        },
        {
          type: "bar",
          orientation: "h",
          y,
          x: UNIT_FEATURES.map((f) => track[f]),
          name: "Esta música",
          marker: { color: GRAYS.strong },
        },
      ]}
      layout={{
        ...baseLayout,
        barmode: "group",
        showlegend: true,
        legend: { orientation: "h", y: -0.12 },
        height: 400,
        xaxis: { range: [0, 1], gridcolor: GRAYS.grid, zeroline: false },
        yaxis: { autorange: "reversed" },
      }}
      config={plotConfig}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
