import { label, UNIT_FEATURES } from "./labels";
import Plot from "./Plot";
import { baseLayout, GRAYS, plotConfig } from "./plotlyTheme";

interface Props {
  track: Record<string, number>;
  hitMeans: Record<string, number>;
}

const close = <T,>(a: T[]) => [...a, a[0]];

export default function RadarComparison({ track, hitMeans }: Props) {
  const theta = close(UNIT_FEATURES.map(label));
  return (
    <Plot
      data={[
        {
          type: "scatterpolar",
          r: close(UNIT_FEATURES.map((f) => hitMeans[f])),
          theta,
          name: "Média dos hits",
          fill: "toself",
          fillcolor: "rgba(154,154,154,0.25)",
          line: { color: GRAYS.muted, width: 2, dash: "dot" },
        },
        {
          type: "scatterpolar",
          r: close(UNIT_FEATURES.map((f) => track[f])),
          theta,
          name: "Esta música",
          fill: "toself",
          fillcolor: "rgba(255,255,255,0.18)",
          line: { color: GRAYS.strong, width: 2.5 },
        },
      ]}
      layout={{
        ...baseLayout,
        showlegend: true,
        legend: { orientation: "h", y: -0.1 },
        height: 400,
        polar: {
          bgcolor: "rgba(0,0,0,0)",
          radialaxis: { range: [0, 1], gridcolor: GRAYS.grid, linecolor: GRAYS.grid, tickfont: { color: GRAYS.muted } },
          angularaxis: { gridcolor: GRAYS.grid, linecolor: GRAYS.grid },
        },
      }}
      config={plotConfig}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
