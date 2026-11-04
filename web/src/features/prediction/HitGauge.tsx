import Plot from "./Plot";
import { baseLayout, GRAYS, plotConfig } from "./plotlyTheme";

interface Props {
  probability: number;
  threshold: number;
}

export default function HitGauge({ probability, threshold }: Props) {
  const pct = probability * 100;
  return (
    <div
      role="img"
      aria-label={`Probabilidade de hit: ${pct.toFixed(0)}%. Limiar de decisão: ${(threshold * 100).toFixed(0)}%.`}
    >
      <Plot
        data={[
          {
            type: "indicator",
            mode: "gauge+number",
            value: pct,
            number: { suffix: "%", font: { size: 44, color: GRAYS.strong }, valueformat: ".0f" },
            gauge: {
              axis: { range: [0, 100], tickcolor: GRAYS.muted, tickfont: { color: GRAYS.muted } },
              bar: { color: GRAYS.strong, thickness: 0.28 },
              bgcolor: "rgba(0,0,0,0)",
              borderwidth: 0,
              steps: [
                { range: [0, threshold * 100], color: "#2a2a2a" },
                { range: [threshold * 100, 100], color: "#4a4a4a" },
              ],
              threshold: { line: { color: GRAYS.strong, width: 3 }, thickness: 0.85, value: threshold * 100 },
            },
          },
        ]}
        layout={{ ...baseLayout, height: 220, margin: { t: 24, r: 44, b: 8, l: 36 } }}
        config={plotConfig}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
