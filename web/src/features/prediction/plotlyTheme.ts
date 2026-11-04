import type { Layout } from "plotly.js-dist-min";

// Só cinzas: sem o azul/vermelho padrão do Plotly.
export const GRAYS = { text: "#d4d4d4", muted: "#9a9a9a", grid: "#4a4a4a", strong: "#ffffff", surface: "#1a1a1a" };

export const baseLayout: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Schibsted Grotesk, system-ui, sans-serif", color: GRAYS.text, size: 13 },
  margin: { t: 24, r: 24, b: 40, l: 24 },
  showlegend: false,
};

export const plotConfig = { displayModeBar: false, responsive: true };
