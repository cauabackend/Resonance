import { compile } from "sass";
import { describe, expect, it } from "vitest";

// Gate da Fase 3: o tema compilado (Bootstrap + overrides + app.scss) não pode conter nenhum matiz.
describe("tema monocromático", () => {
  const css = ["src/styles/theme.scss", "src/styles/app.scss"]
    .map(
      (f) =>
        compile(f, {
          loadPaths: ["node_modules"],
          quietDeps: true,
          silenceDeprecations: ["import", "global-builtin", "color-functions", "if-function"],
        }).css,
    )
    .join("\n");

  const chroma = (r: number, g: number, b: number) => Math.max(r, g, b) - Math.min(r, g, b);

  it("não tem cores hex cromáticas", () => {
    const hexes = [...css.matchAll(/#([0-9a-f]{6}|[0-9a-f]{3})\b/gi)].map((m) => m[1]);
    const bad = hexes.filter((h) => {
      const full = h.length === 3 ? [...h].map((c) => c + c).join("") : h;
      const [r, g, b] = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16));
      return chroma(r, g, b) > 8;
    });
    expect(bad).toEqual([]);
  });

  it("não tem rgb()/rgba() ou triplas --bs-*-rgb cromáticas", () => {
    const bad = [...css.matchAll(/(?<![\d.])(\d{1,3}), ?(\d{1,3}), ?(\d{1,3})(?![\d.])/g)]
      .map((m) => m.slice(1).map(Number))
      .filter(([r, g, b]) => Math.max(r, g, b) <= 255 && Math.max(r, g, b) > 20 && chroma(r, g, b) > 8);
    expect(bad).toEqual([]);
  });
});
