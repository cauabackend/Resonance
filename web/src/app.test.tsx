import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, predictTrack } from "./api/client";
import { VerdictBadge } from "./components/ui/Badge";
import Tabs from "./components/ui/Tabs";
import { formatValue } from "./features/prediction/labels";

describe("VerdictBadge", () => {
  it("diferencia por texto e classe, não por cor", () => {
    const { rerender } = render(<VerdictBadge isHit />);
    expect(screen.getByText("Potencial de hit")).toHaveClass("verdict--hit");
    rerender(<VerdictBadge isHit={false} />);
    expect(screen.getByText("Baixo potencial")).toHaveClass("verdict--miss");
  });
});

describe("Tabs", () => {
  it("renderiza só o painel ativo e expõe roles ARIA", () => {
    render(
      <Tabs
        tabs={[
          { id: "a", label: "A", content: <p>conteudo-a</p> },
          { id: "b", label: "B", content: <p>conteudo-b</p> },
        ]}
      />,
    );
    expect(screen.getAllByRole("tab")).toHaveLength(2);
    expect(screen.getByText("conteudo-a")).toBeVisible();
    expect(screen.queryByText("conteudo-b")).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "B" }));
    expect(screen.getByRole("tab", { name: "B" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("formatValue", () => {
  it("formata duração, BPM e dB", () => {
    expect(formatValue("duration_ms", 185000)).toBe("3:05");
    expect(formatValue("tempo", 120.4)).toBe("120 BPM");
    expect(formatValue("loudness", -5.25)).toBe("-5.3 dB");
    expect(formatValue("track_genre", "pop")).toBe("pop");
  });
});

describe("API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  const features = {
    danceability: 0.5,
    energy: 0.5,
    valence: 0.5,
    tempo: 100,
    loudness: -5,
    speechiness: 0.1,
    acousticness: 0.1,
    instrumentalness: 0,
    duration_ms: 1000,
    track_genre: "pop",
  };

  it("envia o contrato esperado e devolve o JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ is_hit: true }) });
    vi.stubGlobal("fetch", fetchMock);
    await predictTrack(features);
    await predictTrack(features, "catalog");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body).source).toBe("catalog");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/v1\/predict$/);
    expect(JSON.parse(init.body)).toEqual({ features, source: "user", use_artist_context: true });
  });

  it("lança ApiError com o status em respostas não-ok", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503, text: async () => "down" }));
    await expect(predictTrack(features)).rejects.toMatchObject({ status: 503 });
    await expect(predictTrack(features)).rejects.toBeInstanceOf(ApiError);
  });

  it("converte falha de rede em ApiError status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
    await expect(predictTrack(features)).rejects.toMatchObject({ status: 0 });
  });
});
