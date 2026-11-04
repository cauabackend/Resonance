import { describe, expect, it } from "vitest";
import { parseCsv, parseDemos } from "./features/ranking/parseCsv";

const HEADER =
  "name,artist,danceability,energy,valence,tempo,loudness,speechiness,acousticness,instrumentalness,duration_ms,track_genre";
const ROW = '"Song, with comma",Artist,0.5,0.6,0.7,120,-5,0.05,0.1,0,200000,pop';

describe("parseCsv", () => {
  it("lida com aspas, vírgulas internas e CRLF", () => {
    expect(parseCsv('a,"b,c",d\r\n"x ""q""",2,3\n')).toEqual([
      ["a", "b,c", "d"],
      ['x "q"', "2", "3"],
    ]);
  });
});

describe("parseDemos", () => {
  it("converte linhas em faixas com features numéricas", () => {
    const r = parseDemos(`${HEADER}\n${ROW}\n`);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.demos[0].name).toBe("Song, with comma");
      expect(r.demos[0].features).toMatchObject({ danceability: 0.5, duration_ms: 200000, track_genre: "pop" });
    }
  });

  it("name e artist são opcionais", () => {
    const r = parseDemos(
      "danceability,energy,valence,tempo,loudness,speechiness,acousticness,instrumentalness,duration_ms,track_genre\n0.5,0.6,0.7,120,-5,0.05,0.1,0,200000,pop",
    );
    expect(r.ok && r.demos[0].name).toBe("Faixa 1");
  });

  it("falha com mensagem clara", () => {
    expect(parseDemos("")).toMatchObject({ ok: false });
    expect(parseDemos("danceability\n0.5")).toMatchObject({
      ok: false,
      error: expect.stringContaining("Colunas ausentes"),
    });
    expect(parseDemos(`${HEADER}\n${ROW.replace(",0.6,", ",abc,")}`)).toMatchObject({
      error: expect.stringContaining("Linha 2"),
    });
    expect(parseDemos(`${HEADER}\n${ROW.replace(",pop", ", ")}`)).toMatchObject({
      error: expect.stringContaining("track_genre"),
    });
    const many = `${HEADER}\n${Array(1001).fill(ROW).join("\n")}`;
    expect(parseDemos(many)).toMatchObject({ error: expect.stringContaining("Máximo de 1000") });
  });

  it("lê colunas opcionais e deixa vazio como ausente (artista novo)", () => {
    const h = `${HEADER},liveness,explicit,artist_n_tracks,artist_mean_popularity`;
    const r = parseDemos(`${h}
"Com histórico",A,0.5,0.6,0.7,120,-5,0.05,0.1,0,200000,pop,0.2,true,12,71.5
Novo,B,0.5,0.6,0.7,120,-5,0.05,0.1,0,200000,pop,,sim,,
`);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.demos[0].features).toMatchObject({
        liveness: 0.2,
        explicit: true,
        artist_n_tracks: 12,
        artist_mean_popularity: 71.5,
      });
      expect(r.demos[1].features.artist_mean_popularity).toBeUndefined();
      expect(r.demos[1].features.explicit).toBe(true);
    }
    expect(
      parseDemos(`${h}
X,A,0.5,0.6,0.7,120,-5,0.05,0.1,0,200000,pop,abc,true,1,50`),
    ).toMatchObject({
      error: expect.stringContaining("liveness"),
    });
  });
});
