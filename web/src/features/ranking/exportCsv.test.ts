import { describe, expect, it } from "vitest";
import { rankingToCsv } from "./exportCsv";
import type { Demo } from "./parseCsv";

const demo = (name: string, artist: string): Demo => ({
  name,
  artist,
  features: {
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
  },
});

describe("rankingToCsv", () => {
  it("mantém a ordem da API e escapa vírgulas e aspas", () => {
    const csv = rankingToCsv(
      [
        {
          input_index: 1,
          rank: 1,
          mode: "artist_aware",
          percentile_score: 62.5,
          expected_popularity: 55,
          probability: 0.31,
          is_hit: true,
        },
        {
          input_index: 0,
          rank: 2,
          mode: "audio_only",
          percentile_score: 40,
          expected_popularity: 30,
          probability: 0.1,
          is_hit: false,
        },
      ],
      [demo('Song "A", live', "X"), demo("B", "Y")],
    );
    const [header, first, second] = csv.split("\n");
    expect(header).toBe("rank,name,artist,track_genre,mode,percentile_score,expected_popularity,probability,is_hit");
    expect(first).toBe("1,B,Y,pop,artist_aware,62.50,55.0,0.3100,true");
    expect(second).toBe('2,"Song ""A"", live",X,pop,audio_only,40.00,30.0,0.1000,false');
  });
});
