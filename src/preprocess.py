import pandas as pd

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "loudness", "speechiness", "acousticness",
    "instrumentalness", "duration_ms",
]


def preprocess(
    input_path: str = "data/raw/tracks.csv",
    output_path: str = "data/processed/tracks_clean.csv",
    year_min: int = 2010,
    year_max: int = 2026,
    hit_threshold: int = 60,
):
    """Limpa o dataset bruto e cria a coluna target is_hit."""
    df = pd.read_csv(input_path)

    df = df.drop_duplicates(subset=["name", "artist"])
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    df = df.dropna(subset=AUDIO_FEATURES)
    df["is_hit"] = (df["popularity"] >= hit_threshold).astype(int)

    df.to_csv(output_path, index=False)

    print(f"Dataset limpo: {len(df)} músicas")
    print(f"\nDistribuição do target:")
    print(df["is_hit"].value_counts().to_string())
    print(f"\nMúsicas por ano:")
    print(df["year"].value_counts().sort_index().to_string())

    return df


if __name__ == "__main__":
    preprocess()