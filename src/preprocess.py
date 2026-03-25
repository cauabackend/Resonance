import pandas as pd

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "loudness", "speechiness", "acousticness",
    "instrumentalness", "duration_ms",
]


def preprocess(
    input_path: str = "data/raw/dataset.csv",
    output_path: str = "data/processed/tracks_clean.csv",
    hit_threshold: int = 60,
):
    """Limpa o dataset do Kaggle e cria a coluna target is_hit."""
    df = pd.read_csv(input_path)

    df = df.rename(columns={"track_name": "name", "artists": "artist"})
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    df = df.drop_duplicates(subset=["name", "artist"])
    df = df.dropna(subset=AUDIO_FEATURES)
    df["is_hit"] = (df["popularity"] >= hit_threshold).astype(int)

    df.to_csv(output_path, index=False)

    print(f"Dataset limpo: {len(df)} músicas")
    print(f"\nDistribuição do target:")
    print(df["is_hit"].value_counts().to_string())
    print(f"\nGêneros mais frequentes:")
    print(df["track_genre"].value_counts().head(10).to_string())

    return df


if __name__ == "__main__":
    preprocess()