import pandas as pd

from .common import FEATURES, load_params, repo_path
from .validate import validate_clean, validate_raw


def clean(df: pd.DataFrame, max_duration_ms: int, min_tempo: float) -> pd.DataFrame:
    df = validate_raw(df.drop(columns=["Unnamed: 0"], errors="ignore"))

    # Dedup por id do Spotify; sem id, name+artist+duration_ms (remix/live têm duração diferente).
    has_id = df["track_id"].notna() if "track_id" in df else pd.Series(False, index=df.index)
    df = pd.concat(
        [
            df[has_id].drop_duplicates("track_id"),
            df[~has_id].drop_duplicates(["name", "artist", "duration_ms"]),
        ]
    )

    df = df.dropna(subset=[*FEATURES, "name", "artist"])
    # Colaborações vêm como "A;B": o grupo dos splits e do histórico é o PRIMEIRO artista, senão a faixa solo de A
    # cairia no treino e a colaboração A;B no teste (vazamento) e o histórico de A ficaria incompleto.
    df["artist"] = df["artist"].str.split(";").str[0].str.strip()
    df = df[df["duration_ms"].between(1, max_duration_ms) & (df["tempo"] >= min_tempo)].copy()
    return validate_clean(df).reset_index(drop=True)


def main() -> None:
    p = load_params()["data"]
    df = clean(pd.read_csv(repo_path(p["raw"])), p["max_duration_ms"], p["min_tempo"])
    out = repo_path(p["clean"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Dataset limpo: {len(df)} músicas")


if __name__ == "__main__":
    main()
