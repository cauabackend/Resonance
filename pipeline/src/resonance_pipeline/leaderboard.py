"""Leaderboard: compara abordagens no mesmo split por artista (não é gate; documenta a escolha do modelo).

Ridge, Random Forest, HistGradientBoosting, XGBRegressor e XGBRanker (LambdaMART via `rank:ndcg`, grupos = gênero), todos
com as MESMAS features (engenharia compartilhada) e o mesmo mascaramento do histórico do artista no treino, avaliados nas
duas modalidades. LightGBM/CatBoost ficam de fora de propósito: seriam dependências novas para um ganho que o ADR 0002 já
mostra ser pequeno.
"""

import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from resonance_features import engineer, fit_stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from xgboost import XGBRanker, XGBRegressor

from .common import GENRE, ROOT, load_params, repo_path
from .schema import PCT
from .train import MODES, build_preprocessor, has_history, mask_artist, ranking_metrics, setup_mlflow, split

RANK_LEVELS = 30  # rank:ndcg exige relevância inteira (0..30)


def run(df: pd.DataFrame, p: dict, hp: dict) -> dict[str, dict]:
    train, test = split(df, p)
    e, seed = p["evaluate"], p["split"]["seed"]
    stats = fit_stats(train)
    pre = build_preprocessor(p["features"]["min_genre_count"])
    etr = mask_artist(engineer(train, stats), p["features"]["artist_mask_rate"], seed)
    Xtr = pre.fit_transform(etr)
    views = {m: pre.transform(engineer(test, stats, m == "artist_aware")) for m in MODES}
    y = train[PCT]
    xgb = {**p["xgb"], **hp}

    def ranker():
        order = np.argsort(train[GENRE].to_numpy(), kind="stable")  # XGBRanker exige as linhas agrupadas por qid
        labels = np.minimum((y.to_numpy()[order] / 100 * RANK_LEVELS).astype(int), RANK_LEVELS)
        qid = train[GENRE].astype("category").cat.codes.to_numpy()[order]
        return XGBRanker(objective="rank:ndcg", **xgb, random_state=seed).fit(Xtr[order], labels, qid=qid)

    models = {
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0)).fit(Xtr, y),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed
        ).fit(Xtr, y),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=seed).fit(Xtr, y),
        "xgb_regressor": lambda: XGBRegressor(**xgb, random_state=seed).fit(Xtr, y),
        "xgb_ranker_lambdamart": ranker,
    }
    board = {}
    for name, make in models.items():
        t = time.perf_counter()
        model = make()
        row: dict = {"fit_seconds": round(time.perf_counter() - t, 1)}
        for m, X in views.items():
            keep = has_history(test) if m == "artist_aware" else np.ones(len(test), dtype=bool)
            r = ranking_metrics(
                test[PCT].to_numpy()[keep],
                np.asarray(model.predict(X))[keep],
                e["ndcg_k"],
                e["batch_size"],
                e["n_batches"],
                seed,
            )
            row[m] = {"spearman": r["spearman"], "ndcg_at_k": r["ndcg_at_k"]}
        board[name] = row
    return board


def main() -> None:
    p = load_params()
    df = pd.read_csv(repo_path(p["data"]["processed"]))
    hp = joblib.load(ROOT / "model" / "production" / "bundle.joblib")["hyperparams"]
    board = run(df, p, hp)
    setup_mlflow()
    for name, row in board.items():
        with mlflow.start_run(run_name=f"leaderboard-{name}"):
            mlflow.log_metrics({f"test_{m}_{k}": v for m in MODES for k, v in row[m].items()})
    out = ROOT / "reports" / "leaderboard.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(board, indent=2), encoding="utf-8")
    for name, row in sorted(board.items(), key=lambda kv: -kv[1]["artist_aware"]["ndcg_at_k"]):
        a, o = row["artist_aware"], row["audio_only"]
        print(
            f"{name:<24} com histórico: ndcg={a['ndcg_at_k']:.3f} rho={a['spearman']:+.3f} | só áudio: ndcg={o['ndcg_at_k']:.3f} rho={o['spearman']:+.3f}"
        )


if __name__ == "__main__":
    main()
