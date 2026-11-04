"""Treina baselines + regressor de percentil por gênero, calibra o badge e grava o artefato candidato.

O produto é um RANKING (separar as 50 melhores de 500 demos), não uma classificação binária:
- alvo: `popularity_percentile_within_genre` (0-100), calculado em target.py;
- modelo: XGBRegressor; o score contínuo é o `percentile_score`;
- duas modalidades no MESMO modelo (o XGBoost trata NaN): **só áudio** (demo de artista novo) e **com histórico do
  artista** (artista que já tem faixas). No treino o histórico é mascarado em metade das linhas para o modelo aprender
  as duas; na avaliação as duas são medidas separadamente;
- badge "é hit": camada derivada, calibrada por modalidade num fold separado. A entrada da isotônica é a **popularidade
  esperada no gênero** (quantil do gênero no percentile_score), que devolve ao badge a informação de gênero que o
  percentil descarta (ADR 0007).
"""

import itertools
import json
import os
import subprocess
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike
from resonance_features import (
    ARTIST,
    AUDIO,
    DERIVED,
    MODEL_COLUMNS,
    engineer,
    expected_popularity,
    factor_groups,
    fit_stats,
)
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .calibrate import best_threshold, brier_before_after, calibration, fit_calibrator, reliability_plot  # noqa: F401
from .common import FEATURES, GENRE, ROOT, TARGET, load_params, repo_path
from .schema import PCT

MODES = ("artist_aware", "audio_only")
RAW_COLUMNS = [*AUDIO, "liveness", "key", "mode", "time_signature", "explicit", *ARTIST, "popularity", GENRE]


def _genre_encoder(min_count: int) -> OneHotEncoder:
    # Gêneros raros viram "infrequent"; gênero inédito em produção cai na mesma coluna (sem skew).
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_count, sparse_output=False)


def build_preprocessor(min_count: int, columns: list[str] | None = None) -> ColumnTransformer:
    return ColumnTransformer(
        [("num", "passthrough", columns or MODEL_COLUMNS), ("genre", _genre_encoder(min_count), [GENRE])],
        sparse_threshold=0,
    )


def group_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split por artista: popularidade é em boa parte efeito do artista, e um split por linha vazaria."""
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(df, groups=df["artist"]))
    return df.iloc[tr], df.iloc[te]


def split(df: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return group_split(df, p["split"]["test_size"], p["split"]["seed"])


def has_history(df: pd.DataFrame) -> np.ndarray:
    """Faixas cujo artista tem OUTRAS faixas no dataset: só nelas a modalidade com histórico faz sentido (a API manda as
    demais para audio_only), então é nelas que ela é calibrada e medida."""
    return df["artist_mean_popularity"].notna().to_numpy()


def mask_artist(e: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Apaga o histórico do artista em `rate` das linhas de TREINO: o modelo aprende a modalidade só-áudio também."""
    e = e.copy()
    m = np.random.default_rng(seed).random(len(e)) < rate
    e.loc[m, [*ARTIST, DERIVED[1]]] = np.nan
    return e


def _xgb(p: dict, hp: dict) -> XGBRegressor:
    return XGBRegressor(**{**p["xgb"], **hp}, random_state=p["split"]["seed"])


def _spearman(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def tune(train: pd.DataFrame, p: dict) -> tuple[dict, list[dict]]:
    """Loop INTERNO do nested CV: grid de hiperparâmetros por Spearman (média das duas modalidades), folds por artista."""
    grid, seed = p["tune"]["grid"], p["split"]["seed"]
    folds = []
    for tr, va in GroupKFold(p["tune"]["folds"]).split(train, groups=train["artist"]):
        a, b = train.iloc[tr], train.iloc[va]
        stats = fit_stats(a)
        pre = build_preprocessor(p["features"]["min_genre_count"])
        X = pre.fit_transform(mask_artist(engineer(a, stats), p["features"]["artist_mask_rate"], seed))
        views = [pre.transform(engineer(b, stats, use_artist_context=m == "artist_aware")) for m in MODES]
        folds.append((X, a[PCT], views, b[PCT]))
    results: list[dict] = []
    for values in itertools.product(*grid.values()):
        hp = dict(zip(grid.keys(), values))
        scores = [
            np.mean([_spearman(yv, _xgb(p, hp).fit(X, y).predict(v)) for v in views])
            for X, y, views, yv in folds  # um fit por fold, avaliado nas duas modalidades
        ]
        results.append({"params": hp, "cv_spearman": float(np.mean(scores))})
    best: dict = max(results, key=lambda r: float(r["cv_spearman"]))
    return best["params"], results


def fit_model(train: pd.DataFrame, p: dict, hp: dict) -> dict:
    """Regressor no 'fit'; um calibrador de badge por modalidade num fold SEPARADO ('calib')."""
    fit, calib = group_split(train, p["split"]["calib_size"], p["split"]["seed"])
    stats = fit_stats(fit)
    pre = build_preprocessor(p["features"]["min_genre_count"])
    X = pre.fit_transform(mask_artist(engineer(fit, stats), p["features"]["artist_mask_rate"], p["split"]["seed"]))
    model = _xgb(p, hp).fit(X, fit[PCT])

    calibrators, thresholds = {}, {}
    for mode in MODES:
        aware = mode == "artist_aware"
        c = calib[has_history(calib)] if aware else calib
        score = np.clip(model.predict(pre.transform(engineer(c, stats, aware))), 0, 100)
        ep = expected_popularity(score, c[GENRE].to_numpy(), stats)
        calibrators[mode], thresholds[mode] = fit_calibrator(
            ep, c[TARGET].to_numpy(), p["decision"]["fn_cost"], p["decision"]["fp_cost"]
        )
    enc = pre.named_transformers_["genre"]
    rare = enc.infrequent_categories_[0] if enc.infrequent_categories_[0] is not None else []
    names = list(pre.get_feature_names_out())
    return {
        "preprocessor": pre,
        "model": model,
        "calibrators": calibrators,
        "thresholds": thresholds,
        "stats": stats,
        "features": FEATURES,
        "genre_col": GENRE,
        "feature_names": names,
        "factor_groups": factor_groups(names),
        "known_genres": [g for g in enc.categories_[0] if g not in rare],
        "hyperparams": hp,
    }


def predict_score(bundle: dict, df: pd.DataFrame, use_artist_context: bool = True) -> np.ndarray:
    """percentile_score previsto (0-100)."""
    e = engineer(df, bundle["stats"], use_artist_context)
    return np.clip(bundle["model"].predict(bundle["preprocessor"].transform(e)), 0, 100)


def predict_proba(bundle: dict, df: pd.DataFrame, mode: str = "artist_aware") -> np.ndarray:
    """P(is_hit) calibrada na modalidade pedida, derivada da popularidade esperada no gênero."""
    score = predict_score(bundle, df, mode == "artist_aware")
    return bundle["calibrators"][mode].predict(expected_popularity(score, df[GENRE].to_numpy(), bundle["stats"]))


def ranking_metrics(y_pct: ArrayLike, score: ArrayLike, k: int, batch_size: int, n_batches: int, seed: int) -> dict:
    """Spearman global + NDCG@K médio em lotes aleatórios (o caso de uso: top-K de um lote de demos)."""
    y_pct, score = np.asarray(y_pct), np.asarray(score)
    rng = np.random.default_rng(seed)
    size = min(batch_size, len(y_pct))
    ndcgs = []
    for _ in range(n_batches):
        idx = rng.choice(len(y_pct), size, replace=False)
        ndcgs.append(ndcg_score(y_pct[idx][None], score[idx][None], k=min(k, size)))
    return {"spearman": _spearman(y_pct, score), "ndcg_at_k": float(np.mean(ndcgs)), "ndcg_k": k}


def _mode_metrics(bundle: dict, test: pd.DataFrame, p: dict, mode: str) -> dict:
    e, aware = p["evaluate"], mode == "artist_aware"
    if aware:
        test = test[has_history(test)]  # a modalidade com histórico é medida onde ela se aplica
    score = predict_score(bundle, test, aware)
    ep = expected_popularity(score, test[GENRE].to_numpy(), bundle["stats"])
    prob = bundle["calibrators"][mode].predict(ep)
    y = test[TARGET].to_numpy()
    badge = prob >= bundle["thresholds"][mode]
    return {
        **ranking_metrics(test[PCT], score, e["ndcg_k"], e["batch_size"], e["n_batches"], p["split"]["seed"]),
        **calibration(y, prob),
        **brier_before_after(y, ep, prob),
        "hit_roc_auc": float(roc_auc_score(y, prob)),
        "badge_precision": float(precision_score(y, badge, zero_division=0)),
        "badge_recall": float(recall_score(y, badge, zero_division=0)),
        "score_std": float(score.std()),
    }


def holdout_metrics(bundle: dict, test: pd.DataFrame, p: dict) -> dict:
    """Métricas do holdout nas duas modalidades. Sem prefixo = com histórico; `audio_only_*` = demo de artista novo."""
    aware = _mode_metrics(bundle, test, p, "artist_aware")
    audio = _mode_metrics(bundle, test, p, "audio_only")
    return {**aware, **{f"audio_only_{k}": v for k, v in audio.items() if k != "ndcg_k"}}


def baselines(train: pd.DataFrame, test: pd.DataFrame, p: dict) -> dict:
    """Sem baseline, 'NDCG@50 = 0.8' não diz se o modelo captura sinal ou repete a ordem óbvia do catálogo."""
    e = p["evaluate"]

    def rank(score):
        return ranking_metrics(test[PCT], score, e["ndcg_k"], e["batch_size"], e["n_batches"], p["split"]["seed"])

    genre_mean = train.groupby(GENRE)["popularity"].mean()
    audio_cols = [*AUDIO]
    pre = ColumnTransformer(
        [("num", StandardScaler(), audio_cols), ("genre", _genre_encoder(p["features"]["min_genre_count"]), [GENRE])],
        sparse_threshold=0,
    )
    ridge = make_pipeline(pre, Ridge(alpha=1.0)).fit(train[[*audio_cols, GENRE]], train[PCT])
    return {
        # ordena só pela popularidade média do gênero: como o alvo é relativo ao gênero, tende ao acaso
        "genre_mean": rank(test[GENRE].map(genre_mean).fillna(train["popularity"].mean()).to_numpy()),
        # regressão linear só com áudio: o piso da modalidade só-áudio
        "ridge": rank(ridge.predict(test[[*audio_cols, GENRE]])),
        # ordenar só pela popularidade média do artista: o piso da modalidade com histórico
        # (na mesma população em que a modalidade com histórico é medida: artistas com outras faixas)
        "artist_mean": ranking_metrics(
            test[has_history(test)][PCT],
            test[has_history(test)]["artist_mean_popularity"].to_numpy(),
            e["ndcg_k"],
            e["batch_size"],
            e["n_batches"],
            p["split"]["seed"],
        ),
    }


def lineage() -> dict:
    """Hash DVC do dataset + commit git: responde 'que dado gerou esse modelo'."""
    dvc_file = ROOT / "data" / "raw" / "dataset.csv.dvc"
    md5 = yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"] if dvc_file.exists() else "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"dataset_md5": md5, "git_commit": commit}


def setup_mlflow() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("resonance")
        return
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("resonance") is None:
        mlflow.create_experiment("resonance", artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment("resonance")


def _loggable(m: dict) -> dict:
    return {k: v for k, v in m.items() if isinstance(v, float)}


def train_and_save() -> dict:
    p = load_params()
    df = pd.read_csv(repo_path(p["data"]["processed"]), usecols=lambda c: c in {*RAW_COLUMNS, "artist", PCT, TARGET})
    train, test = split(df, p)

    setup_mlflow()
    base = baselines(train, test, p)
    for name, m in base.items():  # baselines registrados ANTES do XGBoost
        with mlflow.start_run(run_name=f"baseline-{name}"):
            mlflow.log_metrics({f"test_{k}": v for k, v in _loggable(m).items()})
    print(f"Baselines: {json.dumps(base)}")

    hp, grid = tune(train, p)
    print(f"Hiperparâmetros escolhidos (loop interno): {hp}  grid: {grid}")
    bundle = fit_model(train, p, hp)
    metrics = holdout_metrics(bundle, test, p)
    print(json.dumps({k: v for k, v in metrics.items() if "reliability" not in k}, indent=2))

    lin = lineage()
    version = f"{datetime.now(UTC):%Y%m%d%H%M%S}-{lin['dataset_md5'][:7]}"
    bundle |= {
        "version": version,
        "metrics": metrics,
        "baseline": base,
        "params": p,
        **lin,
        "tune_grid": grid,
        "hit_means": train[train[TARGET] == 1][FEATURES].mean().to_dict(),
        "created_at": datetime.now(UTC).isoformat(),
    }

    out = ROOT / "model" / "candidate"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "bundle.joblib")

    with mlflow.start_run(run_name="xgboost-regressor") as run:
        mlflow.log_params({**p["xgb"], **hp, **p["decision"], "artist_mask_rate": p["features"]["artist_mask_rate"]})
        mlflow.log_metrics({f"test_{k}": v for k, v in _loggable(metrics).items()})
        mlflow.set_tags({"model_version": version, **lin})
        mlflow.log_artifact(str(out / "bundle.joblib"), artifact_path="bundle")
        for name, key in (("reliability", "reliability"), ("reliability_audio_only", "audio_only_reliability")):
            mlflow.log_artifact(str(reliability_plot(metrics[key], out / f"{name}.png")), artifact_path="plots")
        (out / "meta.json").write_text(json.dumps({"run_id": run.info.run_id, "version": version, **lin}))
    return bundle


if __name__ == "__main__":
    train_and_save()
