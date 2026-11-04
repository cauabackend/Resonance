"""Avalia o candidato (nested CV + gêneros de nicho) e só o promove a Production se passar o gate.

Tudo é medido nas duas modalidades: **com histórico do artista** (`artist_aware`) e **só áudio** (`audio_only`, a demo
de um artista novo). O gate exige que NENHUMA das duas piore.
"""

import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from resonance_features import aggregate_shap, engineer
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from .common import GENRE, ROOT, TARGET, load_params, repo_path
from .schema import PCT
from .train import (
    MODES,
    fit_model,
    has_history,
    holdout_metrics,
    predict_score,
    ranking_metrics,
    setup_mlflow,
    split,
    tune,
)

MODEL_NAME = "resonance"
CANDIDATE = ROOT / "model" / "candidate"
PRODUCTION = ROOT / "model" / "production"


def shap_importance(bundle: dict, rows: pd.DataFrame) -> dict[str, float]:
    """mean(|SHAP|) por fator (colunas do mesmo fator somadas), em pontos de percentil."""
    x = bundle["preprocessor"].transform(engineer(rows, bundle["stats"]))
    sv = np.asarray(shap.TreeExplainer(bundle["model"]).shap_values(x))
    names, mat = aggregate_shap(sv, bundle["factor_groups"])
    return dict(zip(names, np.abs(mat).mean(axis=0).tolist()))


def importance_stability(per_fold: list[dict[str, float]]) -> dict:
    """Correlação de rank (Spearman) das importâncias SHAP entre os folds: detecta instabilidade que um SHAP único não mostra."""
    names = list(per_fold[0])
    mat = np.array([[f[n] for n in names] for f in per_fold])
    pairs = [float(spearmanr(mat[i], mat[j]).statistic) for i in range(len(mat)) for j in range(i + 1, len(mat))]
    return {
        "mean_pairwise_spearman": float(np.mean(pairs)),
        "min_pairwise_spearman": float(np.min(pairs)),
        "mean_abs_shap": dict(zip(names, mat.mean(axis=0).tolist())),
    }


def nested_cv(df: pd.DataFrame, p: dict) -> dict:
    """Nested CV agrupado por artista. Externo: estima a performance. Interno (`tune`): escolhe hiperparâmetros.

    Reporta Spearman e NDCG@K das duas modalidades, agregados e por gênero de nicho (predições out-of-fold).
    """
    e, seed = p["evaluate"], p["split"]["seed"]
    outer = StratifiedGroupKFold(e["cv_folds"], shuffle=True, random_state=seed)
    oof = {m: np.zeros(len(df)) for m in MODES}
    folds: dict[str, list[dict]] = {m: [] for m in MODES}
    chosen, importances = [], []
    for tr, te in outer.split(df, df[GENRE], groups=df["artist"]):
        hp, _ = tune(df.iloc[tr], p)
        bundle = fit_model(df.iloc[tr], p, hp)
        for m in MODES:
            oof[m][te] = predict_score(bundle, df.iloc[te], m == "artist_aware")
            keep = has_history(df.iloc[te]) if m == "artist_aware" else np.ones(len(te), dtype=bool)
            folds[m].append(
                ranking_metrics(
                    df.iloc[te][PCT].to_numpy()[keep],
                    oof[m][te][keep],
                    e["ndcg_k"],
                    e["batch_size"],
                    e["n_batches"],
                    seed,
                )
            )
        chosen.append(hp)
        importances.append(shap_importance(bundle, df.iloc[te].sample(min(1000, len(te)), random_state=seed)))

    out: dict = {"chosen_hyperparams": chosen, "shap_stability": importance_stability(importances)}
    for m in MODES:
        sp, nd = [f["spearman"] for f in folds[m]], [f["ndcg_at_k"] for f in folds[m]]
        segments = {}
        for g in e["niche_genres"]:
            mask = (df[GENRE] == g).to_numpy()
            if m == "artist_aware":
                mask &= has_history(df)
            n = int(mask.sum())
            seg = {"n": n, "hits": int(df.loc[mask, TARGET].sum())}
            if n >= 10:  # lote = o gênero inteiro; K proporcional ao tamanho
                seg |= ranking_metrics(df.loc[mask, PCT], oof[m][mask], min(e["ndcg_k"], n // 10), n, 1, seed)
            segments[g] = seg
        out[m] = {
            "spearman_mean": float(np.mean(sp)),
            "spearman_std": float(np.std(sp)),
            "ndcg_at_k_mean": float(np.mean(nd)),
            "ndcg_at_k_std": float(np.std(nd)),
            "niche_segments": segments,
        }
    return out


# (métrica do candidato, baseline contra o qual é comparada, nome da métrica no baseline)
BASELINE_CHECKS = (
    ("ndcg_at_k", "artist_mean", "ndcg_at_k"),
    ("spearman", "artist_mean", "spearman"),
    ("audio_only_ndcg_at_k", "ridge", "ndcg_at_k"),
    ("audio_only_spearman", "ridge", "spearman"),
)
PROD_CHECKS = ("ndcg_at_k", "spearman", "audio_only_ndcg_at_k", "audio_only_spearman")


def gate(new: dict, baselines: dict, prod: dict | None, p: dict) -> tuple[bool, str]:
    """Reprova se NDCG@K, Spearman ou Brier piorarem além da tolerância (em QUALQUER modalidade) vs baseline ou produção."""
    tol, btol = p["evaluate"]["gate_tolerance"], p["evaluate"]["brier_tolerance"]
    for key, base, base_key in BASELINE_CHECKS:
        ref = baselines[base][base_key]
        if new[key] < ref - tol:
            return False, f"{key} {new[key]:.4f} < baseline {base} {ref:.4f} - tol {tol}"
    if prod is not None:
        for key in PROD_CHECKS:
            if new[key] < prod[key] - tol:
                return False, f"{key} {new[key]:.4f} < produção {prod[key]:.4f} - tol {tol}"
        for key in ("brier", "audio_only_brier"):
            if new[key] > prod[key] + btol:
                return False, f"{key} {new[key]:.4f} > produção {prod[key]:.4f} + tol {btol}"
    return True, "ok"


def promote(meta: dict) -> None:
    """Registra a versão (alias 'production') no MLflow Registry e copia candidato -> production."""
    setup_mlflow()
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":  # não engole erro de rede/auth
            raise
    # Artefato genérico (joblib), não um flavor do MLflow: registra apontando p/ o artifact do run.
    source = f"{client.get_run(meta['run_id']).info.artifact_uri}/bundle"
    mv = client.create_model_version(
        MODEL_NAME,
        source,
        run_id=meta["run_id"],
        tags={"model_version": meta["version"], "dataset_md5": meta["dataset_md5"]},
    )
    # Cópia local atômica (rename); o alias é o último passo, o "commit" da promoção.
    tmp = PRODUCTION.with_name(PRODUCTION.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(CANDIDATE, tmp)
    shutil.rmtree(PRODUCTION, ignore_errors=True)
    tmp.rename(PRODUCTION)
    client.set_registered_model_alias(MODEL_NAME, "production", mv.version)


def production_metrics(df: pd.DataFrame, p: dict, prod_file, dataset_md5: str) -> dict | None:
    """Produção é reavaliada no MESMO holdout do candidato (o número guardado no bundle vem de outro split)."""
    _, test = split(df, p)
    bundle = joblib.load(prod_file)
    if bundle.get("dataset_md5") != dataset_md5:
        return None  # outro dataset => outro holdout: a produção já pode ter visto essas linhas no treino
    return holdout_metrics(bundle, test, p)


def main() -> None:
    p = load_params()
    df = pd.read_csv(repo_path(p["data"]["processed"]))
    cand = joblib.load(CANDIDATE / "bundle.joblib")
    meta = json.loads((CANDIDATE / "meta.json").read_text())

    cv = nested_cv(df, p)
    prod_file = PRODUCTION / "bundle.joblib"
    prod = production_metrics(df, p, prod_file, cand["dataset_md5"]) if prod_file.exists() else None
    ok, reason = gate(cand["metrics"], cand["baseline"], prod, p)

    report = {
        "holdout": cand["metrics"],
        "baseline": cand["baseline"],
        "nested_cv": cv,
        "production_before": prod,
        "promoted": ok,
        "gate_reason": reason,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    m, b = cand["metrics"], cand["baseline"]
    flat = {
        "holdout_ndcg_at_k": m["ndcg_at_k"],
        "holdout_spearman": m["spearman"],
        "holdout_hit_roc_auc": m["hit_roc_auc"],
        "holdout_brier": m["brier"],
        "holdout_ece": m["ece"],
        "holdout_audio_only_ndcg_at_k": m["audio_only_ndcg_at_k"],
        "holdout_audio_only_spearman": m["audio_only_spearman"],
        "holdout_audio_only_hit_roc_auc": m["audio_only_hit_roc_auc"],
        "holdout_audio_only_brier": m["audio_only_brier"],
        "baseline_artist_mean_ndcg_at_k": b["artist_mean"]["ndcg_at_k"],
        "baseline_ridge_ndcg_at_k": b["ridge"]["ndcg_at_k"],
        "baseline_genre_mean_ndcg_at_k": b["genre_mean"]["ndcg_at_k"],
        "nested_cv_ndcg_at_k_mean": cv["artist_aware"]["ndcg_at_k_mean"],
        "nested_cv_spearman_mean": cv["artist_aware"]["spearman_mean"],
        "nested_cv_audio_only_ndcg_at_k_mean": cv["audio_only"]["ndcg_at_k_mean"],
        "nested_cv_audio_only_spearman_mean": cv["audio_only"]["spearman_mean"],
    }
    (reports / "metrics.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")

    print(json.dumps(flat, indent=2))
    if not ok:
        sys.exit(f"Gate reprovou o modelo: {reason}")
    promote(meta)
    print(f"Modelo {cand['version']} promovido a production")


if __name__ == "__main__":
    main()
