"""Drift (Evidently): (1) features recebidas vs treino; (2) distribuição do percentile_score previsto."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from .common import FEATURES, GENRE, ROOT, load_params, repo_path
from .train import predict_score

SCORE = "percentile_score"
MODE = "mode"  # artist_aware | audio_only: a distribuição do score só é comparável dentro da mesma modalidade


def load_prediction_logs(log_dir: Path, days: int) -> pd.DataFrame:
    """Lê os JSONL particionados por data (dt=YYYY-MM-DD) das últimas `days` partições."""
    parts = sorted(log_dir.glob("dt=*"))[-days:]
    rows = [
        {**rec["features"], SCORE: rec.get(SCORE), MODE: rec.get(MODE)}
        for d in parts
        for f in d.glob("*.jsonl")
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for rec in [json.loads(line)]
    ]
    return pd.DataFrame(rows, columns=[*FEATURES, GENRE, SCORE, MODE])


def synthetic_production(reference: pd.DataFrame, n: int = 500, shift: bool = True, seed: int = 0) -> pd.DataFrame:
    """Tráfego sintético: amostra do treino, com danceability/energy deslocadas se `shift`."""
    cur = reference[[*FEATURES, GENRE]].sample(n, random_state=seed).reset_index(drop=True)
    if shift:
        rng = np.random.default_rng(seed)
        cur["danceability"] = (cur["danceability"] * 0.5 + rng.normal(0, 0.02, n)).clip(0, 1)
        cur["energy"] = (cur["energy"] + 0.3).clip(0, 1)
    return cur


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path, share_threshold: float) -> dict:
    snap = Report([DataDriftPreset(drift_share=share_threshold)]).run(
        Dataset.from_pandas(current[FEATURES]), Dataset.from_pandas(reference[FEATURES])
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(out_html))
    count = next(m["value"] for m in snap.dict()["metrics"] if m["metric_name"].startswith("DriftedColumnsCount"))
    return {
        "drifted_columns": int(count["count"]),
        "share": float(count["share"]),
        "alert": bool(count["share"] >= share_threshold),
        "n_current": len(current),
    }


def score_drift_report(ref_scores: pd.Series, cur_scores: pd.Series, out_html: Path) -> dict:
    """Drift de predição: mesmo sem mudança nas features, o modelo pode passar a prever scores mais altos/baixos."""
    ref, cur = ref_scores.to_frame(SCORE), cur_scores.dropna().to_frame(SCORE)
    snap = Report([DataDriftPreset(drift_share=0.5)]).run(Dataset.from_pandas(cur), Dataset.from_pandas(ref))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(out_html))
    count = next(m["value"] for m in snap.dict()["metrics"] if m["metric_name"].startswith("DriftedColumnsCount"))
    return {
        "alert": bool(count["count"] >= 1),
        "mean_ref": float(ref[SCORE].mean()),
        "mean_current": float(cur[SCORE].mean()),
        "n_current": len(cur),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(ROOT / "data" / "prediction_logs"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--share", type=float, default=0.2, help="fração de features driftadas que dispara alerta")
    ap.add_argument("--synthetic", action="store_true", help="usa tráfego sintético deslocado")
    ap.add_argument("--out", default=str(ROOT / "reports" / "drift.html"))
    ap.add_argument("--out-scores", default=str(ROOT / "reports" / "drift_predictions.html"))
    ap.add_argument("--bundle", default=str(ROOT / "model" / "production" / "bundle.joblib"))
    a = ap.parse_args(argv)

    reference = pd.read_csv(repo_path(load_params()["data"]["processed"]))
    current = synthetic_production(reference) if a.synthetic else load_prediction_logs(Path(a.logs), a.days)
    if current.empty:
        print("Sem logs de predição na janela; nada a comparar.")
        return 0
    result = {"features": drift_report(reference, current, Path(a.out), a.share)}
    alert = result["features"]["alert"]

    bundle_path = Path(a.bundle)
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        # Referência e atual na modalidade só-áudio (a das demos): misturar com "histórico do artista" geraria falso drift.
        sample = reference.sample(min(5000, len(reference)), random_state=0)
        ref_scores = pd.Series(predict_score(bundle, sample, use_artist_context=False))
        cur_scores = None
        if SCORE in current and current[SCORE].notna().any():
            only_audio = current[MODE].eq("audio_only")  # logs sem modalidade (versões antigas) não entram
            cur_scores = current.loc[only_audio, SCORE] if only_audio.any() else None
        if cur_scores is None:  # tráfego sintético ou logs sem a modalidade: reavalia com o modelo em Production
            cur_scores = pd.Series(predict_score(bundle, current, use_artist_context=False))
        result["predictions"] = score_drift_report(ref_scores, cur_scores, Path(a.out_scores))
        alert = alert or result["predictions"]["alert"]
    print(json.dumps(result))
    return 1 if alert else 0


if __name__ == "__main__":
    sys.exit(main())
