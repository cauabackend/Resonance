"""Calibração do badge: percentile_score -> P(is_hit histórico), em fold separado do treino e do teste.

Isotônica (não Platt): ver docs/adr/0007. O threshold do badge é escolhido por custo de negócio, não por 0.5.
"""

from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def best_threshold(y: np.ndarray, prob: np.ndarray, fn_cost: float, fp_cost: float) -> float:
    """Threshold do badge que minimiza o custo de negócio (FN custa mais que FP), não 0.5."""
    cands = np.unique(prob)
    costs = [fn_cost * ((prob < t) & (y == 1)).sum() + fp_cost * ((prob >= t) & (y == 0)).sum() for t in cands]
    return float(cands[int(np.argmin(costs))])


def fit_calibrator(
    score: np.ndarray, y: np.ndarray, fn_cost: float, fp_cost: float
) -> tuple[IsotonicRegression, float]:
    """Ajusta a isotônica no fold de calibração e devolve também o threshold do badge (em probabilidade)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score, y)
    return iso, best_threshold(np.asarray(y), iso.predict(score), fn_cost, fp_cost)


def calibration(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> dict:
    """Brier + ECE + pontos do reliability diagram."""
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            table.append({"mean_pred": float(prob[m].mean()), "frac_pos": float(y[m].mean()), "n": int(m.sum())})
            ece += m.mean() * abs(prob[m].mean() - y[m].mean())
    return {"brier": float(brier_score_loss(y, prob)), "ece": float(ece), "reliability": table}


def brier_before_after(y: np.ndarray, score: np.ndarray, prob: np.ndarray) -> dict:
    """Brier antes (score/100 usado como probabilidade) e depois da calibração, mais a taxa-base como piso."""
    return {
        "brier_uncalibrated": float(brier_score_loss(y, np.clip(score / 100, 0, 1))),
        "brier": float(brier_score_loss(y, prob)),
        "brier_base_rate": float(brier_score_loss(y, np.full(len(y), np.mean(y)))),
    }


def reliability_plot(table: list[dict], out: Path) -> Path:
    """Reliability diagram monocromático (artefato visual do MLflow)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    ax.plot([0, 1], [0, 1], color="#4a4a4a", linestyle="--", label="calibração perfeita")
    ax.plot(
        [r["mean_pred"] for r in table], [r["frac_pos"] for r in table], color="#ffffff", marker="o", label="modelo"
    )
    for r in table:
        ax.annotate(str(r["n"]), (r["mean_pred"], r["frac_pos"]), color="#9a9a9a", fontsize=7, xytext=(4, -9),
                    textcoords="offset points")  # fmt: skip
    ax.set(xlabel="probabilidade prevista", ylabel="fração observada de hits", xlim=(0, 1), ylim=(0, 1))
    for item in (ax.xaxis.label, ax.yaxis.label, ax.title):
        item.set_color("#d4d4d4")
    ax.tick_params(colors="#9a9a9a")
    for s in ax.spines.values():
        s.set_color("#4a4a4a")
    ax.legend(facecolor="#1a1a1a", edgecolor="#4a4a4a", labelcolor="#d4d4d4")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
