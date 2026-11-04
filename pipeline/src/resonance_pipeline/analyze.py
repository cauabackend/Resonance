"""Rigor estatístico da EDA: testes de hipótese, multicolinearidade, sensibilidade ao threshold."""

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .common import FEATURES, GENRE, ROOT, TARGET, load_params, repo_path


def feature_tests(df: pd.DataFrame, alpha: float) -> list[dict]:
    """Mann-Whitney U por feature vs is_hit, com Bonferroni e FDR (BH) sobre as 9 features."""
    hit, miss = df[df[TARGET] == 1], df[df[TARGET] == 0]
    rows = []
    for f in FEATURES:
        u, p = mannwhitneyu(hit[f], miss[f], alternative="two-sided")
        r, _ = pointbiserialr(df[TARGET], df[f])
        rows.append({"feature": f, "p": p, "point_biserial": r, "rank_biserial": 2 * u / (len(hit) * len(miss)) - 1})
    ps = [r["p"] for r in rows]
    bonf = multipletests(ps, alpha=alpha, method="bonferroni")
    fdr = multipletests(ps, alpha=alpha, method="fdr_bh")
    for r, bs, bp, fs, fp in zip(rows, bonf[0], bonf[1], fdr[0], fdr[1]):
        r.update(p_bonferroni=bp, significant_bonferroni=bool(bs), p_fdr=fp, significant_fdr=bool(fs))
    return rows


def multicollinearity(df: pd.DataFrame) -> dict:
    X = StandardScaler().fit_transform(df[FEATURES])
    X = np.column_stack([np.ones(len(X)), X])
    vif = {f: float(variance_inflation_factor(X, i + 1)) for i, f in enumerate(FEATURES)}
    return {"correlation": df[FEATURES].corr().round(4).to_dict(), "vif": vif}


def threshold_sensitivity(df: pd.DataFrame, thresholds: list[int]) -> list[dict]:
    """Prevalência de hits e AUC de uma regressão logística para cada corte de popularidade."""
    X = StandardScaler().fit_transform(df[FEATURES])
    out = []
    for t in thresholds:
        y = (df["popularity"] >= t).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        auc = roc_auc_score(yte, LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
        out.append({"threshold": t, "hit_rate": float(y.mean()), "logreg_auc": float(auc)})
    return out


def genre_cutoff_sensitivity(df: pd.DataFrame, cutoffs: list[int]) -> list[dict]:
    """Quantos gêneros e faixas caem em 'infrequent' para cada corte de tamanho mínimo de amostra por gênero.

    O percentil é calculado por gênero ORIGINAL (antes do agrupamento), então o corte só afeta o encoder, mas um corte
    alto demais agrupa gêneros de nicho inteiros e apaga o que o modelo sabe sobre eles.
    """
    sizes = df[GENRE].value_counts()
    return [
        {
            "min_genre_count": c,
            "genres_kept": int((sizes >= c).sum()),
            "genres_grouped": int((sizes < c).sum()),
            "share_tracks_grouped": float(sizes[sizes < c].sum() / len(df)),
        }
        for c in cutoffs
    ]


def main() -> None:
    params = load_params()
    df = pd.read_csv(repo_path(params["data"]["processed"]))
    a = params["analysis"]
    report = {
        "feature_tests": feature_tests(df, a["alpha"]),
        "multicollinearity": multicollinearity(df),
        "threshold_sensitivity": threshold_sensitivity(df, a["sensitivity_thresholds"]),
        "genre_cutoff_sensitivity": genre_cutoff_sensitivity(df, a["genre_cutoffs"]),
    }
    out = ROOT / "reports" / "analysis.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in report["feature_tests"]:
        print(f"{r['feature']:<18} p_fdr={r['p_fdr']:.2e} rank_biserial={r['rank_biserial']:+.3f}")


if __name__ == "__main__":
    main()
