import numpy as np
import pandas as pd
import shap
from resonance_features import FACTOR_VALUE, aggregate_shap

from app.schemas.prediction import ShapFactor


class ExplainService:
    """TreeExplainer criado UMA vez por versão de modelo (o app mantém uma instância por bundle)."""

    def __init__(self, bundle: dict):
        self.explainer = shap.TreeExplainer(bundle["model"])
        self.groups: dict[str, list[int]] = bundle["factor_groups"]
        self.preprocessor = bundle["preprocessor"]

    def explain(self, engineered: pd.DataFrame) -> list[ShapFactor]:
        """SHAP de uma linha já engenheirada; colunas do mesmo fator (x e x_z; as 3 de artista; os one-hots) somadas."""
        sv = np.asarray(self.explainer.shap_values(self.preprocessor.transform(engineered)))
        names, mat = aggregate_shap(sv, self.groups)
        row = engineered.iloc[0]
        factors = []
        for name, s in zip(names, mat[0]):
            raw = row.get(FACTOR_VALUE[name]) if name in FACTOR_VALUE else None
            value: float | str = "n/d" if raw is None or (not isinstance(raw, str) and pd.isna(raw)) else raw
            factors.append(
                ShapFactor(
                    feature=name,
                    value=value if isinstance(value, str) else float(value),
                    shap_value=float(s),
                    direction="increases" if s > 0 else "decreases",
                )
            )
        return sorted(factors, key=lambda f: abs(f.shap_value), reverse=True)
