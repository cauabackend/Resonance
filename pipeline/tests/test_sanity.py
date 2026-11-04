"""Rede de segurança contra regressões silenciosas: o ranking precisa ter o formato certo.

Um perfil sabidamente ruim cai para o fundo, um sabidamente bom sobe para o topo, e faixas de ruído puro ficam
no meio (nunca no topo). Falha no CI se o modelo (ou o pré-processamento) quebrar a direção do sinal.
"""

import numpy as np
import pandas as pd
import pytest

from resonance_pipeline import train
from resonance_pipeline.common import FEATURES, GENRE


@pytest.fixture
def scored(clean_df, params):
    tr, te = train.split(clean_df, params)
    bundle = train.fit_model(tr, params, {"max_depth": 3, "n_estimators": 60})
    return bundle, te


def _profile(base: pd.DataFrame, **over) -> pd.DataFrame:
    row = base[[*FEATURES, GENRE]].head(1).copy()
    for k, v in over.items():
        row[k] = v
    return row


def test_worst_profile_sinks_and_best_profile_rises(scored):
    bundle, te = scored
    ref = np.sort(train.predict_score(bundle, te))
    worst = train.predict_score(bundle, _profile(te, danceability=0.0, loudness=-30.0, track_genre="pop"))[0]
    best = train.predict_score(bundle, _profile(te, danceability=1.0, loudness=0.0, track_genre="pop"))[0]
    assert worst <= np.percentile(ref, 10)
    assert best >= np.percentile(ref, 90)


def test_pure_noise_tracks_do_not_reach_the_top(scored):
    bundle, te = scored
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(
        {
            **{f: rng.random(200) for f in FEATURES},
            "tempo": rng.uniform(60, 200, 200),
            "loudness": rng.uniform(-30, 0, 200),
            "duration_ms": rng.integers(120_000, 400_000, 200),
            GENRE: rng.choice(["pop", "rock", "jazz"], 200),
        }
    )
    ref = train.predict_score(bundle, te)
    noise_scores = train.predict_score(bundle, noise)
    assert np.median(noise_scores) == pytest.approx(np.median(ref), abs=float(0.25 * ref.std() + 5))
    assert (noise_scores >= np.percentile(ref, 99)).mean() < 0.1
