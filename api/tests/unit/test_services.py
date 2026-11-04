import json

import numpy as np
import pytest
from resonance_features import expected_popularity

from app.core.rate_limit import RateLimiter
from app.models_registry.loader import load_bundle, resolve_uri
from app.schemas.track import TrackFeatures
from app.services.inference import InferenceService
from app.services.prediction_log import PredictionLogger
from app.services.search import CatalogService
from tests.conftest import SAMPLE, WITH_ARTIST


@pytest.fixture
def svc(bundle_dir):
    bundle, digest = load_bundle(bundle_dir)
    return InferenceService(bundle, digest)


def test_batch_matches_single(svc):
    t = TrackFeatures(**SAMPLE)
    batch, modes = svc.score([t, t])
    assert batch[0] == pytest.approx(svc.predict(t).percentile_score) and batch[0] == batch[1]
    assert ((batch >= 0) & (batch <= 100)).all() and set(modes) == {"audio_only"}


def test_mode_follows_the_artist_history(svc):
    plain, rich = TrackFeatures(**SAMPLE), TrackFeatures(**WITH_ARTIST)
    assert svc.predict(plain).mode == "audio_only" and svc.predict(rich).mode == "artist_aware"
    # use_artist_context=False ignora o histórico enviado: é a pergunta "e se o artista fosse novo?"
    off = svc.predict(rich, use_artist=False)
    assert off.mode == "audio_only" and off.percentile_score == pytest.approx(svc.predict(plain).percentile_score)
    assert svc.predict(rich).percentile_score > off.percentile_score  # artista de popularidade 75 sobe a faixa


def test_probability_uses_expected_popularity_and_the_modes_calibrator(svc):
    for payload in (SAMPLE, WITH_ARTIST):
        t = TrackFeatures(**payload)
        out = svc.predict(t)
        ep = expected_popularity(np.array([out.percentile_score]), np.array([t.track_genre]), svc.stats)[0]
        assert out.expected_popularity == pytest.approx(ep)
        assert out.probability == pytest.approx(float(svc.bundle["calibrators"][out.mode].predict([ep])[0]))
        assert out.threshold == svc.bundle["thresholds"][out.mode]


def test_explanation_merges_columns_into_factors(svc):
    out = svc.predict(TrackFeatures(**WITH_ARTIST))
    names = [f.feature for f in out.factors]
    assert names.count("track_genre") == 1 and names.count("artist_context") == 1
    assert "danceability" in names and "danceability_z" not in names  # x e x_z viram um fator só
    assert out.top_factors == out.factors[:3]
    impacts = [abs(f.shap_value) for f in out.factors]
    assert impacts == sorted(impacts, reverse=True)
    artist = next(f for f in out.factors if f.feature == "artist_context")
    assert artist.value == 75.0
    assert next(f for f in svc.predict(TrackFeatures(**SAMPLE)).factors if f.feature == "artist_context").value == "n/d"


def test_unseen_genre_still_predicts(svc):
    assert svc.predict(TrackFeatures(**{**SAMPLE, "track_genre": "inedito"})).model_version == "test-v1"


def test_threshold_decides_is_hit(svc):
    out = svc.predict(TrackFeatures(**SAMPLE))
    assert out.is_hit == (out.probability >= out.threshold)


def test_metadata_only_numeric_metrics(svc):
    assert svc.metadata().metrics == {"spearman": 0.5, "ndcg_at_k": 0.9, "brier": 0.05, "ece": 0.01}


def test_loader_pin_and_missing(tmp_path):
    assert resolve_uri("models:/resonance@production", "3") == "models:/resonance/3"
    assert resolve_uri("models:/resonance@production") == "models:/resonance@production"
    assert resolve_uri("/some/dir", "3") == "/some/dir"
    with pytest.raises(FileNotFoundError):
        load_bundle(str(tmp_path))


def test_search_and_get(dataset, tmp_path):
    csv = tmp_path / "t.csv"
    dataset.to_csv(csv, index=False)
    cat = CatalogService.from_csv(str(csv))
    assert len(cat.search("song 1", 5)) == 5
    assert cat.search("zzzz-nada") == []
    assert cat.get("id3").name == "Song 3" and cat.get("nope") is None


def test_prediction_logger_partitions_by_date(tmp_path):
    PredictionLogger(str(tmp_path)).log("rid", "v1", [({"danceability": 0.5}, 61.5, 0.4, True, "audio_only")])
    (part,) = tmp_path.glob("dt=*")
    row = json.loads((part / "predictions.jsonl").read_text())
    assert row["model_version"] == "v1" and row["features"] == {"danceability": 0.5}
    PredictionLogger("").log("rid", "v1", [({}, 5.0, 0.1, False, "audio_only")])  # desligado: não escreve nada
    assert np.isclose(row["probability"], 0.4) and row["percentile_score"] == 61.5


def test_expected_sha256_blocks_tampered_bundle_before_unpickling(bundle_dir):
    _, digest = load_bundle(bundle_dir)
    assert load_bundle(bundle_dir, expected_sha256=digest.upper())[1] == digest
    with pytest.raises(RuntimeError, match="sha256"):
        load_bundle(bundle_dir, expected_sha256="0" * 64)


def test_rate_limiter_sliding_window():
    rl = RateLimiter(2, window=10)
    assert rl.allow("a", now=0) and rl.allow("a", now=1) and not rl.allow("a", now=2)
    assert rl.allow("b", now=2)  # outro cliente tem a própria cota
    assert rl.allow("a", now=11)  # a janela deslizou
    assert RateLimiter(0).allow("x") and RateLimiter(0).allow("x")  # 0 = desligado
