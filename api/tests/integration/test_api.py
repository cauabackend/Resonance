import json

from tests.conftest import SAMPLE, WITH_ARTIST


def test_predict(client):
    r = client.post("/v1/predict", json={"features": SAMPLE})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1 and 0 <= body["percentile_score"] <= 100
    assert body["model_version"] == "test-v1" and body["mode"] == "audio_only"
    assert 0 <= body["expected_popularity"] <= 100
    assert len(body["top_factors"]) == 3
    assert {f["feature"] for f in body["factors"]} >= {"danceability", "track_genre", "artist_context", "liveness"}
    assert r.headers["x-request-id"]


def test_predict_with_artist_history_uses_the_other_mode(client):
    r = client.post("/v1/predict", json={"features": WITH_ARTIST}).json()
    off = client.post("/v1/predict", json={"features": WITH_ARTIST, "use_artist_context": False}).json()
    assert r["mode"] == "artist_aware" and off["mode"] == "audio_only"
    assert r["percentile_score"] > off["percentile_score"]  # histórico de popularidade 75 puxa o percentil para cima


def test_request_id_is_propagated(client):
    r = client.get("/v1/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_predict_validates_contract(client):
    assert client.post("/v1/predict", json={"features": {**SAMPLE, "energy": 1.5}}).status_code == 422
    assert client.post("/v1/predict", json={"features": {**SAMPLE, "duration_ms": 0}}).status_code == 422
    for bad in ({"artist_mean_popularity": 101}, {"artist_hit_share": 1.5}, {"mode": 2}, {"liveness": -0.1}):
        assert client.post("/v1/predict", json={"features": {**SAMPLE, **bad}}).status_code == 422
    assert (
        client.post("/v1/predict", json={"features": {**SAMPLE, "tempo": 5}}).status_code == 422
    )  # = schema de treino
    bad = {k: v for k, v in SAMPLE.items() if k != "track_genre"}
    assert client.post("/v1/predict", json={"features": bad}).status_code == 422


def test_batch_is_ranked_by_percentile_score(client):
    tracks = [{**SAMPLE, "danceability": d} for d in (0.1, 0.9, 0.5)]
    body = client.post("/v1/predict/batch", json={"tracks": tracks}).json()
    scores = [r["percentile_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)  # já ordenado
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    assert sorted(r["input_index"] for r in body["results"]) == [0, 1, 2]
    # rank[i] = posição da i-ésima faixa ENVIADA; o item de input_index i carrega o mesmo rank
    for r in body["results"]:
        assert body["rank"][r["input_index"]] == r["rank"]
    assert body["results"][0]["input_index"] == 1  # a mais dançante vem primeiro
    assert set(body["thresholds"]) == {"artist_aware", "audio_only"}


def test_batch_mixes_modes_per_track(client):
    body = client.post("/v1/predict/batch", json={"tracks": [SAMPLE, WITH_ARTIST]}).json()
    modes = {r["input_index"]: r["mode"] for r in body["results"]}
    assert modes == {0: "audio_only", 1: "artist_aware"}
    body_off = client.post("/v1/predict/batch", json={"tracks": [WITH_ARTIST], "use_artist_context": False}).json()
    assert body_off["results"][0]["mode"] == "audio_only"


def test_batch_limits(client):
    assert client.post("/v1/predict/batch", json={"tracks": [SAMPLE] * 6}).status_code == 413  # max=5 no fixture
    assert client.post("/v1/predict/batch", json={"tracks": []}).status_code == 422


def test_predictions_are_logged_for_drift(client, settings):
    client.post("/v1/predict", json={"features": SAMPLE})
    client.post("/v1/predict", json={"features": SAMPLE, "source": "catalog"})  # navegação: não loga
    client.post("/v1/predict/batch", json={"tracks": [SAMPLE, SAMPLE]})
    (part,) = (p for p in __import__("pathlib").Path(settings.prediction_log_dir).glob("dt=*"))
    lines = (part / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["features"]["track_genre"] == "pop"
    assert "percentile_score" in json.loads(lines[0]) and json.loads(lines[0])["mode"] == "audio_only"


def test_search_and_track(client):
    hits = client.get("/v1/tracks/search", params={"q": "song 12"}).json()
    assert hits and hits[0]["name"].lower().startswith("song 12")
    assert client.get(f"/v1/tracks/{hits[0]['track_id']}").json()["features"]["track_genre"]
    assert client.get("/v1/tracks/nope").status_code == 404
    assert client.get("/v1/tracks/search", params={"q": "a"}).status_code == 422


def test_metadata_and_health(client):
    meta = client.get("/v1/model/metadata").json()
    assert meta["version"] == "test-v1" and meta["baseline_ridge_ndcg_at_k"] == 0.8 and meta["hit_means"]
    assert meta["baseline_artist_mean_ndcg_at_k"] == 0.85
    h = client.get("/v1/health")
    assert h.status_code == 200 and h.json() == {"status": "ok", "model_version": "test-v1", "catalog_loaded": True}


def test_degraded_mode(degraded_client):
    h = degraded_client.get("/v1/health")
    assert h.status_code == 503 and h.json()["status"] == "degraded"
    assert degraded_client.post("/v1/predict", json={"features": SAMPLE}).status_code == 503
    assert degraded_client.get("/v1/tracks/search", params={"q": "ab"}).status_code == 503


def test_openapi_documents_v1_contract(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/predict", "/v1/predict/batch", "/v1/tracks/search", "/v1/tracks/{track_id}",
            "/v1/model/metadata", "/v1/health"} <= set(paths)  # fmt: skip


def test_deep_health_detects_registry_change(client, settings, tmp_path):
    ok = client.get("/v1/health/deep")
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    assert ok.json()["loaded_sha256"] == ok.json()["registry_sha256"]
    # o registry passa a servir outro artefato: o processo segue de pé, mas o health profundo acusa
    other = tmp_path / "other"
    other.mkdir()
    (other / "bundle.joblib").write_bytes(b"outro artefato")
    client.app.state.settings.model_registry_uri = str(other)
    bad = client.get("/v1/health/deep")
    assert bad.status_code == 409 and bad.json()["status"] == "mismatch"


def test_deep_health_degraded_without_model(degraded_client):
    r = degraded_client.get("/v1/health/deep")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


def test_rate_limit_returns_429_but_not_for_health(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings.rate_limit_per_minute = 3
    with TestClient(create_app(settings)) as c:
        codes = [c.get("/v1/tracks/search", params={"q": "song"}).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        assert c.get("/v1/tracks/search", params={"q": "song"}).headers["retry-after"] == "60"
        assert c.get("/v1/health").status_code == 200


def test_latency_budget_p95_under_200ms(client):
    """Orçamento de performance (docs/performance-budget.md): p95 de /v1/predict < 200 ms, com SHAP."""
    import statistics
    import time

    times = []
    for _ in range(60):
        t = time.perf_counter()
        assert client.post("/v1/predict", json={"features": SAMPLE}).status_code == 200
        times.append((time.perf_counter() - t) * 1000)
    assert statistics.quantiles(times[5:], n=100)[94] < 200  # ignora o aquecimento


def test_deep_health_is_cached_and_rate_limited(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings.rate_limit_per_minute = 2
    with TestClient(create_app(settings)) as c:
        assert c.get("/v1/health").status_code == 200  # só /v1/health é isento
        assert [c.get("/v1/health/deep").status_code for _ in range(3)] == [200, 200, 429]
        sha = c.app.state.deep_cache[2]
        assert sha == c.app.state.inference.sha256  # hash do registry cacheado (não rebaixa o artefato a cada chamada)
