import os
import re
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Resonance",
    page_icon="🎵",
    layout="centered",
)

FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "loudness", "speechiness", "acousticness",
    "instrumentalness", "duration_ms",
]

FEATURE_LABELS = {
    "danceability": "Dançabilidade",
    "energy": "Energia",
    "valence": "Positividade",
    "tempo": "BPM",
    "loudness": "Volume (dB)",
    "speechiness": "Presença vocal",
    "acousticness": "Acústico",
    "instrumentalness": "Instrumental",
    "duration_ms": "Duração (ms)",
}


@st.cache_resource
def load_model():
    """Carrega modelo treinado."""
    return joblib.load("model/model.pkl")


@st.cache_data
def load_dataset():
    """Carrega dataset processado pra buscas e comparações."""
    return pd.read_csv("data/processed/tracks_clean.csv")


@st.cache_data
def get_hit_means():
    """Médias das features dos hits pra comparação no radar."""
    df = load_dataset()
    return df[df["is_hit"] == 1][FEATURES].mean()


def search_tracks(query, df, max_results=10):
    """Busca músicas no dataset por nome ou artista."""
    query_lower = query.lower()
    mask = (
        df["name"].str.lower().str.contains(query_lower, na=False)
        | df["artist"].str.lower().str.contains(query_lower, na=False)
    )
    return df[mask].head(max_results)


def explain_prediction(model, track_features, feature_names):
    """Retorna os 3 fatores que mais pesaram na predição."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(track_features)[0]

    factors = sorted(
        zip(feature_names, sv, track_features[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    explanations = []
    for feat, shap_val, feat_val in factors:
        direction = "aumenta" if shap_val > 0 else "diminui"
        label = FEATURE_LABELS.get(feat, feat)
        explanations.append({
            "feature": label,
            "value": feat_val,
            "direction": direction,
            "impact": abs(shap_val),
        })
    return explanations


def build_radar_chart(track_data, hit_means):
    """Radar chart comparando a música com a média dos hits."""
    radar_feats = [f for f in FEATURES if f not in ("duration_ms", "tempo", "loudness")]
    labels = [FEATURE_LABELS.get(f, f) for f in radar_feats]

    track_vals = [track_data[f] for f in radar_feats]
    hit_vals = [hit_means[f] for f in radar_feats]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=track_vals, theta=labels, fill="toself",
        name="Esta música", opacity=0.6,
        line_color="#636EFA",
    ))
    fig.add_trace(go.Scatterpolar(
        r=hit_vals, theta=labels, fill="toself",
        name="Média dos hits", opacity=0.3,
        line_color="#EF553B",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400, template="plotly_white",
        margin=dict(t=40, b=40),
    )
    return fig


def build_shap_waterfall(model, track_features):
    """Waterfall SHAP da predição."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(track_features)[0]

    sorted_idx = np.argsort(np.abs(sv))
    labels = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in sorted_idx]
    values = sv[sorted_idx]

    fig = go.Figure(go.Waterfall(
        y=labels, x=values, orientation="h",
        connector={"line": {"color": "gray"}},
    ))
    fig.update_layout(
        xaxis_title="Impacto na predição",
        height=350, template="plotly_white",
        margin=dict(t=20, b=40),
    )
    return fig


def build_gauge(prob):
    """Gauge/velocímetro da probabilidade de hit."""
    color = "#2ecc71" if prob >= 0.5 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#fadbd8"},
                {"range": [30, 50], "color": "#fdebd0"},
                {"range": [50, 70], "color": "#d5f5e3"},
                {"range": [70, 100], "color": "#abebc6"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=10, l=30, r=30))
    return fig


def build_feature_bars(track_data, hit_means):
    """Barras horizontais comparando a música com a média dos hits."""
    bar_feats = [f for f in FEATURES if f not in ("duration_ms", "tempo", "loudness")]
    labels = [FEATURE_LABELS.get(f, f) for f in bar_feats]
    track_vals = [track_data[f] for f in bar_feats]
    hit_vals = [hit_means[f] for f in bar_feats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=track_vals, orientation="h",
        name="Esta música", marker_color="#636EFA", opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=hit_vals, orientation="h",
        name="Média dos hits", marker_color="#EF553B", opacity=0.5,
    ))
    fig.update_layout(
        barmode="group", height=350, template="plotly_white",
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(range=[0, 1]),
    )
    return fig


def build_feature_table(track_data, hit_means):
    """Tabela com os valores numéricos das features."""
    rows = []
    for f in FEATURES:
        val = track_data[f]
        hit_val = hit_means[f]
        diff = val - hit_val
        if f == "duration_ms":
            val_str = f"{val/1000:.0f}s"
            hit_str = f"{hit_val/1000:.0f}s"
            diff_str = f"{diff/1000:+.0f}s"
        else:
            val_str = f"{val:.3f}"
            hit_str = f"{hit_val:.3f}"
            diff_str = f"{diff:+.3f}"
        rows.append({
            "Feature": FEATURE_LABELS.get(f, f),
            "Música": val_str,
            "Média hits": hit_str,
            "Diferença": diff_str,
        })
    return pd.DataFrame(rows)


def show_prediction(track_data, model, hit_means):
    """Mostra a predição completa pra uma música."""
    features_array = np.array([[track_data[f] for f in FEATURES]])
    prob = model.predict_proba(features_array)[0][1]
    is_hit = prob >= 0.5

    st.markdown("---")
    st.subheader(f"{track_data['name']}")
    st.caption(f"por {track_data['artist']}")

    if "track_genre" in track_data and pd.notna(track_data.get("track_genre")):
        st.caption(f"Gênero: {track_data['track_genre']}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(build_gauge(prob), use_container_width=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_hit:
            st.success("Potencial de hit")
        else:
            st.warning("Baixo potencial")
        st.metric("Probabilidade", f"{prob:.1%}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Radar", "Comparação", "Valores", "Por que essa decisão?",
    ])

    with tab1:
        st.plotly_chart(build_radar_chart(track_data, hit_means), use_container_width=True)

    with tab2:
        st.plotly_chart(build_feature_bars(track_data, hit_means), use_container_width=True)

    with tab3:
        table = build_feature_table(track_data, hit_means)
        st.dataframe(table, use_container_width=True, hide_index=True)

    with tab4:
        st.plotly_chart(build_shap_waterfall(model, features_array), use_container_width=True)

        st.markdown("**Fatores que mais pesaram:**")
        explanations = explain_prediction(model, features_array, FEATURES)
        for exp in explanations:
            symbol = "↑" if exp["direction"] == "aumenta" else "↓"
            st.write(
                f"{symbol} **{exp['feature']}** = {exp['value']:.3f} — "
                f"{exp['direction']} a chance de hit"
            )


# --- App ---

st.title("Resonance")
st.caption("Descubra o potencial de hit de uma música.")

model = load_model()
df = load_dataset()
hit_means = get_hit_means()

search_query = st.text_input(
    "Buscar por nome da música ou artista",
    placeholder="Ex: Bohemian Rhapsody, Drake, Billie Eilish...",
)

if search_query:
    results = search_tracks(search_query, df)

    if len(results) == 0:
        st.info("Nenhuma música encontrada. Tenta outro termo.")
    else:
        options = [
            f"{row['name']} — {row['artist']}"
            for _, row in results.iterrows()
        ]
        selected = st.selectbox("Selecione a música:", options)

        if st.button("Analisar"):
            idx = options.index(selected)
            track = results.iloc[idx]
            show_prediction(track, model, hit_means)

st.divider()
st.caption("Resonance — [GitHub](https://github.com/seu-usuario/music-hit-predictor)")