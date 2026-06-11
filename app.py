"""RecruiterIQ — interactive candidate ranking dashboard.

Re-ranking is live: precomputed artifacts make scoring 100K candidates a
single matrix multiply, so weight sliders and custom job descriptions
re-rank instantly without any pipeline rerun.
"""

import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    ARTIFACTS_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    WEIGHTS,
)
from engine import (
    SUBSCORE_ORDER,
    build_matrices,
    compute_scores,
    mmr_rerank,
    stability_analysis,
    top_k_indices,
)
from evidence import collect_evidence, generate_reasoning
from rank import load_candidates_by_ids

st.set_page_config(
    page_title="RecruiterIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#4F8EF7"
GREEN = "#3ECF8E"
AMBER = "#F59E0B"
RED = "#EF4444"
MUTED = "#94A3B8"

SUBSCORE_LABELS = {
    "technical_fit": "Technical fit",
    "career_quality": "Career quality",
    "availability_signal": "Availability",
    "seniority_fit": "Seniority fit",
    "semantic_similarity": "Semantic match",
}


# ---------------------------------------------------------------- data layer

@st.cache_resource(show_spinner="Loading artifacts ...")
def load_artifacts():
    artifacts = {}
    artifacts["embeddings"] = np.load(str(ARTIFACTS_DIR / "embeddings.npy")).astype(np.float32)
    artifacts["candidate_ids"] = np.load(
        str(ARTIFACTS_DIR / "candidate_ids.npy"), allow_pickle=True
    )
    artifacts["jd_embedding"] = np.load(str(ARTIFACTS_DIR / "jd_embedding.npy")).astype(np.float32)
    with open(ARTIFACTS_DIR / "subscores.pkl", "rb") as f:
        artifacts["subscores"] = pickle.load(f)
    with open(ARTIFACTS_DIR / "disqualified.json", "r") as f:
        artifacts["disqualified"] = json.load(f)
    return artifacts


@st.cache_resource(show_spinner="Packing score matrices ...")
def get_matrices():
    a = load_artifacts()
    subscore_matrix, penalties = build_matrices(a["candidate_ids"], a["subscores"])
    return subscore_matrix, penalties


@st.cache_resource(show_spinner="Loading embedding model (first custom JD only) ...")
def get_model():
    from sentence_transformers import SentenceTransformer

    # CPU is fine for embedding a single query string.
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@st.cache_data(show_spinner=False)
def embed_text(text: str) -> np.ndarray:
    return get_model().encode(text, normalize_embeddings=True).astype(np.float32)


@st.cache_data(show_spinner="Loading candidate profiles ...")
def cached_candidates(ids: tuple) -> dict:
    return load_candidates_by_ids(ids)


@st.cache_data(show_spinner=False)
def cached_stability(weights_key: tuple, jd_key: str, k: int) -> dict:
    a = load_artifacts()
    subscore_matrix, penalties = get_matrices()
    weights = dict(weights_key)
    jd_emb = st.session_state.get("jd_embedding_override")
    if jd_emb is None:
        jd_emb = a["jd_embedding"]
    semantic_sim = a["embeddings"] @ jd_emb
    return stability_analysis(subscore_matrix, penalties, semantic_sim, weights, k)


# ---------------------------------------------------------------- ui helpers

def score_bar_html(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(float(value), 1.0)) * 100
    return f"""
    <div style="margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:{MUTED}">
        <span>{label}</span><span>{value:.0%}</span>
      </div>
      <div style="background:#262b3d;border-radius:6px;height:8px">
        <div style="background:{color};width:{pct:.1f}%;height:8px;border-radius:6px"></div>
      </div>
    </div>
    """


def chip(text: str, color: str = ACCENT, title: str = "") -> str:
    return (
        f'<span title="{title}" style="background:{color}22;color:{color};'
        f"border:1px solid {color}55;border-radius:12px;padding:2px 10px;"
        f'margin:2px;font-size:0.75rem;display:inline-block">{text}</span>'
    )


def stability_badge(freq: float) -> str:
    if freq >= 0.90:
        return chip(f"stable {freq:.0%}", GREEN, "Stays in top-100 under ±20% weight perturbation")
    if freq >= 0.60:
        return chip(f"moderate {freq:.0%}", AMBER, "Sensitive to weight choices")
    return chip(f"fragile {freq:.0%}", RED, "Only in top-100 for some weightings")


def normalized_weights() -> dict:
    raw = {name: st.session_state.get(f"w_{name}", WEIGHTS[name]) for name in WEIGHTS}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


# ------------------------------------------------------------------- sidebar

def render_sidebar(artifacts_ready: bool, n_candidates: int, n_disqualified: int) -> dict:
    with st.sidebar:
        st.markdown("## 🎯 RecruiterIQ")
        st.caption("Live candidate ranking — every control re-ranks instantly.")

        st.markdown("### Job description")
        jd_text = st.text_area(
            "Custom JD or natural-language query",
            height=110,
            placeholder='e.g. "RAG engineer, 5+ yrs, strong vector search, short notice"',
            label_visibility="collapsed",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Apply JD", type="primary", use_container_width=True, disabled=not jd_text.strip()):
                st.session_state["jd_embedding_override"] = embed_text(jd_text.strip())
                st.session_state["jd_label"] = jd_text.strip()[:60]
        with col_b:
            if st.button("Default JD", use_container_width=True):
                st.session_state.pop("jd_embedding_override", None)
                st.session_state.pop("jd_label", None)
        if "jd_label" in st.session_state:
            st.info(f'Ranking against: "{st.session_state["jd_label"]}…"')

        st.markdown("### Signal weights")
        st.caption("Drag to see the shortlist reshuffle live.")
        for name in WEIGHTS:
            st.slider(
                SUBSCORE_LABELS[name],
                min_value=0.0,
                max_value=1.0,
                value=float(WEIGHTS[name]),
                step=0.01,
                key=f"w_{name}",
            )
        weights = normalized_weights()
        if st.button("Reset weights", use_container_width=True):
            for name in WEIGHTS:
                st.session_state[f"w_{name}"] = float(WEIGHTS[name])
            st.rerun()
        st.caption(
            "Effective (normalized): "
            + " · ".join(f"{SUBSCORE_LABELS[k].split()[0]} {v:.0%}" for k, v in weights.items())
        )

        st.markdown("### Shortlist")
        diversity = st.slider(
            "Relevance ↔ Diversity (MMR)",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05,
            help="0 = pure score ranking. Higher values penalize near-duplicate "
            "profiles so the shortlist covers distinct candidate archetypes.",
        )
        anonymized = st.toggle(
            "🕶️ Blind screening mode",
            value=False,
            help="Hides names, companies and institutions to reduce reviewer bias.",
        )

        st.divider()
        if artifacts_ready:
            c1, c2 = st.columns(2)
            c1.metric("Candidates", f"{n_candidates:,}")
            c2.metric("Disqualified", n_disqualified)
        else:
            st.warning("Artifacts not loaded — run `python precompute.py` first.")

    return {"weights": weights, "diversity": diversity, "anonymized": anonymized}


# ------------------------------------------------------------------ ranking

def run_ranking(artifacts, weights: dict, diversity: float):
    subscore_matrix, penalties = get_matrices()
    jd_emb = st.session_state.get("jd_embedding_override")
    if jd_emb is None:
        jd_emb = artifacts["jd_embedding"]
    semantic_sim = artifacts["embeddings"] @ jd_emb
    scores = compute_scores(subscore_matrix, penalties, semantic_sim, weights)

    if diversity > 0:
        pool = top_k_indices(scores, TOP_K * 5)
        lambda_rel = 1.0 - diversity
        top_idx = mmr_rerank(pool, scores, artifacts["embeddings"], lambda_rel, TOP_K)
        top_idx = np.array(top_idx)
    else:
        top_idx = top_k_indices(scores, TOP_K)

    return top_idx, scores, semantic_sim, subscore_matrix


# ------------------------------------------------------------ shortlist tab

def candidate_display_name(cand: dict, rank: int, anonymized: bool) -> str:
    profile = cand.get("profile", {})
    if anonymized:
        return f"Candidate #{rank:03d} — {profile.get('current_title', '?')}"
    return (
        f"{profile.get('anonymized_name', '?')} — "
        f"{profile.get('current_title', '?')} @ {profile.get('current_company', '?')}"
    )


def render_evidence(ev: dict):
    chips = []
    for m in ev["matched"]:
        color = GREEN if m["group"] == "must-have" else ACCENT
        label = f"✓ {m['criterion']}"
        if m["source"] == "skill":
            label += f" ← {m['detail']}"
        chips.append(chip(label, color, title=str(m.get("detail", ""))))
    for miss in ev["missing_must_haves"]:
        chips.append(chip(f"✗ {miss['criterion']}", RED, "Missing must-have"))
    if ev["production"]:
        kws = ", ".join(dict.fromkeys(h["keyword"] for h in ev["production"][:3]))
        chips.append(chip(f"🚀 production: {kws}", AMBER))
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    text_hits = [m for m in ev["matched"] if m["source"] == "text" and m["detail"]]
    snippets = [h for h in ev["production"] if h["snippet"]][:1] + [
        {"keyword": m["keyword"], "snippet": m["detail"]} for m in text_hits[:2]
    ]
    if snippets:
        with st.container():
            for s in snippets[:3]:
                st.markdown(
                    f'<div style="border-left:3px solid {ACCENT};padding:4px 10px;'
                    f'margin:4px 0;color:{MUTED};font-size:0.8rem">'
                    f'<b style="color:{ACCENT}">{s["keyword"]}</b>: "{s["snippet"]}"</div>',
                    unsafe_allow_html=True,
                )


def render_shortlist(artifacts, top_idx, scores, semantic_sim, controls, stability):
    ids = artifacts["candidate_ids"]
    subs = artifacts["subscores"]
    top_ids = [str(ids[i]) for i in top_idx]
    candidates_by_id = cached_candidates(tuple(top_ids))

    header_l, header_r = st.columns([3, 1])
    with header_l:
        st.subheader(f"Top {len(top_idx)} candidates")
    with header_r:
        show_n = st.selectbox("Show", [25, 50, 100], index=0, label_visibility="collapsed")

    for rank_pos, i in enumerate(top_idx[:show_n]):
        rank = rank_pos + 1
        cid = str(ids[i])
        cand = candidates_by_id.get(cid)
        if cand is None:
            continue
        ss = subs.get(cid, {})
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        ev = collect_evidence(cand)

        title_line = candidate_display_name(cand, rank, controls["anonymized"])
        with st.expander(
            f"#{rank}  ·  {title_line}  ·  score {scores[i]:.3f}",
            expanded=(rank <= 3),
        ):
            col1, col2 = st.columns([1.1, 1.6])
            with col1:
                badge_bits = [stability_badge(stability.get(int(i), 0.0))]
                penalty = ss.get("penalty_multiplier", 1.0)
                if penalty < 1.0:
                    badge_bits.append(chip(f"⚠ penalty ×{penalty:.2f}", RED))
                if signals.get("open_to_work_flag"):
                    badge_bits.append(chip("open to work", GREEN))
                st.markdown(" ".join(badge_bits), unsafe_allow_html=True)

                st.markdown(
                    f"**{profile.get('years_of_experience', '?')} yrs** · "
                    f"{'📍 ' + str(profile.get('location', '?')) if not controls['anonymized'] else '📍 hidden'}"
                )
                if not controls["anonymized"]:
                    st.caption(profile.get("headline", "")[:120])
                rr = signals.get("recruiter_response_rate", 0) or 0
                notice = signals.get("notice_period_days", 90) or 90
                st.markdown(
                    f"Response rate **{rr:.0%}** · notice **{notice}d** · "
                    f"interviews **{(signals.get('interview_completion_rate', 0) or 0):.0%}**"
                )

            with col2:
                bars = [
                    ("Technical fit", ss.get("technical_fit", 0), ACCENT),
                    ("Career quality", ss.get("career_quality", 0), GREEN),
                    ("Availability", ss.get("availability_signal", 0), AMBER),
                    ("Seniority fit", ss.get("seniority_fit", 0), RED),
                    ("Semantic match", float(semantic_sim[i]), MUTED),
                ]
                st.markdown(
                    "".join(score_bar_html(l, v, c) for l, v, c in bars),
                    unsafe_allow_html=True,
                )

            st.markdown("**Why this candidate** — evidence from the profile:")
            render_evidence(ev)
            st.caption(f"*{generate_reasoning(cand, ev)}*")


# --------------------------------------------------------------- main layout

def main():
    try:
        artifacts = load_artifacts()
        artifacts_ready = True
    except Exception as e:
        artifacts = None
        artifacts_ready = False
        st.error(f"Artifacts not found ({e}). Run `python precompute.py` first.")

    n = len(artifacts["candidate_ids"]) if artifacts_ready else 0
    n_disq = len(artifacts["disqualified"]) if artifacts_ready else 0
    controls = render_sidebar(artifacts_ready, n, n_disq)

    st.title("RecruiterIQ")
    st.caption(
        "Explainable AI candidate ranking · 100K profiles re-ranked live · Redrob AI Challenge"
    )

    if not artifacts_ready:
        st.stop()

    top_idx, scores, semantic_sim, subscore_matrix = run_ranking(
        artifacts, controls["weights"], controls["diversity"]
    )
    weights_key = tuple(sorted(controls["weights"].items()))
    jd_key = st.session_state.get("jd_label", "__default__")
    stability = cached_stability(weights_key, jd_key, TOP_K)

    tab_shortlist, tab_method = st.tabs(["🏆 Shortlist", "📖 Methodology"])

    with tab_shortlist:
        render_shortlist(artifacts, top_idx, scores, semantic_sim, controls, stability)

    with tab_method:
        render_methodology(controls["weights"])


def render_methodology(weights: dict):
    st.subheader("Composite score")
    formula = " + ".join(
        f"{weights[name]:.2f} × {SUBSCORE_LABELS[name].replace(' ', '_')}" for name in weights
    )
    st.code(f"S = penalty × ({formula})")
    st.markdown(
        """
        | Signal | What it measures |
        |---|---|
        | Technical fit | JD must-haves (embeddings/retrieval, vector DBs, Python, eval) + nice-to-haves, weighted by declared proficiency and assessment scores |
        | Career quality | Product-company history, median tenure, upward title progression |
        | Availability | Open-to-work, recency, response rate, interview completion, notice period |
        | Seniority fit | Ideal 6–9 yrs experience band, education tier bonus |
        | Semantic match | MiniLM embedding cosine vs the JD — catches strong profiles that use plain language |
        """
    )
    st.subheader("Integrity rules")
    from config import (
        CONSULTING_PENALTY,
        CV_SPEECH_ROBOTICS_PENALTY,
        GHOST_COMPLETENESS_THRESHOLD,
        HONEYPOT_YEAR_BUFFER,
        NO_CODE_PENALTY,
    )

    st.markdown(
        f"""
        - **Honeypot** (disqualified): claimed experience exceeds the career
          timeline by more than **{HONEYPOT_YEAR_BUFFER} years**.
        - **Ghost** (disqualified): profile completeness below
          **{GHOST_COMPLETENESS_THRESHOLD}** with no verified email or phone.
        - **Pure research** (disqualified): all roles are research titles with
          zero production/deployment evidence.
        - **All-consulting career**: composite ×**{CONSULTING_PENALTY}**.
        - **No code shipped in 18 months**: composite ×**{NO_CODE_PENALTY}**.
        - **CV/speech/robotics-only ML profile**: composite ×**{CV_SPEECH_ROBOTICS_PENALTY}**.
        """
    )


if __name__ == "__main__":
    main()
