# RecruiterIQ — AI-Powered Candidate Ranking

Built for the **Redrob AI Challenge** (India Runs Hackathon). Given 100,000 candidate profiles, rank the **top 100** most suitable candidates for a **Senior AI Engineer** role under tight sandbox constraints (CPU-only, 16 GB RAM, no network, ranking done in <5 min).

## Pipeline

```
candidates.jsonl (100K)
       │
       ▼
┌──────────────────────────────┐
│  PHASE A: Precompute         │  offline, ~30-60 min
│  ─────────────────           │
│  · Stream & parse JSONL      │
│  · Disqualify bad actors     │  honeypot / ghost / pure research
│  · Embed profiles            │  all-MiniLM-L6-v2 → 384-dim
│  · Compute 4 sub-scores      │  technical, career, availability, seniority
│  · Serialize artifacts       │  .npy + .pkl → ~150 MB
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  PHASE B: Ranking            │  sandbox, <5 min
│  ────────────                │
│  · Load cached artifacts     │
│  · Cosine similarity (vec'd) │
│  · Composite score            │  S = penalty × Σ(wᵢ · scoreᵢ)
│  · argpartition → top 100    │
│  · Generate reasoning        │
│  · Validate & write CSV      │
└──────────────────────────────┘
       │
       ▼
  submission.csv (100 ranked)
```

## Composite Score

```
S = penalty_multiplier × (
    0.35 × technical_fit
  + 0.25 × career_quality
  + 0.20 × availability_signal
  + 0.12 × seniority_fit
  + 0.08 × semantic_similarity
)
```

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | Weights, paths, keyword lists, penalties |
| `disqualify.py` | Honeypot/ghost/research detection + soft penalties |
| `signals.py` | 4 sub-score functions (technical, career, availability, seniority) |
| `precompute.py` | Phase A: ingest → disqualify → embed → subscore → serialize |
| `rank.py` | Phase B: load → score → top-100 → reasoning → CSV |
| `app.py` | Streamlit demo (shortlist, explorer, methodology, audit tabs) |
| `Dockerfile` | Python 3.10-slim sandbox image |
| `requirements.txt` | numpy, sentence-transformers, streamlit, etc. |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Phase A — precompute (offline)
python precompute.py

# Phase B — rank & generate submission
python rank.py

# Demo dashboard
streamlit run app.py
```

## Disqualification & Penalties

- **Hard disqualify**: honeypot (impossible YoE), ghost (empty profile), pure research (no industry evidence)
- **Soft penalties**: consulting-only (×0.15), no-code >18mo (×0.80), CV/speech/robotics only (×0.85)

## Sandbox Constraints

| Constraint | How It's Met |
|---|---|
| CPU-only | `SentenceTransformer(device="cpu")` fallback |
| 16 GB RAM | Streaming JSONL, vectorized NumPy, batch embedding |
| No network | Model pre-downloaded in Docker build |
| <5 min ranking | `np.argpartition` O(N), ~150 MB artifacts |
