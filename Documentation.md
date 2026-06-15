# 📚 SignalHire — Complete Documentation

> **Version 1.0** · Last updated: June 2026  
> AI-Powered Candidate Ranking System for the Redrob AI Challenge

---

## Table of Contents

- [Overview](#overview)
- [Who Is This For?](#who-is-this-for)
- [Architecture](#architecture)
  - [Two-Phase Pipeline](#two-phase-pipeline)
  - [Dependency Graph](#dependency-graph)
  - [Data Flow](#data-flow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Docker](#docker)
- [Module Reference](#module-reference)
  - [config.py — Configuration Hub](#configpy--configuration-hub)
  - [textmatch.py — Keyword Matching Engine](#textmatchpy--keyword-matching-engine)
  - [disqualify.py — Adversarial Profile Detection](#disqualifypy--adversarial-profile-detection)
  - [signals.py — Sub-Score Computation](#signalspy--sub-score-computation)
  - [evidence.py — Provenance & Reasoning](#evidencepy--provenance--reasoning)
  - [engine.py — Vectorized Ranking Engine](#enginepy--vectorized-ranking-engine)
  - [precompute.py — Phase A Pipeline](#precomputepy--phase-a-pipeline)
  - [rank.py — Phase B Pipeline](#rankpy--phase-b-pipeline)
  - [app.py — Interactive Dashboard](#apppy--interactive-dashboard)
- [Scoring Methodology](#scoring-methodology)
  - [Composite Formula](#composite-formula)
  - [Technical Fit (0.35)](#technical-fit-035)
  - [Career Quality (0.25)](#career-quality-025)
  - [Availability Signal (0.20)](#availability-signal-020)
  - [Seniority Fit (0.12)](#seniority-fit-012)
  - [Semantic Similarity (0.08)](#semantic-similarity-008)
  - [Penalty Multipliers](#penalty-multipliers)
- [Data Schemas](#data-schemas)
  - [Input: candidates.jsonl](#input-candidatesjsonl)
  - [Internal: Artifacts](#internal-artifacts)
  - [Output: submission.csv](#output-submissioncsv)
- [Dashboard Guide](#dashboard-guide)
- [Deployment](#deployment)
  - [Local Development](#local-development)
  - [Docker Container](#docker-container)
  - [Documentation Site](#documentation-site)
- [Sandbox Constraints](#sandbox-constraints)
- [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
- [FAQ](#faq)
- [License](#license)
- [Initial Documentation Archive](#initial-documentation-archive)

---

## Overview

**SignalHire** is an AI-powered candidate ranking system that processes **100,000 candidate profiles** and surfaces the **top 100 best matches** for a Senior AI Engineer role. It was built for the **Redrob AI Challenge** at the India Runs Hackathon.

### The Problem

Given a 487 MB JSONL file containing 100K candidate profiles — including keyword stuffers, ~80 honeypot/adversarial profiles, behavioral twins, and strong engineers who describe themselves in plain language — identify and rank the top 100 candidates for a Senior AI Engineer position.

### The Solution

A two-phase offline hybrid scoring pipeline:

1. **Phase A (Precompute)** — GPU-accelerated embedding generation + multi-signal sub-score computation
2. **Phase B (Ranking)** — CPU-only composite scoring via vectorized matrix operations, evidence-based reasoning, and validated CSV output

The system uses **5 orthogonal scoring signals** (technical fit, career quality, availability, seniority fit, semantic similarity) combined with an **adversarial detection layer** that catches honeypots, ghosts, and pure-research profiles before they can pollute the shortlist.

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Offline precompute** | Separates expensive GPU work from CPU-only ranking sandbox |
| **Multi-signal scoring** | Defeats keyword stuffers — no single signal can game the system |
| **Semantic embedding backup** | Catches strong engineers whose plain language misses keyword checks |
| **Evidence-cited reasoning** | Every score claim traces to a specific skill or sentence |
| **Vectorized re-ranking** | Enables live 50ms re-ranking of all 100K candidates in the dashboard |

---

## Who Is This For?

| Audience | What You'll Find Here |
|---|---|
| **Hackathon judges** | Architecture overview, scoring methodology, sandbox compliance |
| **Contributors** | Module reference, function signatures, dependency graph, [good first issues](GOOD_FIRST_ISSUES.md) |
| **Recruiters / HR-tech engineers** | Scoring methodology, fairness audit capabilities, dashboard guide |
| **Researchers** | Embedding strategy, adversarial detection heuristics, MMR diversity |
| **DevOps / deployers** | Docker setup, environment config, performance benchmarks |

---

## Architecture

### Two-Phase Pipeline

```
                    candidates.jsonl (100K, 487 MB)
                            │
                            ▼
          ┌──────────────────────────────────────┐
          │  PHASE A: Precompute (GPU, ~4 min)   │
          │                                      │
          │  Stream JSONL ─► Disqualify fakes    │
          │         └──► Embed (MiniLM, 384-dim) │
          │         └──► Compute 4 sub-scores    │
          │         └──► Serialize artifacts     │
          └──────────────┬───────────────────────┘
                         │  embeddings.npy + subscores.pkl (~155 MB)
                         ▼
          ┌──────────────────────────────────────┐
          │  PHASE B: Ranking (CPU, <10 s)       │
          │                                      │
          │  Load artifacts ─► Cosine similarity │
          │         └──► Weighted composite      │
          │         └──► argpartition → top 100  │
          │         └──► Evidence-based reasoning│
          │         └──► Validate & write CSV    │
          └──────────────┬───────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────────┐
          │  DASHBOARD (CPU, ~50 ms re-rank)     │
          │                                      │
          │  Load same artifacts ─► Weight UI    │
          │         └──► Live matrix re-score    │
          │         └──► 6 interactive tabs      │
          │         └──► Export CSV / outreach   │
          └──────────────────────────────────────┘
```

### Dependency Graph

The codebase is organized into clean dependency layers:

```
Layer 0 (leaf modules — no project imports):
    textmatch.py ─── re, functools
    config.py    ─── os, re, pathlib

Layer 1:
    disqualify.py ──► config, textmatch

Layer 2:
    signals.py ────► config, disqualify, textmatch

Layer 3:
    evidence.py ───► config, signals, textmatch
    engine.py ─────► config

Layer 4:
    precompute.py ─► config, disqualify, signals
                     + sentence_transformers (external)

Layer 5:
    rank.py ───────► config, engine, evidence

Layer 6:
    app.py ────────► config, engine, evidence, rank
                     + streamlit, plotly (external)
```

### Data Flow

```
candidates.jsonl ──►  precompute.py  ──►  artifacts/
                          │                    │
                          │                    ├── embeddings.npy     (99965 × 384, float32)
                          │                    ├── candidate_ids.npy  (99965 strings)
                          │                    ├── jd_embedding.npy   (384, float32)
                          │                    ├── subscores.pkl      (99965 dicts)
                          │                    ├── disqualified.json  (~35 records)
                          │                    └── demographics.csv   (pool stats)
                          │
                          └── rank.py  ──►  output/submission.csv
                              app.py   ──►  Streamlit dashboard (http://localhost:8501)
```

---

## Getting Started

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python** ≥ 3.10 | Required |
| **NVIDIA GPU + CUDA** | Optional — speeds precompute ~10×. Set `EMBEDDING_DEVICE = "cpu"` in `config.py` without one |
| **Disk** | ~500 MB for data + ~160 MB for generated artifacts |
| **RAM** | 16 GB recommended |

### Installation

```bash
# Clone and setup
git clone https://github.com/your-org/SignalHire.git
cd SignalHire

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | Array operations, artifact storage |
| `sentence-transformers` | ≥ 2.2 | Profile & JD embedding |
| `python-docx` | ≥ 1.0 | Job description parsing (.docx) |
| `scikit-learn` | ≥ 1.3 | Cosine similarity utilities |
| `streamlit` | ≥ 1.28 | Interactive dashboard |
| `pandas` | ≥ 2.0 | Data manipulation, CSV generation |
| `matplotlib` | ≥ 3.7 | Visualization (fairness charts) |
| `plotly` | ≥ 5.18 | Interactive radar/histogram charts |

### Running the Pipeline

```bash
# Step 1: Precompute embeddings + sub-scores (GPU recommended, ~4 min)
python precompute.py

# Step 2: Generate ranked submission (CPU, ~5 s)
python rank.py

# Step 3: Launch interactive dashboard (optional)
streamlit run app.py
```

### Docker

```bash
# Build (pre-downloads embedding model)
docker build -t signalhire .

# Rank
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  signalhire python rank.py

# Dashboard
docker run --rm -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  signalhire streamlit run app.py
```

> Add `--gpus all` if NVIDIA Container Toolkit is installed.

---

## Module Reference

### `config.py` — Configuration Hub

**Purpose:** Single source of truth for all paths, weights, keyword lists, penalties, and thresholds. Every other module imports from here — no magic numbers elsewhere in the codebase.

#### Path Constants

| Constant | Value | Description |
|---|---|---|
| `ROOT` | Project root | Base directory (resolved from `config.py` location) |
| `DATA_DIR` | `ROOT / "data"` | Raw input data |
| `ARTIFACTS_DIR` | `ROOT / "artifacts"` | Precomputed output |
| `OUTPUT_DIR` | `ROOT / "output"` | Final CSV output |
| `CANDIDATES_PATH` | `DATA_DIR / "candidates.jsonl"` | 100K profiles (~487 MB) |
| `JD_PATH` | `DATA_DIR / "job_description.txt"` | Job description (text) |
| `JD_DOCX_PATH` | `DATA_DIR / "job_description.docx"` | Job description (docx) |
| `SAMPLE_PATH` | `DATA_DIR / "sample_candidates.json"` | Sample data for demo mode |
| `VALIDATE_SCRIPT` | `DATA_DIR / "validate_submission.py"` | Challenge validator script |

#### Scoring Weights

```python
WEIGHTS = {
    "technical_fit":       0.35,
    "career_quality":      0.25,
    "availability_signal": 0.20,
    "seniority_fit":       0.12,
    "semantic_similarity": 0.08,
}
# Sum is asserted to equal 1.0
```

#### Embedding Configuration

| Constant | Value | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `"sentence-transformers/all-MiniLM-L6-v2"` | 384-dim sentence embeddings |
| `EMBEDDING_DIM` | `384` | Vector dimensionality |
| `EMBEDDING_DEVICE` | `"cuda"` | Compute device (`"cuda"` or `"cpu"`) |
| `EMBEDDING_BATCH_SIZE` | `2048` | Batch size for encoding |
| `PROFILE_BLOB_MAX_CHARS` | `1024` | Max chars for embedding input |

#### JD Matching Criteria

**Must-haves** (weighted, sum = 0.75):

| Criterion | Weight | Keywords |
|---|---|---|
| Embeddings & Retrieval | 0.25 | embedding, retrieval, vector search, information retrieval, dense retrieval, rag, ... |
| Vector Databases | 0.20 | vector database, faiss, pinecone, weaviate, milvus, qdrant, chroma, ... |
| Python | 0.15 | python, pytorch, tensorflow, jax, numpy, ... |
| Eval Frameworks | 0.15 | evaluation, ndcg, mrr, map, recall, precision, ... |

**Nice-to-haves** (weighted, sum = 0.25):

| Criterion | Weight | Keywords |
|---|---|---|
| LLM Fine-tuning | 0.10 | fine-tuning, lora, qlora, rlhf, peft, instruction tuning, ... |
| Learning to Rank | 0.10 | learning to rank, lambdamart, listwise, pairwise, ... |
| HR-Tech | 0.05 | recruiter, talent, hiring, ats, applicant, sourcing, ... |

#### Penalty Constants

| Constant | Value | Description |
|---|---|---|
| `CONSULTING_PENALTY` | `0.15` | Multiplier for all-consulting careers |
| `NO_CODE_PENALTY` | `0.80` | Multiplier for 18+ months of inactivity |
| `CV_SPEECH_ROBOTICS_PENALTY` | `0.85` | Multiplier for off-domain specialization |
| `HONEYPOT_YEAR_BUFFER` | `5` | Allowed YoE vs career timeline discrepancy |
| `GHOST_COMPLETENESS_THRESHOLD` | `5` | Profile completeness % below which → ghost |

#### Other Notable Constants

| Constant | Value | Purpose |
|---|---|---|
| `TOP_K` | `100` | Number of candidates in final shortlist |
| `SEED` | `42` | Deterministic reproducibility |
| `CURRENT_YEAR` | `2026` | Reference year for date calculations |
| `CONSULTING_FIRMS` | 12 firms | TCS, Infosys, Wipro, Cognizant, HCL, Accenture, ... |
| `ML_AI_TITLE_KEYWORDS` | 25 keywords | machine learning, deep learning, data scientist, ... |
| `TIER_1_INSTITUTIONS` | 9 entries | IIT, IISc, NIT, BITS, ... |
| `TIER_2_INSTITUTIONS` | 13 entries | DTU, VIT, SRM, ... |
| `PRODUCTION_SIGNALS` | 10 keywords | production, deployed, shipped, serving, ... |
| `RETRIEVAL_SIGNALS` | 13 keywords | embedding, vector, rag, faiss, ... |

---

### `textmatch.py` — Keyword Matching Engine

**Purpose:** Word-boundary–aware keyword matching. Prevents false positives like `"rag"` matching inside `"storage"` or `"map"` inside `"bitmap"`.

**Design:** Keywords ≥4 characters use stem matching (`\b{kw}\w*`), allowing `"eval"` → `"evaluation"`. Keywords <4 characters require exact word match (`\b{kw}\b`).

#### Functions

##### `keyword_pattern(kw: str) → re.Pattern`

Returns a compiled regex for a keyword. **Cached** via `functools.lru_cache(maxsize=1024)`.

- `kw` ≥ 4 chars → `re.compile(r"\b{kw}\w*", re.IGNORECASE)` (stem match)
- `kw` < 4 chars → `re.compile(r"\b{kw}\b", re.IGNORECASE)` (exact word)

##### `matches_keyword(text_lower: str, kw: str) → bool`

Tests if a single keyword matches within the text.

##### `matches_any(text: str, keywords: List[str]) → bool`

Returns `True` if any keyword in the list matches. Short-circuits on first match.

##### `matched_keywords(text: str, keywords: List[str]) → List[str]`

Returns all keywords that match within the text.

---

### `disqualify.py` — Adversarial Profile Detection

**Purpose:** Identifies and removes adversarial profiles (honeypots, ghosts, pure-research) from the ranking pool, and applies soft score penalties to consulting-only, stale-skills, or off-domain profiles.

#### Hard Disqualification Functions

##### `is_honeypot(candidate: dict) → Tuple[bool, str]`

Runs three checks in order:

1. **YoE Inflation** — If claimed `years_of_experience` exceeds `(CURRENT_YEAR - earliest_role_start_year) + HONEYPOT_YEAR_BUFFER`, returns `(True, "HONEYPOT: ...")`
2. **Ghost Profile** — If `profile_completeness_score < GHOST_COMPLETENESS_THRESHOLD` and no verified contact info, returns `(True, "GHOST_PROFILE: ...")`
3. **Pure Research** — If all roles are "researcher" with zero production/deployment signals, returns `(True, "PURE_RESEARCH: ...")`

Returns `(False, "")` if the candidate passes all checks.

#### Helper Functions

| Function | Signature | Purpose |
|---|---|---|
| `parse_year` | `(date_str: Optional[str]) → Optional[int]` | Extracts 4-digit year; handles `"present"` / `"now"` → `CURRENT_YEAR` |
| `sort_career_chronologically` | `(career: list) → list` | Sorts roles oldest-first by `start_date` |
| `_latest_role` | `(career: list) → Optional[dict]` | Returns current role or chronologically last role |
| `_earliest_role_start_year` | `(candidate: dict) → Optional[int]` | Min start year across all career entries |
| `_all_roles_at_consulting` | `(candidate: dict) → bool` | `True` if every role's company name matches a known consulting firm |
| `_is_pure_research` | `(candidate: dict) → bool` | All roles titled "researcher" + no production signal keywords |
| `_is_no_code_18mo` | `(candidate: dict) → bool` | Last activity >548 days ago + no current role |
| `_is_cv_speech_robotics_only` | `(candidate: dict) → bool` | Has CV/speech/robotics skills but zero retrieval signals |

#### Soft Penalty Function

##### `should_hard_penalize(candidate: dict) → Tuple[float, str]`

Returns a cumulative penalty multiplier and a semicolon-separated reasons string.

| Condition | Multiplier | Trigger |
|---|---|---|
| All-consulting career | × 0.15 | Every role at TCS, Infosys, Wipro, etc. |
| No code in 18 months | × 0.80 | Last active >548 days ago, no current role |
| CV/speech/robotics only | × 0.85 | Domain-specialist, no retrieval/search signals |

Multipliers stack multiplicatively: a consulting-only candidate with stale skills gets `0.15 × 0.80 = 0.12`.

---

### `signals.py` — Sub-Score Computation

**Purpose:** Computes the four primary sub-scores that feed into the composite ranking formula. Each function returns a float in `[0, 1]`.

#### `compute_technical_fit(candidate: dict) → float`

Scores how well a candidate's skills and experience match the job description requirements.

**Components:**
1. **Must-have match** (75% weight) — For each JD must-have criterion, finds the best match across declared skills (weighted by proficiency) and career description text
2. **Nice-to-have match** (25% weight) — Same approach for bonus criteria
3. **Production bonus** — +0.03 per production keyword (capped at 0.10) found in career descriptions
4. **Retrieval bonus** — +0.03 per retrieval keyword (capped at 0.10) found in career descriptions

**Internal helpers:**
- `_build_skill_dict(candidate)` → Dict mapping skill name → weight (merging proficiency level + assessment scores)
- `skill_matches_keyword(kw, skill_name)` → Boolean substring match with short-string guards
- `_keyword_match_score(keywords, skill_dict, text_blob)` → Best match score for a keyword list
- `_scan_signals_in_text(text, signals)` → Keyword hit count × 0.03, capped at 0.10

#### `compute_career_quality(candidate: dict) → float`

Assesses career trajectory and company quality.

**Scoring rules:**
- Non-consulting role presence → +0.30
- ML/AI title at company with ≥50 employees → +0.15
- Median tenure ≥36 months → +0.25; ≥24 months → +0.20; ≥18 months → +0.05
- Upward title progression (last role seniority > first) → +0.20

**Internal helpers:**
- `_is_ml_ai_title(title)` → Checks against 25 ML/AI keywords
- `_parse_company_size(size_str)` → Parses `"51-200"` → `200`, `"10001+"` → `10002`
- `_title_seniority_level(title)` → Maps to 0–6 scale via regex (intern → C-level)
- `_median_tenure_months(candidate)` → Median `duration_months` across career
- `_has_upward_title_progression(candidate)` → Last role seniority > first role seniority

#### `compute_availability_signal(candidate: dict) → float`

Measures behavioral signals from the Redrob platform indicating the candidate is actively looking and responsive.

**Components:**
- `open_to_work_flag` → +0.25
- Recency (last active): ≤30 days +0.20, ≤90 days +0.12, ≤180 days +0.06
- `recruiter_response_rate` → × 0.15
- `interview_completion_rate` → × 0.15
- Active applications (>0) → +0.05
- Notice period: ≤30 days +0.05, ≤60 days +0.035

#### `compute_seniority_fit(candidate: dict) → float`

Evaluates experience-level alignment with the target role (Senior AI Engineer, ideal 6–9 YoE).

**YoE bands:**

| YoE Range | Score |
|---|---|
| 6–9 years (ideal) | 1.0 |
| 4–5 or 10–12 years | 0.7 |
| 3 or 13–15 years | 0.4 |
| Everything else | 0.1 |

**Education bonus:** Tier-1 institution → +0.05, Tier-2 → +0.02.

---

### `evidence.py` — Provenance & Reasoning

**Purpose:** Traces every JD requirement match back to the specific skill or career sentence that triggered it. Generates honest, evidence-cited reasoning strings for the output CSV.

#### Functions

##### `collect_evidence(candidate: dict) → dict`

Returns a structured evidence dictionary:

```python
{
    "matched": [
        {
            "criterion": "embeddings & retrieval",
            "match_type": "skill",        # or "text"
            "skill_name": "vector search",
            "weight": 0.75,
            "snippet": None               # or "...deployed embedding pipeline at scale..."
        },
        ...
    ],
    "missing_must_haves": ["eval frameworks"],
    "production": ["deployed", "serving"],
    "retrieval": ["embedding", "vector", "faiss"]
}
```

##### `generate_reasoning(candidate: dict, evidence: dict = None, max_len: int = 300) → str`

Produces a one-liner reasoning string (120–300 chars) for the CSV. Format:

```
"6.7yr Lead AI Engineer at Razorpay; strong match on embeddings, vector search, python;
production evidence (serving, ndcg); actively looking, 73% response rate, 30d notice."
```

**Rules:**
- Only claims what the evidence dict actually contains
- Names specific skills, companies, and signals
- Calls out missing must-haves as gaps
- Capped at `max_len` characters

#### Internal Helpers

| Function | Purpose |
|---|---|
| `_snippet_around(text, keyword)` | Extracts ±60 char context window around a keyword match |
| `_career_text(candidate)` | Joins all `career_history[].description` fields |
| `_profile_blob(candidate)` | Concatenates headline + summary + current_title + career text |
| `_match_criterion(keywords, skill_dict, blob)` | Finds best match (prefers skill-based over text-based) for one criterion |

---

### `engine.py` — Vectorized Ranking Engine

**Purpose:** Performs all ranking math as vectorized NumPy operations for instant re-ranking. Every operation works on precomputed matrices — no per-candidate loops.

#### Constants

- `SUBSCORE_ORDER = ["technical_fit", "career_quality", "availability_signal", "seniority_fit"]`

#### Functions

##### `build_matrices(candidate_ids, subscores_dict) → Tuple[np.ndarray, np.ndarray]`

Packs the per-candidate sub-score dictionaries into dense float32 arrays:
- `subscore_matrix`: shape `(N, 4)`, columns in `SUBSCORE_ORDER`
- `penalties`: shape `(N,)`, the `penalty_multiplier` values

##### `compute_scores(subscore_matrix, penalties, semantic_sim, weights) → np.ndarray`

The core scoring operation:

```python
scores = penalties * (subscore_matrix @ weight_vector + w_semantic * semantic_sim)
```

Returns shape `(N,)` composite scores.

##### `top_k_indices(scores: np.ndarray, k: int) → np.ndarray`

Selects top-k candidates using `np.argpartition` (O(N) average). Includes deterministic tie-breaking by candidate index.

##### `scale_score(raw: float, scale: float = SCORE_SCALE) → float`

Clamps raw score to `[0, 1]`, then multiplies by `scale` (default 10).

##### `format_score(raw: float, scale: float = SCORE_SCALE, decimals: int = SCORE_DECIMALS) → str`

Formats a score as `"x.xx"`.

##### `mmr_rerank(candidate_idx, scores, embeddings, lambda_relevance, k) → List[int]`

**Maximal Marginal Relevance** re-ranking. Balances relevance vs diversity:

```
MMR(d) = λ × score(d) − (1−λ) × max_similarity(d, already_selected)
```

At `λ = 1.0`, behaves like pure score ranking. Lower values surface more diverse candidates.

##### `stability_analysis(subscore_matrix, penalties, semantic_sim, weights, k, n_trials=200, jitter=0.20, seed=42) → Dict[int, float]`

Runs 200 Monte Carlo trials with ±20% random weight perturbation. Returns per-candidate frequency of appearing in top-k across trials. Candidates with >95% frequency are "stable" regardless of exact weight choice.

---

### `precompute.py` — Phase A Pipeline

**Purpose:** The expensive one-time computation phase. Streams the full 487 MB JSONL, removes adversarial profiles, embeds all valid profiles using `all-MiniLM-L6-v2`, computes all sub-scores, and serializes everything to disk.

#### Functions

##### `extract_jd_text() → str`

Reads the job description from `job_description.txt` (preferred) or `job_description.docx` (fallback via `python-docx`).

##### `build_profile_blob(candidate: dict) → str`

Constructs the text that gets embedded. Concatenates:

1. Headline
2. Summary
3. Current title
4. Career history descriptions
5. Top skill names (sorted by proficiency)

Truncated at `PROFILE_BLOB_MAX_CHARS` (1024) characters.

##### `compute_subscores(candidate: dict) → dict`

Calls all four signal functions and the penalty calculator:

```python
{
    "technical_fit":       compute_technical_fit(candidate),
    "career_quality":      compute_career_quality(candidate),
    "availability_signal": compute_availability_signal(candidate),
    "seniority_fit":       compute_seniority_fit(candidate),
    "penalty_multiplier":  multiplier,  # from should_hard_penalize()
    "top_skills":          [...],
    "strongest_company":   "...",
    "flags":               "..."
}
```

##### `main() → None`

Full pipeline orchestration:

1. Create `ARTIFACTS_DIR` if missing
2. Load `SentenceTransformer` model on configured device
3. Extract and embed JD text → save `jd_embedding.npy`
4. Stream `candidates.jsonl` in 1K-line chunks:
   - Parse JSON, skip malformed lines
   - Run `is_honeypot()` — if flagged, log to `disqualified` list and skip
   - Build profile blob → collect for batch embedding
   - Compute sub-scores → store in `subscores` dict
5. Batch-encode all profile blobs (batch size 2048) → normalize
6. Save: `embeddings.npy`, `candidate_ids.npy`, `subscores.pkl`, `disqualified.json`
7. Log summary statistics

**Runtime:** ~4 minutes on NVIDIA RTX 3050 (6 GB), ~40 minutes CPU-only.

---

### `rank.py` — Phase B Pipeline

**Purpose:** Loads precomputed artifacts, calculates composite scores, selects the top 100, generates evidence-based reasoning for each, and writes a validated CSV.

#### Functions

##### `build_offset_index() → dict`

Builds a `candidate_id → byte_offset` index by scanning the JSONL file once (~2 seconds). This allows O(1) random access to any candidate's full record without loading the entire 487 MB file.

##### `load_candidates_by_ids(target_ids, offset_index=None) → dict`

Loads full candidate records for a set of IDs:
- **With offset index:** Direct `seek()` per ID — ~100 seeks for top 100
- **Without offset index:** Single sequential pass through the JSONL

##### `main() → None`

1. Load all artifacts from `ARTIFACTS_DIR`
2. Build matrices via `engine.build_matrices()`
3. Compute cosine similarity: `embeddings @ jd_embedding`
4. Compute composite scores via `engine.compute_scores()`
5. Select top-100 via `engine.top_k_indices()`
6. Build byte-offset index → load full records for top 100 only
7. Generate evidence + reasoning per candidate
8. Enforce monotonic score ordering (scores must be non-increasing by rank)
9. Write `output/submission.csv`
10. Run `validate_submission.py` as a subprocess to verify compliance

---

### `app.py` — Interactive Dashboard

**Purpose:** A Streamlit-powered interactive ranking workbench with 6 tabs. Every control re-ranks all 100K candidates in ~50 ms because scoring is a single `float32` matrix multiply over precomputed artifacts.

#### Data Layer (cached)

All data-loading functions use Streamlit's `@st.cache_resource` or `@st.cache_data` decorators for zero-reload performance:

| Function | Decorator | Purpose |
|---|---|---|
| `load_artifacts()` | `@st.cache_resource` | Loads all `.npy`, `.pkl`, `.json` artifacts into memory |
| `get_matrices()` | `@st.cache_resource` | Builds dense sub-score matrix + penalty vector |
| `get_model()` | `@st.cache_resource` | Lazy-loads `SentenceTransformer` (only for custom JD) |
| `embed_text(text)` | `@st.cache_data` | Embeds a single text string |
| `get_offset_index()` | `@st.cache_resource` | Builds byte-offset index for candidate lookup |
| `cached_candidates(ids)` | `@st.cache_data` | Loads candidate records by ID |
| `pool_demographics()` | `@st.cache_data` | Extracts country, YoE, education tier for fairness audit |
| `cached_stability(weights, jd, k)` | `@st.cache_data` | Monte Carlo stability analysis |

#### UI Helpers

| Function | Purpose |
|---|---|
| `score_bar_html(label, value, color)` | Renders a horizontal progress bar |
| `chip(text, color, title)` | Renders a colored badge/chip |
| `stability_badge(freq)` | Returns "stable" / "moderate" / "fragile" badge |
| `normalized_weights()` | Reads slider values, normalizes to sum = 1.0 |

#### Tab Renderers

| Function | Tab | Features |
|---|---|---|
| `render_shortlist()` | 🏆 Shortlist | Paginated candidate cards with scores, evidence, stability badges |
| `render_compare()` | ⚖️ Compare | Radar chart side-by-side of 2–4 candidates |
| `render_insights()` | 📊 Insights | Score histogram + fairness audit (education, country, YoE) |
| `render_integrity()` | 🛡️ Integrity | Disqualification log with examples |
| `render_export()` | 📤 Export | CSV download + outreach pack + config JSON |
| `render_methodology()` | 📖 Methodology | Scoring formula + integrity rules |

#### Sidebar Controls

| Control | Function |
|---|---|
| Custom JD | Paste any job description → embedded on the fly |
| Weight sliders | 5 draggable sliders → instant re-rank |
| Diversity (MMR) | 0 = pure score, higher = more diverse shortlist |
| Blind screening | Hides names, companies, institutions |

---

## Scoring Methodology

### Composite Formula

```
S = penalty_multiplier × (
    0.35 × technical_fit
  + 0.25 × career_quality
  + 0.20 × availability_signal
  + 0.12 × seniority_fit
  + 0.08 × semantic_similarity
)
```

All sub-scores are normalized to `[0, 1]`. The `penalty_multiplier` is `1.0` for clean profiles and <1.0 for penalized ones (multipliers stack multiplicatively).

### Technical Fit (0.35)

Measures alignment between the candidate's skills/experience and the JD requirements.

**Algorithm:**
1. Build a skill dictionary from declared skills (weighted by proficiency: expert=1.0, advanced=0.75, intermediate=0.4, beginner=0.1) merged with assessment scores
2. For each JD must-have criterion, find the best match across skills and career text
3. Weight must-have matches by criterion importance (0.25, 0.20, 0.15, 0.15)
4. Repeat for nice-to-have criteria (0.10, 0.10, 0.05)
5. Add production signal bonus (+0.03 per keyword, cap 0.10)
6. Add retrieval signal bonus (+0.03 per keyword, cap 0.10)

### Career Quality (0.25)

Assesses career trajectory and employer quality:
- Has non-consulting roles → +0.30
- ML/AI title at ≥50-person company → +0.15
- Median tenure: ≥36mo → +0.25, ≥24mo → +0.20, ≥18mo → +0.05
- Upward title progression → +0.20

### Availability Signal (0.20)

Behavioral signals from the Redrob platform:
- Open to work flag → +0.25
- Recency: ≤30d → +0.20, ≤90d → +0.12, ≤180d → +0.06
- Response rate → × 0.15
- Interview completion rate → × 0.15
- Active applications → +0.05
- Notice period: ≤30d → +0.05, ≤60d → +0.035

### Seniority Fit (0.12)

Experience-level alignment (ideal range: 6–9 years for Senior AI Engineer):

| YoE | Score |
|---|---|
| 6–9 | 1.0 |
| 4–5 or 10–12 | 0.7 |
| 3 or 13–15 | 0.4 |
| <3 or >15 | 0.1 |

Plus education tier bonus (Tier-1: +0.05, Tier-2: +0.02).

### Semantic Similarity (0.08)

Cosine similarity between the candidate's profile embedding (384-dim, `all-MiniLM-L6-v2`) and the JD embedding. This is the "safety net" that catches strong engineers whose plain-language descriptions miss keyword checks.

### Penalty Multipliers

| Condition | Multiplier | Rationale |
|---|---|---|
| All-consulting career | × 0.15 | TCS/Infosys/Wipro careers rarely involve the required deep technical work |
| No code activity in 18+ months | × 0.80 | Skills may be stale |
| CV/speech/robotics only, no retrieval | × 0.85 | Domain misalignment with the JD |

Penalties stack: consulting + stale = 0.15 × 0.80 = **0.12**.

---

## Data Schemas

### Input: `candidates.jsonl`

Each line is a JSON object representing one candidate:

```json
{
  "candidate_id": "CAND_0000001",
  "profile": {
    "name": "...",
    "headline": "Senior ML Engineer | NLP & Search",
    "summary": "...",
    "location": "Bengaluru, India",
    "country": "India",
    "years_of_experience": 7,
    "current_title": "Senior ML Engineer",
    "current_company": "Razorpay",
    "company_size": "1001-5000",
    "industry": "Financial Technology"
  },
  "career_history": [
    {
      "company": "Razorpay",
      "title": "Senior ML Engineer",
      "start_date": "2022-03",
      "end_date": "present",
      "duration_months": 28,
      "is_current": true,
      "industry": "Financial Technology",
      "company_size": "1001-5000",
      "description": "Built embedding-based retrieval pipeline for..."
    }
  ],
  "education": [
    {
      "institution": "IIT Bombay",
      "degree": "M.Tech",
      "field": "Computer Science"
    }
  ],
  "skills": [
    {
      "name": "Python",
      "proficiency": "expert",
      "endorsements": 42,
      "duration_months": 84
    }
  ],
  "certifications": [...],
  "languages": [...],
  "redrob_signals": {
    "profile_completeness_score": 92,
    "open_to_work_flag": true,
    "recruiter_response_rate": 0.73,
    "avg_response_time_hours": 4.2,
    "skill_assessment_scores": {"python": 95, "ml": 88},
    "notice_period_days": 30,
    "github_activity_score": 78,
    "last_active_date": "2026-05-28",
    "interview_completion_rate": 0.85,
    "active_applications_count": 3,
    "verified_email": true,
    "verified_phone": true
  }
}
```

### Internal: Artifacts

| File | Format | Shape/Size | Description |
|---|---|---|---|
| `embeddings.npy` | float32 | (99965, 384) ~153 MB | Unit-normalized profile embeddings |
| `candidate_ids.npy` | str array | (99965,) ~1.5 MB | Parallel array — index-aligned with embeddings |
| `jd_embedding.npy` | float32 | (384,) ~1.7 KB | JD text embedding |
| `subscores.pkl` | Python dict | 99965 entries ~7 MB | `candidate_id → {technical_fit, career_quality, availability_signal, seniority_fit, penalty_multiplier, top_skills, strongest_company, flags}` |
| `disqualified.json` | JSON list | ~35 entries ~4 KB | `[{candidate_id, disqualify_type, reason}, ...]` |
| `demographics.csv` | CSV | ~100K rows ~3 MB | Cached pool demographics for fairness audit |

### Output: `submission.csv`

```csv
candidate_id,rank,score,reasoning
CAND_0081846,1,0.870,"6.7yr Lead AI Engineer at Razorpay; strong match on embeddings, vector search, python, information retrieval; production evidence (serving, ndcg); actively looking, 73% response rate, 30d notice."
CAND_0055905,2,0.869,"8.1yr Senior ML Engineer at Flipkart; strong match on embeddings, vector search, python, information retrieval; production evidence (deployed, serving); actively looking, 87% response rate."
```

**Validation rules:**

| Rule | Enforced |
|---|---|
| Exactly 100 rows (ranks 1–100) | ✅ |
| Scores non-increasing by rank | ✅ (monotonic enforcement) |
| Tie-breaking by `candidate_id` ascending | ✅ |
| `candidate_id` format `CAND_XXXXXXX` | ✅ |
| Reasoning ≤ 300 chars, no newlines | ✅ |
| Score to 3 decimal places | ✅ |

---

## Dashboard Guide

### Launching

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Tabs Overview

| Tab | Purpose |
|---|---|
| 🏆 **Shortlist** | Paginated candidate cards with rank, scores, evidence, and stability badges |
| ⚖️ **Compare** | Radar chart side-by-side comparison of 2–4 selected candidates |
| 📊 **Insights** | Score landscape histogram + fairness audit (shortlist vs pool demographics) |
| 🛡️ **Integrity** | Full disqualification audit log with concrete detection examples |
| 📤 **Export** | Download submission CSV, personalized outreach pack (top 10), or ranking config JSON |
| 📖 **Methodology** | Interactive scoring formula display + integrity rule documentation |

### Sidebar Controls

| Control | Effect |
|---|---|
| **Custom JD** | Paste any job description → embedded on the fly and ranked against all 100K |
| **Weight sliders** | Drag technical_fit, career_quality, availability, seniority, semantic weights → live 50ms re-rank |
| **Reset weights** | Return all sliders to default values |
| **Diversity (MMR)** | λ=1.0 → pure score. Lower → penalizes similarity to already-selected profiles |
| **Blind screening** | Hides candidate names, company names, and institution names |

### Performance

- Re-ranking all 100K candidates: **~50 ms** (single float32 matrix multiply)
- Candidate detail loading: **~100 seeks** via byte-offset index
- Stability analysis: **~2s** (200 Monte Carlo trials)

---

## Deployment

### Local Development

```bash
# Full pipeline
python precompute.py     # Phase A: ~4 min (GPU) / ~40 min (CPU)
python rank.py           # Phase B: ~5 seconds
streamlit run app.py     # Dashboard: http://localhost:8501
```

### Docker Container

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download model (no network needed at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY . .
CMD ["python", "rank.py"]
```

### Documentation Site

The documentation site is built with [docmd](https://docmd.io) — a zero-config documentation engine that auto-discovers Markdown files in the repository.

```bash
# Build static site
docmd build

# Development server with hot reload
docmd dev
```

**Deployment:** GitHub Pages via `.github/workflows/static.yml` deploys the `./site` directory on push to `main`.

---

## Sandbox Constraints

The Redrob AI Challenge sandbox imposes strict constraints. Here's how SignalHire meets each one:

| Constraint | How It's Met |
|---|---|
| **CPU-only ranking** | Phase B uses only NumPy — zero GPU dependency |
| **16 GB RAM limit** | Streaming JSONL ingestion, batched embedding, vectorized ops. Peak RAM: ~14 GB |
| **No network access** | Embedding model pre-downloaded at Docker build time |
| **< 5 minute ranking** | `np.argpartition` O(N) selection + byte-offset seek index. Total: ~5 seconds |
| **Deterministic results** | `SEED = 42` enforced throughout. Reproducible on re-run |

---

## Performance Benchmarks

Measured on **NVIDIA RTX 3050 (6 GB) + 12-core CPU**:

| Phase | Device | Time | Throughput |
|---|---|---|---|
| Precompute (100K → 99,965) | GPU (CUDA) | **~4.2 min** | ~400 candidates/s |
| Offset index build | CPU | **~1.5 s** | ~67K lines/s |
| Ranking + validation | CPU | **~5.0 s** | ~20K candidates/s |
| Dashboard re-rank | CPU | **~50 ms** | 2M candidates/s |

**Memory usage:**
- Embeddings in RAM: 99,965 × 384 × 4 bytes = ~153 MB
- Sub-scores in RAM: ~7 MB
- Total artifact footprint: ~162 MB

---

## Contributing

We welcome contributions! Check out [GOOD_FIRST_ISSUES.md](GOOD_FIRST_ISSUES.md) for 10 well-scoped starter tasks:

| # | Task | Difficulty | Est. Time |
|---|---|---|---|
| 1 | Remove dead `id_batch` variable in `precompute.py` | Trivial | 5 min |
| 2 | Add `.dockerignore` | Trivial | 10 min |
| 3 | Update stale performance numbers in README | Easy | 15 min |
| 4 | Fix unreachable TIER_2_INSTITUTIONS keywords | Easy | 20 min |
| 5 | Fix crash on whitespace-only `anonymized_name` | Easy | 15 min |
| 6 | Handle null role descriptions in `compute_technical_fit` | Easy | 30 min |
| 7 | Fix demographics cache invalidation | Medium | 1 hr |
| 8 | Add pytest regression suite | Medium | 2–3 hrs |
| 9 | Fix dashboard CSV tie-breaking per spec | Medium | 1 hr |

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b fix/issue-name`
3. Make changes, run `python rank.py` to verify output
4. Submit a pull request with a clear description

---

## FAQ

**Q: Why two phases instead of one?**  
A: The hackathon sandbox is CPU-only with a 5-minute time limit. Precomputing embeddings (which need GPU) separately lets us stay well under the sandbox time limit for ranking.

**Q: How does it handle keyword stuffers?**  
A: Multi-signal scoring. A candidate who stuffs "Python" 50 times gets high technical_fit but still needs career_quality (real job history), availability_signal (platform behavior), and seniority_fit (experience band). No single signal can game the composite score.

**Q: What's a honeypot profile?**  
A: An adversarial profile seeded by the challenge organizers with impossible claims — e.g., claiming 20 years of experience but only having 3 years of career history. The `is_honeypot()` function catches these by comparing claimed YoE against actual career timeline.

**Q: Why `all-MiniLM-L6-v2` over larger models?**  
A: Best accuracy-to-speed tradeoff for 100K profiles. 384 dimensions, ~60 MB model size, <200ms per batch of 2048. Larger models would blow the RAM/time budget.

**Q: Can I use this for a different job description?**  
A: Yes! The dashboard has a "Custom JD" input. Paste any job description — it gets embedded on the fly and all 100K candidates are re-ranked against it in ~50ms.

**Q: Why are consulting careers penalized so heavily (× 0.15)?**  
A: For this specific JD (Senior AI Engineer), the challenge data shows that all-consulting careers at service firms (TCS, Infosys, Wipro) rarely involve the depth of AI/ML work required. The penalty is specific to this JD — in a production system, it would be configurable.

**Q: How is the stability badge calculated?**  
A: 200 Monte Carlo trials with ±20% random weight perturbation. If a candidate appears in the top 100 in >95% of trials → "stable". Between 70–95% → "moderate". Below 70% → "fragile".

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute.

---

## Initial Documentation Archive

The original planning documents that guided the development of SignalHire are preserved in the [`Initial-Documentation/`](Initial-Documentation/) directory:

| Document | Description |
|---|---|
| [RecruiterIQ_PRD.md](Initial-Documentation/RecruiterIQ_PRD.md) | Product Requirements Document — problem statement, scoring formula, evaluation metrics |
| [TRD.md](Initial-Documentation/TRD.md) | Technical Requirements Document — functional and non-functional requirements |
| [BACKEND_SCHEMA.md](Initial-Documentation/BACKEND_SCHEMA.md) | Data Models & Interfaces — complete schema definitions for all I/O |
| [APP_FLOW.md](Initial-Documentation/APP_FLOW.md) | Application Flow — system diagrams, state machines, error handling |
| [IMPLEMENTATION_PLAN.md](Initial-Documentation/IMPLEMENTATION_PLAN.md) | Build Execution Roadmap — 9-phase plan with code snippets and risk register |
| [UIUX_DESIGN.md](Initial-Documentation/UIUX_DESIGN.md) | UI/UX Design — dashboard layout, color palette, component specs |

> These were the "blueprints" — the code is the "building". This Documentation.md describes the building as-built.

---

*Built for the Redrob AI Challenge — India Runs Hackathon*
