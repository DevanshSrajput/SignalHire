import csv
import json
import pickle
import subprocess
import sys
import time

import numpy as np

from config import (
    ARTIFACTS_DIR,
    CANDIDATES_PATH,
    OUTPUT_DIR,
    TOP_K,
    VALIDATE_SCRIPT,
    WEIGHTS,
)
def _top_skills(candidate: dict, max_n: int = 2) -> list:
    skills = candidate.get("skills", [])
    scored = []
    for sk in skills:
        name = sk.get("name", "")
        prof = sk.get("proficiency", "beginner")
        prof_w = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}.get(prof.lower(), 0)
        dur = sk.get("duration_months", 0) or 0
        scored.append((prof_w * 12 + dur, name))
    scored.sort(reverse=True)
    return [s[1] for s in scored[:max_n] if s[1]]


def _strongest_company(candidate: dict) -> str:
    career = candidate.get("career_history", [])
    for role in reversed(career):
        c = role.get("company", "")
        if c:
            return c
    return candidate.get("profile", {}).get("current_company", "unknown company")


def _generate_reasoning(candidate: dict, score: float, subscores: dict, penalty_mult: float) -> str:
    if not candidate:
        return ""

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "ML engineer").strip()
    yoe = profile.get("years_of_experience", 0)
    company = _strongest_company(candidate)
    top_skills = _top_skills(candidate, 2)
    skills_str = " + ".join(top_skills) if top_skills else "ML"

    open_work = signals.get("open_to_work_flag", False)
    avail_str = "actively looking" if open_work else "passive"

    rr = signals.get("recruiter_response_rate", 0) or 0
    rr_str = f", {rr:.0%} response rate" if rr > 0 else ""

    notice = signals.get("notice_period_days", 90) or 90
    notice_str = f", {notice}d notice" if notice < 90 else ""

    reasoning = (
        f"{yoe}yr {title} with production {skills_str} experience at {company}; "
        f"{avail_str}{rr_str}{notice_str}."
    )

    for ch in ["\n", "\r", "\"", "\t"]:
        reasoning = reasoning.replace(ch, " ")
    while "  " in reasoning:
        reasoning = reasoning.replace("  ", " ")

    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning.strip()


def _load_candidate_by_id(target_id: str) -> dict:
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            if cand.get("candidate_id") == target_id:
                return cand
    return None


def main():
    t0 = time.time()

    print("Loading artifacts ...")
    embeddings = np.load(str(ARTIFACTS_DIR / "embeddings.npy")).astype(np.float32)
    candidate_ids = np.load(str(ARTIFACTS_DIR / "candidate_ids.npy"), allow_pickle=True)
    jd_embedding = np.load(str(ARTIFACTS_DIR / "jd_embedding.npy")).astype(np.float32)

    with open(ARTIFACTS_DIR / "subscores.pkl", "rb") as f:
        subscores_dict = pickle.load(f)

    n = len(candidate_ids)
    print(f"Loaded {n} candidates, embeddings shape: {embeddings.shape}")

    semantic_sim = embeddings @ jd_embedding

    weight_vector = np.array(
        [WEIGHTS["technical_fit"], WEIGHTS["career_quality"], WEIGHTS["availability_signal"], WEIGHTS["seniority_fit"]],
        dtype=np.float32,
    )

    subscore_matrix = np.zeros((n, 4), dtype=np.float32)
    penalty_multipliers = np.ones(n, dtype=np.float32)

    for i, cid in enumerate(candidate_ids):
        ss = subscores_dict.get(cid, {})
        subscore_matrix[i, 0] = ss.get("technical_fit", 0.0)
        subscore_matrix[i, 1] = ss.get("career_quality", 0.0)
        subscore_matrix[i, 2] = ss.get("availability_signal", 0.0)
        subscore_matrix[i, 3] = ss.get("seniority_fit", 0.0)
        penalty_multipliers[i] = ss.get("penalty_multiplier", 1.0)

    print("Computing composite scores ...")
    base_scores = subscore_matrix @ weight_vector
    scores = penalty_multipliers * (base_scores + WEIGHTS["semantic_similarity"] * semantic_sim)

    top_indices = np.argpartition(-scores, TOP_K)[:TOP_K]
    top_k_pairs = [(float(scores[i]), str(candidate_ids[i])) for i in top_indices]

    top_k_pairs.sort(key=lambda x: (-round(x[0], 3), x[1]))

    top_ids = np.array([p[1] for p in top_k_pairs], dtype=object)

    print(f"Top score: {top_k_pairs[0][0]:.4f}, bottom: {top_k_pairs[-1][0]:.4f}")

    print("Loading top candidates for reasoning ...")
    top_candidates = []
    for score_val, cid in top_k_pairs:
        cand = _load_candidate_by_id(cid)
        top_candidates.append(cand)

    output_path = OUTPUT_DIR / "submission.csv"
    print(f"Writing {output_path} ...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, (score_val, cid) in enumerate(top_k_pairs):
            rank = rank_idx + 1
            cand = top_candidates[rank_idx]
            ss = subscores_dict.get(cid, {})
            penalty_mult = ss.get("penalty_multiplier", 1.0)
            reasoning = _generate_reasoning(cand or {}, score_val, ss, penalty_mult)
            writer.writerow([cid, rank, f"{round(score_val, 3):.3f}", reasoning])

    elapsed = time.time() - t0
    print(f"Ranking complete in {elapsed:.1f}s")

    print("Validating submission ...")
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), str(output_path)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        print("VALIDATION FAILED")
        sys.exit(1)
    print("Submission valid!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
