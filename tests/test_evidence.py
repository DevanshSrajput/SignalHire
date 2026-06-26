import pytest
from evidence import collect_evidence, generate_reasoning, _snippet_around


class TestCollectEvidence:
    def test_must_have_matched(self):
        cand = {
            "profile": {"current_title": "ML Engineer", "headline": "", "summary": ""},
            "skills": [{"name": "python", "proficiency": "expert"}],
            "career_history": [
                {"title": "Engineer", "description": "built embeddings and retrieval"},
            ],
        }
        ev = collect_evidence(cand)
        matched_names = {m["criterion"] for m in ev["matched"]}
        assert "python" in matched_names
        assert "embeddings / retrieval" in matched_names

    def test_missing_must_haves_reported(self):
        cand = {
            "profile": {},
            "skills": [],
            "career_history": [],
        }
        ev = collect_evidence(cand)
        assert len(ev["missing_must_haves"]) > 0

    def test_production_evidence_detected(self):
        cand = {
            "profile": {},
            "skills": [],
            "career_history": [
                {"title": "Engineer", "description": "deployed ranking system to production"},
            ],
        }
        ev = collect_evidence(cand)
        production_keywords = {h["keyword"] for h in ev["production"]}
        assert "production" in production_keywords

    def test_no_production_evidence_when_absent(self):
        cand = {
            "profile": {},
            "skills": [],
            "career_history": [
                {"title": "Engineer", "description": "worked on internal tools"},
            ],
        }
        ev = collect_evidence(cand)
        assert len(ev["production"]) == 0

    def test_retrieval_signals_detected(self):
        cand = {
            "profile": {},
            "skills": [],
            "career_history": [
                {"title": "Engineer", "description": "built rag pipeline with faiss"},
            ],
        }
        ev = collect_evidence(cand)
        retrieval_keywords = {h["keyword"] for h in ev["retrieval"]}
        assert "rag" in retrieval_keywords or "faiss" in retrieval_keywords


class TestGenerateReasoning:
    def test_only_claims_production_when_present(self):
        cand_with_prod = {
            "profile": {"current_title": "ML Engineer", "years_of_experience": 5},
            "skills": [{"name": "python", "proficiency": "expert"}],
            "career_history": [
                {"title": "Engineer", "description": "deployed to production at scale"},
            ],
            "redrob_signals": {},
        }
        r = generate_reasoning(cand_with_prod)
        assert "production" in r.lower()

    def test_no_production_claim_when_absent(self):
        cand_no_prod = {
            "profile": {"current_title": "ML Engineer", "years_of_experience": 5},
            "skills": [{"name": "python", "proficiency": "expert"}],
            "career_history": [
                {"title": "Engineer", "description": "internal research"},
            ],
            "redrob_signals": {},
        }
        r = generate_reasoning(cand_no_prod)
        assert "production" not in r.lower()

    def test_max_len_truncation(self):
        cand = {
            "profile": {"current_title": "Engineer", "years_of_experience": 10},
            "skills": [{"name": "python", "proficiency": "advanced"}],
            "career_history": [
                {"title": "Engineer", "description": "worked on " + "x" * 500},
            ],
            "redrob_signals": {},
        }
        r = generate_reasoning(cand, max_len=50)
        assert len(r) <= 50

    def test_empty_candidate(self):
        assert generate_reasoning({}) == ""


class TestSnippetAround:
    def test_basic_snippet(self):
        text = "the quick brown fox jumps over the lazy dog"
        kw = "fox"
        snippet = _snippet_around(text, kw)
        assert "fox" in snippet

    def test_keyword_at_start(self):
        text = "fox jumps over the lazy dog"
        kw = "fox"
        snippet = _snippet_around(text, kw)
        assert snippet.startswith("fox")
