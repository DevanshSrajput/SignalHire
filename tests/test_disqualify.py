import pytest
from disqualify import (
    is_honeypot,
    _all_roles_at_consulting,
    _is_pure_research,
    _is_no_code_18mo,
    _is_cv_speech_robotics_only,
    should_hard_penalize,
    parse_year,
)


class TestHoneypot:
    def test_impossible_yoe_detected(self):
        cand = {
            "profile": {"years_of_experience": 50},
            "career_history": [
                {"start_date": "2020-01", "title": "Engineer"},
            ],
            "redrob_signals": {},
        }
        flagged, reason = is_honeypot(cand)
        assert flagged
        assert "HONEYPOT" in reason

    def test_normal_profile_not_honeypot(self):
        cand = {
            "profile": {"years_of_experience": 5},
            "career_history": [
                {"start_date": "2018-01", "title": "Engineer"},
            ],
            "redrob_signals": {"profile_completeness_score": 80, "verified_email": True},
        }
        flagged, reason = is_honeypot(cand)
        assert not flagged
        assert reason == ""


class TestGhost:
    def test_low_completeness_no_contact(self):
        cand = {
            "profile": {},
            "career_history": [],
            "redrob_signals": {
                "profile_completeness_score": 2,
                "verified_email": False,
                "verified_phone": False,
            },
        }
        flagged, reason = is_honeypot(cand)
        assert flagged
        assert "GHOST" in reason

    def test_low_completeness_but_verified_not_ghost(self):
        cand = {
            "profile": {},
            "career_history": [],
            "redrob_signals": {
                "profile_completeness_score": 2,
                "verified_email": True,
                "verified_phone": False,
            },
        }
        flagged, reason = is_honeypot(cand)
        assert not flagged or "GHOST" not in reason


class TestPureResearch:
    def test_all_research_titles_no_production(self):
        cand = {
            "career_history": [
                {"title": "Research Scientist", "description": "published papers"},
                {"title": "Research Engineer", "description": "conducted experiments"},
            ],
        }
        assert _is_pure_research(cand) is True

    def test_research_with_production_evidence(self):
        cand = {
            "career_history": [
                {"title": "Research Scientist", "description": "deployed model to production"},
            ],
        }
        assert _is_pure_research(cand) is False

    def test_non_research_title(self):
        cand = {
            "career_history": [
                {"title": "Software Engineer", "description": "built features"},
            ],
        }
        assert _is_pure_research(cand) is False


class TestConsultingPenalty:
    def test_all_roles_at_consulting(self):
        cand = {
            "career_history": [
                {"company": "TCS"},
                {"company": "Infosys"},
            ],
        }
        assert _all_roles_at_consulting(cand) is True

    def test_empty_company_names_not_consulting(self):
        cand = {
            "career_history": [
                {"company": ""},
                {"company": ""},
            ],
        }
        assert _all_roles_at_consulting(cand) is False

    def test_mixed_not_all_consulting(self):
        cand = {
            "career_history": [
                {"company": "TCS"},
                {"company": "Google"},
            ],
        }
        assert _all_roles_at_consulting(cand) is False


class TestCVSpeechRobotics:
    def test_version_control_does_not_trigger(self):
        cand = {
            "skills": [{"name": "version control"}],
            "career_history": [
                {"title": "Software Engineer", "description": "used git for version control"},
            ],
        }
        assert _is_cv_speech_robotics_only(cand) is False

    def test_computer_vision_triggers(self):
        cand = {
            "skills": [{"name": "computer vision"}],
            "career_history": [
                {"title": "CV Engineer", "description": "object detection with YOLO"},
            ],
        }
        assert _is_cv_speech_robotics_only(cand) is True

    def test_cv_with_retrieval_signals_exempted(self):
        cand = {
            "skills": [{"name": "computer vision"}, {"name": "python"}],
            "career_history": [
                {"title": "ML Engineer", "description": "built vector search with faiss"},
            ],
        }
        assert _is_cv_speech_robotics_only(cand) is False

    def test_robotics_keyword_matches(self):
        cand = {
            "skills": [{"name": "ROS"}],
            "career_history": [
                {"title": "ML Engineer", "description": "slam and motion planning"},
            ],
        }
        assert _is_cv_speech_robotics_only(cand) is True


class TestNoCode:
    def test_no_code_18mo_trigger(self):
        cand = {
            "redrob_signals": {
                "last_active_date": "2024-01-01",
            },
            "career_history": [
                {"title": "Engineer", "end_date": "2023-06", "duration_months": 6},
            ],
        }
        assert _is_no_code_18mo(cand) is True

    def test_current_role_not_no_code(self):
        cand = {
            "redrob_signals": {
                "last_active_date": "2024-01-01",
            },
            "career_history": [
                {"title": "Engineer", "is_current": True},
            ],
        }
        assert _is_no_code_18mo(cand) is False


class TestShouldHardPenalize:
    def test_all_consulting_penalty(self):
        cand = {
            "career_history": [
                {"company": "TCS", "title": "Engineer", "description": "coding"},
            ],
            "redrob_signals": {"last_active_date": "2026-05-01"},
            "profile": {"current_title": "Engineer"},
            "skills": [{"name": "python"}],
        }
        penalty, reason = should_hard_penalize(cand)
        assert penalty < 1.0
        assert "ALL_ROLES_CONSULTING" in reason

    def test_no_penalty_for_clean_profile(self):
        cand = {
            "career_history": [
                {"company": "Google", "title": "Engineer", "description": "coding"},
            ],
            "redrob_signals": {"last_active_date": "2026-05-01"},
            "profile": {"current_title": "Engineer"},
            "skills": [{"name": "python"}],
        }
        penalty, reason = should_hard_penalize(cand)
        assert penalty == 1.0
        assert reason == ""


class TestParseYear:
    def test_parse_four_digit_year(self):
        assert parse_year("2020") == 2020
        assert parse_year("2020-01") == 2020

    def test_present_returns_current(self):
        assert parse_year("present") == 2026
        assert parse_year("now") == 2026

    def test_none_returns_none(self):
        assert parse_year(None) is None
