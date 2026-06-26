import pytest
from signals import (
    compute_technical_fit,
    compute_career_quality,
    compute_availability_signal,
    compute_seniority_fit,
    _title_seniority_level,
    _has_upward_title_progression,
    _check_education_tier,
)


class TestSeniorityLevel:
    def test_senior_principal_engineer(self):
        assert _title_seniority_level("Senior Principal Engineer") == 4

    def test_international_sales_lead(self):
        assert _title_seniority_level("International Sales Lead") == 3

    def test_vp(self):
        assert _title_seniority_level("Vice President of Engineering") == 6

    def test_intern_does_not_match_international(self):
        assert _title_seniority_level("International Marketing Manager") == 3

    def test_default_mid_level(self):
        assert _title_seniority_level("Software Engineer") == 2


class TestCareerQuality:
    def test_empty_companies_do_not_zero_score(self):
        cand = {
            "career_history": [
                {"company": "", "title": "Engineer", "duration_months": 36},
                {"company": "", "title": "Senior Engineer", "duration_months": 24},
            ],
            "profile": {"current_title": "Engineer"},
        }
        score = compute_career_quality(cand)
        assert score > 0.0

    def test_single_named_company_not_all_consulting(self):
        cand = {
            "career_history": [
                {"company": "Google", "title": "Engineer", "duration_months": 24},
            ],
            "profile": {"current_title": "Engineer"},
        }
        score = compute_career_quality(cand)
        assert score > 0.0

    def test_all_consulting_zeros_score(self):
        cand = {
            "career_history": [
                {"company": "TCS", "title": "Engineer", "duration_months": 24},
                {"company": "Infosys", "title": "Senior Engineer", "duration_months": 12},
            ],
            "profile": {"current_title": "Engineer"},
        }
        score = compute_career_quality(cand)
        assert score == 0.0

    def test_newest_first_still_detects_upward_progression(self):
        cand = {
            "career_history": [
                {"title": "Senior Engineer", "start_date": "2022-01", "is_current": True},
                {"title": "Engineer", "start_date": "2019-03"},
                {"title": "Junior Engineer", "start_date": "2017-06"},
            ],
            "profile": {"current_title": "Senior Engineer"},
        }
        assert _has_upward_title_progression(cand) is True

    def test_no_progression_when_same_level(self):
        cand = {
            "career_history": [
                {"title": "Engineer", "start_date": "2022-01", "is_current": True},
                {"title": "Engineer", "start_date": "2019-03"},
            ],
            "profile": {"current_title": "Engineer"},
        }
        assert _has_upward_title_progression(cand) is False


class TestTechnicalFit:
    def test_empty_profile_returns_zero(self):
        cand = {"career_history": [], "skills": [], "profile": {}}
        score = compute_technical_fit(cand)
        assert score == 0.0

    def test_python_skill_matches_must_have(self):
        cand = {
            "career_history": [],
            "skills": [{"name": "python", "proficiency": "advanced"}],
            "profile": {"headline": "", "summary": "", "current_title": ""},
        }
        score = compute_technical_fit(cand)
        assert score > 0.0

    def test_null_description_does_not_crash(self):
        cand = {
            "career_history": [
                {"title": "Engineer", "description": None},
                {"title": "Senior Engineer", "description": "built search"},
            ],
            "skills": [],
            "profile": {"headline": None, "summary": None, "current_title": None},
        }
        score = compute_technical_fit(cand)
        assert isinstance(score, float)


class TestAvailabilitySignal:
    def test_open_to_work_boosts_score(self):
        cand = {"redrob_signals": {"open_to_work_flag": True}}
        score = compute_availability_signal(cand)
        assert score >= 0.25

    def test_empty_signals_returns_low_score(self):
        cand = {"redrob_signals": {}}
        score = compute_availability_signal(cand)
        assert score > 0.0  # baseline from defaults

    def test_recent_activity_boosts(self):
        cand = {
            "redrob_signals": {
                "last_active_date": "2026-05-15",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.8,
                "interview_completion_rate": 0.5,
                "avg_response_time_hours": 2,
                "applications_submitted_30d": 5,
                "notice_period_days": 15,
            }
        }
        score = compute_availability_signal(cand)
        assert score > 0.5


class TestSeniorityFit:
    def test_ideal_yoe_range(self):
        cand = {"profile": {"years_of_experience": 7}, "education": []}
        score = compute_seniority_fit(cand)
        assert score >= 1.0

    def test_tier_1_education_bonus(self):
        cand = {
            "profile": {"years_of_experience": 7},
            "education": [{"institution": "IIT Bombay", "tier": "tier_1"}],
        }
        score = compute_seniority_fit(cand)
        assert score >= 1.0

    def test_no_education_entries(self):
        cand = {"profile": {"years_of_experience": 3}, "education": []}
        score = compute_seniority_fit(cand)
        assert 0.3 <= score <= 0.5


class TestEducationTier:
    def test_tier_1_iit_keyword(self):
        cand = {"education": [{"institution": "IIT Delhi"}]}
        assert _check_education_tier(cand) == 1

    def test_tier_2_dtu_keyword(self):
        cand = {"education": [{"institution": "DTU Delhi"}]}
        assert _check_education_tier(cand) == 2

    def test_monitoring_does_not_match_tier_2(self):
        cand = {"education": [{"institution": "monitoring university"}]}
        assert _check_education_tier(cand) == 0

    def test_tier_1_iiit_blocks_tier_2_iiit(self):
        cand = {"education": [{"institution": "IIIT Hyderabad"}]}
        assert _check_education_tier(cand) == 1

    def test_unknown_institution(self):
        cand = {"education": [{"institution": "Some Unknown University"}]}
        assert _check_education_tier(cand) == 0

    def test_explicit_tier_from_data(self):
        cand = {"education": [{"tier": "tier_3", "institution": "Some University"}]}
        assert _check_education_tier(cand) == 0  # only tier_1 and tier_2 match
