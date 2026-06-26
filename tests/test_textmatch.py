import pytest
from textmatch import keyword_pattern, matches_keyword, matches_any, matched_keywords


class TestKeywordPattern:
    def test_short_keyword_whole_word(self):
        p = keyword_pattern("rag")
        assert p.search("rag")
        assert p.search("the rag system")
        assert not p.search("storage")
        assert not p.search("raging")

    def test_long_keyword_stem(self):
        p = keyword_pattern("eval")
        assert p.search("eval")
        assert p.search("evaluation")
        assert p.search("evaluate")
        assert not p.search("medieval")

    def test_multi_word_keyword(self):
        p = keyword_pattern("sentence transformer")
        assert p.search("sentence transformer")
        assert p.search("sentence transformers")


class TestMatchesKeyword:
    def test_exact_match(self):
        assert matches_keyword("hello world", "hello")

    def test_no_false_positive(self):
        assert not matches_keyword("storage backend", "rag")
        assert not matches_keyword("roadmap planning", "map")

    def test_stemming(self):
        assert matches_keyword("evaluation results", "eval")
        assert matches_keyword("benchmarking suite", "benchmark")
        assert matches_keyword("benchmark results", "bench")

    def test_version_control_does_not_match_control_systems(self):
        assert not matches_keyword("version control", "control systems")


class TestMatchesAny:
    def test_any_match(self):
        assert matches_any("python and rust", ["python", "java"])
        assert not matches_any("go and rust", ["python", "java"])

    def test_empty_keywords(self):
        assert not matches_any("hello world", [])


class TestMatchedKeywords:
    def test_returns_matching(self):
        result = matched_keywords("i love python and java", ["python", "ruby", "java"])
        assert result == ["python", "java"]

    def test_empty_when_no_match(self):
        assert matched_keywords("hello", ["rag", "map"]) == []
