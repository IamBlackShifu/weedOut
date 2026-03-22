"""Tests for the text preprocessing pipeline."""

import pytest

from weedout.preprocessing.text_cleaner import TextCleaner


@pytest.fixture()
def cleaner():
    return TextCleaner()


class TestClean:
    def test_lowercases_text(self, cleaner):
        assert cleaner.clean("HELLO World") == "hello world"

    def test_removes_url(self, cleaner):
        result = cleaner.clean("check this out https://example.com ok")
        assert "<url>" in result
        assert "https://" not in result

    def test_replaces_mention_with_token(self, cleaner):
        result = cleaner.clean("hey @alice what's up")
        assert "<mention>" in result
        assert "@alice" not in result

    def test_removes_mention_when_flag_off(self):
        c = TextCleaner(keep_mentions=False)
        result = c.clean("hey @alice!")
        assert "@alice" not in result
        assert "<mention>" not in result

    def test_keeps_hashtag_text(self, cleaner):
        result = cleaner.clean("love #python programming")
        assert "python" in result
        assert "#" not in result

    def test_removes_hashtag_when_flag_off(self):
        c = TextCleaner(keep_hashtags=False)
        result = c.clean("love #python programming")
        assert "python" not in result

    def test_collapses_repeated_punctuation(self, cleaner):
        result = cleaner.clean("wow!!! really???")
        assert "!!!" not in result
        assert "???" not in result

    def test_converts_emoji(self, cleaner):
        # emoji.demojize replaces 😡 with something like ":enraged_face:"
        result = cleaner.clean("I am 😡 at you")
        assert "😡" not in result

    def test_handles_empty_string(self, cleaner):
        assert cleaner.clean("") == ""

    def test_handles_whitespace_only(self, cleaner):
        assert cleaner.clean("   ") == ""


class TestTokenize:
    def test_returns_list(self, cleaner):
        tokens = cleaner.tokenize("hello world")
        assert isinstance(tokens, list)

    def test_basic_tokenization(self, cleaner):
        tokens = cleaner.tokenize("Hello world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_stopword_removal(self):
        c = TextCleaner(remove_stopwords=True)
        tokens = c.tokenize("this is a test")
        # "this", "is", "a" should be removed as stop-words
        assert "is" not in tokens
        assert "a" not in tokens

    def test_stemming(self):
        c = TextCleaner(stem=True)
        tokens = c.tokenize("running runs")
        # Porter stemmer collapses "running" and "runs" to the same stem "run"
        assert len(set(tokens)) == 1

    def test_punctuation_stripped(self, cleaner):
        tokens = cleaner.tokenize("wow!!!")
        assert "!" not in tokens
        assert "!!!" not in tokens
