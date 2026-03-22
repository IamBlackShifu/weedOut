"""
Text preprocessing pipeline for social media content.

Handles tokenization, normalization, and cleaning of Twitter-style text including
emojis, hashtags, mentions, and URLs.
"""

from __future__ import annotations

import re
import string
import unicodedata

import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Download required NLTK data on first use (quiet, idempotent)
def _ensure_nltk_data() -> None:
    for resource in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

_tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
_stemmer = PorterStemmer()
_STOP_WORDS = set(stopwords.words("english"))

# Regex patterns
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_REPEATED_PUNCT_RE = re.compile(r"([!?.]){2,}")
_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    """
    Pre-process a single social-media post.

    Parameters
    ----------
    remove_stopwords : bool
        Strip common English stop-words from the token list.
    stem : bool
        Apply Porter stemming to each token.
    keep_mentions : bool
        When *True*, replace ``@handle`` with the literal token ``<MENTION>``;
        when *False*, remove them entirely.
    keep_hashtags : bool
        When *True*, keep the hashtag text (without ``#``);
        when *False*, remove hashtags entirely.
    """

    def __init__(
        self,
        remove_stopwords: bool = False,
        stem: bool = False,
        keep_mentions: bool = True,
        keep_hashtags: bool = True,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.keep_mentions = keep_mentions
        self.keep_hashtags = keep_hashtags

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Return a normalised, cleaned version of *text*."""
        text = self._normalise_unicode(text)
        text = self._convert_emojis(text)
        text = self._remove_urls(text)
        text = self._handle_mentions(text)
        text = self._handle_hashtags(text)
        text = self._normalise_punctuation(text)
        text = self._lowercase(text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Return a list of tokens from *text* after cleaning."""
        cleaned = self.clean(text)
        tokens = _tweet_tokenizer.tokenize(cleaned)
        tokens = [t for t in tokens if t not in string.punctuation and t.strip()]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _STOP_WORDS]
        if self.stem:
            tokens = [_stemmer.stem(t) for t in tokens]
        return tokens

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _convert_emojis(text: str) -> str:
        """Replace emojis with their text description (e.g. 😡 → :angry_face:)."""
        return emoji.demojize(text, delimiters=(" :", ": "))

    @staticmethod
    def _remove_urls(text: str) -> str:
        return _URL_RE.sub(" <URL> ", text)

    def _handle_mentions(self, text: str) -> str:
        if self.keep_mentions:
            return _MENTION_RE.sub(" <MENTION> ", text)
        return _MENTION_RE.sub(" ", text)

    def _handle_hashtags(self, text: str) -> str:
        if self.keep_hashtags:
            # Keep the word, drop the '#' symbol
            return _HASHTAG_RE.sub(r" \1 ", text)
        return _HASHTAG_RE.sub(" ", text)

    @staticmethod
    def _normalise_punctuation(text: str) -> str:
        # Collapse sequences like "!!!" → "!"
        return _REPEATED_PUNCT_RE.sub(r"\1", text)

    @staticmethod
    def _lowercase(text: str) -> str:
        return text.lower()
