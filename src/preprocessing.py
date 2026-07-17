"""Vietnamese text preprocessing utilities for sentiment analysis.

This module intentionally separates preprocessing for two branches:
- ML/RNN branch: clean text, then apply underthesea word segmentation.
- PLM branch: clean text only; mBERT/PhoBERT use their own tokenizers later.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, List

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

EMOJI_REPLACEMENTS = {
    "😡": " emoji_tuc_gian ",
    "😠": " emoji_tuc_gian ",
    "🤬": " emoji_tuc_gian ",
    "😢": " emoji_buon ",
    "😭": " emoji_buon ",
    "😞": " emoji_buon ",
    "😔": " emoji_buon ",
    "😊": " emoji_vui ",
    "🙂": " emoji_vui ",
    "😀": " emoji_vui ",
    "😁": " emoji_vui ",
    "😍": " emoji_thich ",
    "❤️": " emoji_thich ",
    "❤": " emoji_thich ",
    "👍": " emoji_thich ",
    "👎": " emoji_khong_thich ",
    "😂": " emoji_cuoi ",
    "🤣": " emoji_cuoi ",
    "😅": " emoji_cuoi ",
    "😐": " emoji_binh_thuong ",
}

EMOTICON_REPLACEMENTS = {
    ":)": " emoticon_vui ",
    ":-)": " emoticon_vui ",
    ":d": " emoticon_vui ",
    ":D": " emoticon_vui ",
    ":(": " emoticon_buon ",
    ":-(": " emoticon_buon ",
    ":@": " emoticon_tuc_gian ",
    "<3": " emoticon_thich ",
}


def normalize_unicode(text: str) -> str:
    """Normalize Vietnamese text to Unicode NFC form."""
    return unicodedata.normalize("NFC", text)


def normalize_emojis(text: str) -> str:
    """Convert common emojis/emoticons into stable sentiment tokens."""
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)

    # Both :d and :D are handled because cased PLM preprocessing
    # intentionally preserves uppercase characters.
    for emoticon, replacement in EMOTICON_REPLACEMENTS.items():
        text = text.replace(emoticon, replacement)

    return text


def clean_text(text: str, *, lowercase: bool = True) -> str:
    """
    Clean one Vietnamese text sample in the required fixed order.

    Order:
    1. Unicode NFC normalization
    2. Lowercase when lowercase=True
    3. Remove URLs and e-mails
    4. Remove HTML tags and unescape HTML entities
    5. Normalize emojis/emoticons
    6. Normalize whitespace
    7. Do not remove punctuation completely

    The default lowercase=True preserves the existing ML/RNN pipeline.
    Use lowercase=False for cased pretrained language models such as
    bert-base-multilingual-cased.
    """
    if text is None:
        return ""

    # 1. Unicode NFC normalization
    text = normalize_unicode(str(text))

    # 2. Lowercase when requested.
    # The default remains True to preserve the completed ML/RNN pipelines.
    if lowercase:
        text = text.lower()

    # 3. Remove URLs and e-mails
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)

    # 4. Remove HTML tags and unescape entities such as &nbsp;
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 5. Normalize emojis/emoticons
    text = normalize_emojis(text)

    # 6. Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    # 7. Punctuation such as ?, !, ... is intentionally preserved.
    return text


def tokenize_for_ml_rnn(text: str) -> str:
    """
    Prepare text for NB/SVM/BiLSTM/GRU: clean text, then word-tokenize.

    This branch is intended for traditional ML and RNN-based baselines.
    """
    cleaned = clean_text(text)

    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise ImportError(
            "underthesea is required for ML/RNN tokenization. "
            "Install it with: pip install underthesea"
        ) from exc

    return word_tokenize(cleaned, format="text")


def prepare_for_plm(text: str) -> str:
    """
    Prepare text for mBERT/PhoBERT: clean only.

    Do not apply underthesea here. Use the model tokenizer later, e.g.
    AutoTokenizer.from_pretrained("vinai/phobert-base").
    """
    return clean_text(text)

def prepare_for_mbert(text: str) -> str:
    """
    Prepare raw text for bert-base-multilingual-cased.

    The text is cleaned without lowercasing so the cased tokenizer can
    preserve distinctions such as "AI" versus "ai".

    Do not apply underthesea word segmentation. The mBERT tokenizer
    performs its own subword tokenization.
    """
    return clean_text(text, lowercase=False)


def batch_clean_texts(texts: Iterable[str]) -> List[str]:
    """Clean multiple text samples."""
    return [clean_text(text) for text in texts]


if __name__ == "__main__":
    # Test 3 câu mẫu — chạy python src/preprocessing.py để tự kiểm tra
    test_cases = [
        "Cái app này dở òm 😡😡 xem  http://link.com đi",
        "SẢN PHẨM tốt&nbsp;quá!!! Recommend luôn",
        "bình thường thôi, không có gì đặc biệt...",
    ]

    print("== clean_text ==")
    for t in test_cases:
        print(f"{t!r} -> {clean_text(t)!r}")

    print("\n== prepare_for_plm ==")
    for t in test_cases:
        print(f"{t!r} -> {prepare_for_plm(t)!r}")

    print("\n== prepare_for_mbert ==")
    for t in test_cases:
        print(f"{t!r} -> {prepare_for_mbert(t)!r}")

    # Regression checks: the completed ML/RNN behavior must not change.
    assert clean_text("SẢN PHẨM tốt&nbsp;quá!!! Recommend luôn") == (
        "sản phẩm tốt quá!!! recommend luôn"
    )

    # mBERT must preserve uppercase/lowercase information.
    assert prepare_for_mbert(
        "SẢN PHẨM tốt&nbsp;quá!!! Recommend luôn"
    ) == "SẢN PHẨM tốt quá!!! Recommend luôn"

    assert prepare_for_mbert("AI hỗ trợ tôi") == "AI hỗ trợ tôi"
    assert prepare_for_mbert("AI vui :D") == "AI vui emoticon_vui"
    assert "http" not in prepare_for_mbert(
        "AI tốt http://example.com"
    )

    print("\nPreprocessing assertions: OK")

    print("\n== tokenize_for_ml_rnn ==")
    try:
        for t in test_cases:
            print(f"{t!r} -> {tokenize_for_ml_rnn(t)!r}")
    except ImportError as exc:
        print(f"SKIP ML/RNN tokenization test: {exc}")