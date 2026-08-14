# SPDX-License-Identifier: Apache-2.0

"""Small per-crop language router for the bilingual Option 4 OCR path.

The router intentionally has no model or third-party runtime dependency.  A
Tesseract ``vie+eng`` probe supplies the text signal; only strong Vietnamese
signals are sent to the Vietnamese-only Tesseract recognizer.  English,
mixed, short, noisy, or uncertain crops are routed to multilingual Nemotron.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

VIETNAMESE = "vietnamese"
NON_VIETNAMESE = "non_vietnamese"
UNCERTAIN = "uncertain"
ENGLISH = "english"


@dataclass(frozen=True)
class LanguageDecision:
    """Decision made from one bilingual Tesseract probe."""

    route: str
    confidence: float | None
    reason: str
    probe_score: float | None = None
    probe_text: str = ""

    @property
    def is_vietnamese(self) -> bool:
        return self.route == VIETNAMESE

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "probe_score": self.probe_score,
            "probe_text": self.probe_text[:240],
        }


_VIETNAMESE_SPECIFIC = set("ăâđêôơưĂÂĐÊÔƠƯ")
_WORD_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+")

# Option 3 uses the complete Vietnamese alphabet, including every composed
# tone-marked form.  This is deliberately kept separate from the conservative
# Tesseract-probe alphabet above so Option 4's behavior remains unchanged.
_OPTION3_VIETNAMESE_UNICODE = frozenset(
    "àáảãạăằắẳẵặâầấẩẫậđ"
    "èéẻẽẹêềếểễệìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐ"
    "ÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
)
_OPTION3_CODE_RE = re.compile(r"^[\W_]*[A-Za-z0-9][A-Za-z0-9._:/+\-#]*[\W_]*$")


@dataclass(frozen=True)
class NemotronLanguageDecision:
    """Raw-Nemotron language decision used only by Option 3."""

    route: str
    confidence: float | None
    reason: str
    language_probabilities: Mapping[str, float] = field(default_factory=dict)
    page_prior: Mapping[str, Any] | None = None
    page_prior_used: bool = False
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reason": self.reason,
            "confidence": self.confidence,
            "language_confidence": self.confidence,
            "language_probabilities": dict(self.language_probabilities or {}),
            "page_prior": dict(self.page_prior) if self.page_prior else None,
            "page_prior_used": self.page_prior_used,
            "raw_text": self.raw_text[:240],
        }


def detect_probe_language(
    text: str,
    score: float | None,
    *,
    min_probe_score: float = 0.70,
) -> LanguageDecision:
    """Route a bilingual probe conservatively.

    A crop is classified as Vietnamese only when the probe has a usable
    confidence and contains Vietnamese-specific letters.  ASCII-only text is
    deliberately routed away from Tesseract ``vie`` because unaccented
    Vietnamese and English are indistinguishable from the image probe; the
    safe destination for both cases is multilingual Nemotron.
    """

    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    words = _WORD_RE.findall(compact)
    letters = [char for char in compact if char.isalpha()]
    if not compact or not words or not letters:
        return LanguageDecision(
            route=UNCERTAIN,
            confidence=None,
            reason="probe_empty_or_non_text",
            probe_score=score,
            probe_text=compact,
        )

    normalized_score = None if score is None else max(0.0, min(1.0, float(score)))
    if normalized_score is None or normalized_score < float(min_probe_score):
        return LanguageDecision(
            route=UNCERTAIN,
            confidence=normalized_score,
            reason="probe_confidence_below_language_threshold",
            probe_score=normalized_score,
            probe_text=compact,
        )

    specific_count = sum(char in _VIETNAMESE_SPECIFIC for char in compact)
    if specific_count:
        signal_strength = min(1.0, specific_count / max(1, len(letters)))
        return LanguageDecision(
            route=VIETNAMESE,
            confidence=min(0.99, 0.70 + 0.25 * signal_strength),
            reason="vietnamese_specific_letter_detected",
            probe_score=normalized_score,
            probe_text=compact,
        )

    # English and unaccented Vietnamese share the Latin alphabet.  They both
    # intentionally take the non-Vietnamese/uncertain branch to Nemotron.
    if len(words) >= 2 or len(letters) >= 4:
        return LanguageDecision(
            route=NON_VIETNAMESE,
            confidence=min(0.95, 0.55 + 0.35 * normalized_score),
            reason="latin_text_without_vietnamese_specific_letters",
            probe_score=normalized_score,
            probe_text=compact,
        )

    return LanguageDecision(
        route=UNCERTAIN,
        confidence=normalized_score,
        reason="probe_too_short_to_route",
        probe_score=normalized_score,
        probe_text=compact,
    )


def detect_nemotron_page_prior(
    text: str,
    *,
    min_chars: int = 24,
    min_words: int = 4,
) -> dict[str, Any] | None:
    """Build an optional page prior from raw Nemotron text.

    ``langdetect`` is the repository's existing lightweight language-ID
    mechanism.  It is optional at import time so an unavailable detector can
    never make OCR fail; the caller then falls back to Unicode routing.
    """

    compact = _compact_text(text)
    if not _is_long_enough(compact, min_chars=min_chars, min_words=min_words):
        return None
    probabilities, detector_error = _langdetect_probabilities(compact)
    if not probabilities and detector_error:
        return {
            "available": False,
            "error": detector_error,
            "probabilities": {},
        }
    if not probabilities:
        return None
    return {
        "available": True,
        "probabilities": probabilities,
        "vi": probabilities.get("vi", 0.0),
        "en": probabilities.get("en", 0.0),
        "confidence": max(probabilities.values()),
    }


def route_nemotron_text(
    text: str,
    *,
    page_prior: Mapping[str, Any] | None = None,
    min_chars: int = 24,
    min_words: int = 4,
) -> NemotronLanguageDecision:
    """Route raw Nemotron text to Vietnamese, English, or uncertain.

    No accent-stripping normalization is performed.  Unicode evidence is
    evaluated first; probabilistic detection is used only for sufficiently
    long text and never receives an image.
    """

    compact = _compact_text(text)
    probabilities: dict[str, float] = {}
    if _has_option3_vietnamese_signal(compact):
        return NemotronLanguageDecision(
            route=VIETNAMESE,
            confidence=1.0,
            reason="strong_vietnamese_unicode_signal",
            language_probabilities=probabilities,
            page_prior=page_prior,
            raw_text=compact,
        )

    if not compact or not _has_letter_or_number(compact):
        return _option3_uncertain(
            compact,
            "empty_numeric_or_symbol_text",
            page_prior=page_prior,
        )

    if _is_numeric_or_code(compact):
        return _option3_uncertain(
            compact,
            "numeric_code_or_symbol_text",
            page_prior=page_prior,
        )

    long_enough = _is_long_enough(
        compact,
        min_chars=min_chars,
        min_words=min_words,
    )
    if not long_enough:
        prior_decision = _route_from_probabilities(
            compact,
            _prior_probabilities(page_prior),
            reason_prefix="page_prior",
            page_prior=page_prior,
            page_prior_used=True,
        )
        if prior_decision is not None:
            return prior_decision
        return _option3_uncertain(
            compact,
            "text_too_short_for_language_detection",
            page_prior=page_prior,
        )

    probabilities, detector_error = _langdetect_probabilities(compact)
    detected = _route_from_probabilities(
        compact,
        probabilities,
        reason_prefix="langdetect",
        page_prior=page_prior,
    )
    if detected is not None:
        return detected
    return _option3_uncertain(
        compact,
        "language_detector_error" if detector_error else "mixed_or_low_language_confidence",
        page_prior=page_prior,
        probabilities=probabilities,
    )


def _route_from_probabilities(
    text: str,
    probabilities: Mapping[str, float],
    *,
    reason_prefix: str,
    page_prior: Mapping[str, Any] | None,
    page_prior_used: bool = False,
) -> NemotronLanguageDecision | None:
    vi = float(probabilities.get("vi", 0.0))
    en = float(probabilities.get("en", 0.0))
    if vi >= 0.80 and vi - en >= 0.20:
        return NemotronLanguageDecision(
            route=VIETNAMESE,
            confidence=vi,
            reason=f"{reason_prefix}_vietnamese_threshold",
            language_probabilities=probabilities,
            page_prior=page_prior,
            page_prior_used=page_prior_used,
            raw_text=text,
        )
    if en >= 0.80 and not _has_option3_vietnamese_signal(text):
        return NemotronLanguageDecision(
            route=ENGLISH,
            confidence=en,
            reason=f"{reason_prefix}_english_threshold",
            language_probabilities=probabilities,
            page_prior=page_prior,
            page_prior_used=page_prior_used,
            raw_text=text,
        )
    return None


def _langdetect_probabilities(text: str) -> tuple[dict[str, float], str | None]:
    try:
        from langdetect import DetectorFactory, detect_langs

        # langdetect's global seed is the supported deterministic switch.
        DetectorFactory.seed = 0
        detected = detect_langs(text)
    except Exception as exc:  # noqa: BLE001 - detector is optional and page-local
        return {}, f"{type(exc).__name__}: {exc}"
    probabilities: dict[str, float] = {}
    for item in detected:
        language = str(getattr(item, "lang", "") or "").lower()
        try:
            probability = max(0.0, min(1.0, float(getattr(item, "prob", 0.0))))
        except (TypeError, ValueError):
            continue
        if language in {"vi", "en"}:
            probabilities[language] = probability
    return probabilities, None


def _prior_probabilities(page_prior: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(page_prior, Mapping):
        return {}
    raw = page_prior.get("probabilities")
    if isinstance(raw, Mapping):
        return {
            str(language): float(value)
            for language, value in raw.items()
            if str(language) in {"vi", "en"}
            and isinstance(value, (int, float))
        }
    result: dict[str, float] = {}
    for language in ("vi", "en"):
        value = page_prior.get(language)
        if isinstance(value, (int, float)):
            result[language] = float(value)
    return result


def _option3_uncertain(
    text: str,
    reason: str,
    *,
    page_prior: Mapping[str, Any] | None,
    probabilities: Mapping[str, float] | None = None,
) -> NemotronLanguageDecision:
    return NemotronLanguageDecision(
        route=UNCERTAIN,
        confidence=None,
        reason=reason,
        language_probabilities=dict(probabilities or {}),
        page_prior=page_prior,
        raw_text=text,
    )


def _compact_text(text: str) -> str:
    # Whitespace compaction preserves every non-whitespace Unicode character,
    # including Vietnamese combining/composed marks.
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_long_enough(text: str, *, min_chars: int, min_words: int) -> bool:
    words = _WORD_RE.findall(text)
    letters = [char for char in text if char.isalpha()]
    return len(letters) >= int(min_chars) or len(words) >= int(min_words)


def _has_option3_vietnamese_signal(text: str) -> bool:
    return any(char in _OPTION3_VIETNAMESE_UNICODE for char in text)


def _has_letter_or_number(text: str) -> bool:
    return any(char.isalpha() or char.isdigit() for char in text)


def _is_numeric_or_code(text: str) -> bool:
    if not any(char.isalpha() for char in text):
        return True
    if not _OPTION3_CODE_RE.fullmatch(text):
        return False
    # A plain short English word is not a code.  Codes generally expose a
    # digit or a structural separator, which is enough to suppress page-prior
    # borrowing without misclassifying normal prose.
    return any(char.isdigit() or char in "-_/.:+#" for char in text)
