# SPDX-License-Identifier: Apache-2.0

"""Tests for the bilingual Option 4 language router."""

from nemo_retriever.common.modality.ocr.isolated.language_router import (
    NON_VIETNAMESE,
    UNCERTAIN,
    VIETNAMESE,
    detect_probe_language,
)


def test_routes_strong_vietnamese_probe_to_tesseract() -> None:
    decision = detect_probe_language("Hợp đồng thuê nhà tại Hà Nội", 0.91)
    assert decision.route == VIETNAMESE
    assert decision.is_vietnamese is True


def test_routes_english_probe_away_from_vietnamese_tesseract() -> None:
    decision = detect_probe_language("OFFICE LEASE AGREEMENT", 0.93)
    assert decision.route == NON_VIETNAMESE
    assert decision.is_vietnamese is False


def test_routes_weak_or_empty_probe_to_safe_fallback() -> None:
    assert detect_probe_language("Hợp đồng", 0.42).route == UNCERTAIN
    assert detect_probe_language("", 0.95).route == UNCERTAIN
