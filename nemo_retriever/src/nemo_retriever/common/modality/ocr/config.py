"""Selectors shared by integrated and split document OCR backends."""

from __future__ import annotations

import os
from typing import Literal, Mapping

OCRVersion = Literal["v1", "v2"]
OCRLang = Literal["multi", "english", "vietnamese"]

__all__ = ["OCRLang", "OCRVersion", "resolve_ocr_v2_lang", "resolve_ocr_v2_model_dir"]


def resolve_ocr_v2_model_dir(environ: Mapping[str, str] | None = None) -> str:
    """Return the first configured v2-compatible local model directory."""
    env = os.environ if environ is None else environ
    return (
        env.get("RETRIEVER_NEMOTRON_OCR_MODEL_DIR", "").strip()
        or env.get("NEMOTRON_OCR_MODEL_DIR", "").strip()
        or env.get("NEMOTRON_OCR_V2_MODEL_DIR", "").strip()
    )


def resolve_ocr_v2_lang(ocr_version: OCRVersion = "v2", ocr_lang: OCRLang | None = None) -> str:
    """Map public selectors to Nemotron OCR v2's local language modes."""
    if ocr_version == "v1":
        if ocr_lang is not None:
            raise ValueError("ocr_lang is only supported when ocr_version='v2'.")
        return "v1"
    if ocr_version != "v2":
        raise ValueError("ocr_version must be one of ['v1', 'v2'].")
    if ocr_lang in {None, "multi", "vietnamese"}:
        # Nemotron's multilingual checkpoint is the Vietnamese-capable mode.
        return "multi"
    if ocr_lang == "english":
        return "english"
    raise ValueError("ocr_lang must be one of ['multi', 'english', 'vietnamese'].")
