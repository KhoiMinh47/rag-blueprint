# SPDX-License-Identifier: Apache-2.0

"""Option 2: independent Nemotron baseline with Vietnamese VietOCR routing.

Page Elements v3 and Table Structure v1 provide semantic OCR units.  One
document-level language sample is taken from the same batched Nemotron pass
that returns local text boxes; Vietnamese observations then go to VietOCR,
while English/ambiguous observations keep Nemotron.  A local horizontal
projection is used only when an adapter response has no local bbox.  The
selector remains ``pipeline-ppocrv6`` for API compatibility while the
internal pipeline name describes the actual architecture.
"""

from __future__ import annotations

import re
import time
import random
import hashlib
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.adapters import (
    OCRBackend,
    VietnameseRecognizerBackend,
    make_nemotron_backend,
    make_vietnamese_recognizer,
    recognition_items,
)
from nemo_retriever.common.modality.ocr.isolated.contracts import (
    OCRCandidate,
    OCRPage,
    OCRPageOutput,
    OCRUnit,
    page_value,
)
from nemo_retriever.common.modality.ocr.isolated.geometry import (
    PageImageCropper,
    bbox_iou,
    clamp_bbox,
    containment,
    crop_image_b64,
    expand_bbox_adaptive,
    image_size,
    map_local_bbox,
    text_similarity,
    union_bbox,
)
from nemo_retriever.common.modality.ocr.isolated.language_router import (
    ENGLISH,
    VIETNAMESE,
    NemotronLanguageDecision,
    detect_nemotron_page_prior,
    route_nemotron_text,
)
from nemo_retriever.common.modality.ocr.isolated.multiline import (
    split_multiline_units,
)
from nemo_retriever.common.modality.ocr.isolated.units import (
    build_ocr_units,
    table_payload,
)

OPTION2_SELECTOR = "pipeline-ppocrv6"
OPTION2_PIPELINE_NAME = "option2_nemotron_language_routed_vietnamese_ocr"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Option2Config:
    """Configuration for Option 2's semantic-unit and line OCR branch."""

    language: str | None = "auto"
    skip_native_text: bool = True
    include_table_cells: bool = True
    scan_page_fallback: bool = True
    batch_size: int = 8
    # Option 2 keeps page outputs in input order while allowing a bounded
    # number of page-local Nemotron/VietOCR calls to overlap.  The single-GPU
    # development preset sets the same limit at the VietOCR sidecar.
    page_concurrency: int = 4
    # Keep the semantic bbox unchanged for UI mapping, but make the OCR crop
    # slightly more forgiving than the shared baseline. This protects
    # Vietnamese tone marks when Page Elements returns a tight text box.
    text_crop_padding_scale: float = 1.7
    request_timeout_s: float = 120.0
    vietnamese_score_threshold: float = 0.80
    allow_scoreless_vietnamese: bool = False
    scoreless_language_confidence: float = 0.90
    language_min_chars: int = 24
    language_min_words: int = 4
    document_language_sample_pages: int = 3
    document_language_sample_units_per_page: int = 3
    document_language_confidence: float = 0.95
    # A cheap local projection is a bounded fallback when Nemotron does not
    # return local line geometry. Option 2 never calls a remote line detector;
    # the public selector remains unchanged.
    line_detection: bool = True
    # Kept for configuration compatibility with older Option 2 callers.  It
    # is no longer used to size or invoke a detector service.
    line_detector_batch_size: int = 64
    # VietOCR's batch sidecar can omit a confidence for an otherwise useful
    # projection-fallback line. Accept a non-empty, non-prompt-like result
    # for such a line; Nemotron-backed crops retain the strict score policy.
    allow_scoreless_line_vietnamese: bool = True


@dataclass
class _NemotronObservation:
    unit: OCRUnit
    text: str
    score: float | None
    model: str
    language: str | None
    local_bbox: Sequence[float] | None
    bbox_xyxy_norm: tuple[float, float, float, float]
    bbox_fallback: bool
    error: Exception | None = None


@dataclass
class _CandidateState:
    candidate: OCRCandidate
    unit: OCRUnit
    decision: NemotronLanguageDecision
    debug: dict[str, Any]


@dataclass
class _Option2PageWork:
    """Prepared page state used by the document-wide indexed batch pass."""

    page_index: int
    page: OCRPage
    started: float
    cropper: PageImageCropper
    units: list[OCRUnit]
    document_language_prior: Mapping[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    observations: list[_NemotronObservation] = field(default_factory=list)
    states: list[_CandidateState] = field(default_factory=list)
    page_prior: Mapping[str, Any] | None = None
    router_seconds: float = 0.0
    fallback_count: int = 0
    vietnamese_states: list[_CandidateState] = field(default_factory=list)
    language_probe_seconds: float = 0.0
    language_probe_input_count: int = 0
    direct_vietnamese_input_count: int = 0
    line_detector_seconds: float = 0.0
    line_detector_input_count: int = 0
    line_count: int = 0
    line_split_seconds: float = 0.0
    line_split_input_count: int = 0


class Option2Pipeline:
    """Route a small language sample, then OCR Vietnamese units in batch."""

    pipeline_name = OPTION2_PIPELINE_NAME

    def __init__(
        self,
        nemotron: OCRBackend | Any,
        vietnamese_recognizer: VietnameseRecognizerBackend | Any,
        *,
        line_detector: Any | None = None,
        config: Option2Config | None = None,
    ) -> None:
        if not hasattr(nemotron, "recognize"):
            raise TypeError("nemotron backend must expose recognize(images)")
        if not hasattr(vietnamese_recognizer, "recognize"):
            raise TypeError(
                "vietnamese_recognizer backend must expose recognize(images)"
            )
        self.nemotron = nemotron
        self.vietnamese_recognizer = vietnamese_recognizer
        # The argument remains accepted for old callers, but Option 2 no
        # longer invokes a remote line detector.  Line splitting is local and
        # deterministic; this prevents an accidentally configured sidecar
        # from reintroducing the old latency/VRAM path.
        self.line_detector = None
        self.config = config or Option2Config()
        self._nemotron_response_cache: dict[str, Any] = {}
        self._nemotron_cache_lock = Lock()

    def process_page(
        self,
        page: OCRPage | Mapping[str, Any] | Any,
        *,
        document_language_prior: Mapping[str, Any] | None = None,
    ) -> OCRPageOutput:
        started = time.perf_counter()
        normalized_page = page_value(page)
        if self.config.skip_native_text and _is_native_page(normalized_page):
            return _native_output(normalized_page, time.perf_counter() - started)

        if not normalized_page.image_b64:
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                source=self.pipeline_name,
                model="Nemotron OCR v2",
                language=self.config.language,
                errors=[_error("input", "page image is unavailable")],
                timing=_timing(
                    total_seconds=time.perf_counter() - started,
                    unit_count=0,
                ),
                status="failed",
            )

        cropper = PageImageCropper(normalized_page.image_b64)

        units = build_ocr_units(
            normalized_page,
            include_table_cells=self.config.include_table_cells,
            # Visual regions are deliberately not OCR units in Option 2.
            include_visual_regions=False,
            # Padding normal text crops is useful; table cells stay inside
            # their structure box so adjacent cells cannot be duplicated.
            pad_table_cells=False,
            cropper=cropper,
        )
        units = _remove_table_text_overlaps(normalized_page, units)
        units = _deduplicate_option2_units(units)
        _apply_option2_text_crop_padding(
            normalized_page,
            units,
            self.config,
            cropper=cropper,
        )
        if (
            self.config.scan_page_fallback
            and _is_scan_page(normalized_page)
            and not units
        ):
            fallback = cropper.crop(
                (0.0, 0.0, 1.0, 1.0),
            )
            if fallback is not None:
                units.append(
                    OCRUnit(
                        unit_id=f"page-{normalized_page.page_number or 0}-scan-page",
                        kind="text_block",
                        source="nemotron_scan_page_fallback",
                        bbox_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                        crop_bbox_xyxy_norm=fallback.bbox_xyxy_norm,
                        crop_b64=fallback.image_b64,
                        crop_shape_hw=fallback.shape_hw,
                        reading_order=100000,
                        label="text",
                        metadata={"scan_page_fallback": True},
                    )
                )

        # No semantic units means no detector-shaped recall pass.  This is an
        # intentional phase-one limitation: a line Nemotron misses entirely
        # cannot be recovered without adding a separate detector.
        if not units:
            elapsed = time.perf_counter() - started
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                source=self.pipeline_name,
                model="Nemotron OCR v2",
                language=self.config.language,
                timing=_timing(total_seconds=elapsed, unit_count=0),
                status="completed",
            )

        errors: list[dict[str, Any]] = []
        nemotron_started = time.perf_counter()
        observations = self._recognize_nemotron(units, errors)
        nemotron_seconds = time.perf_counter() - nemotron_started
        valid_observations = [
            observation
            for observation in observations
            if observation.error is None and observation.text.strip()
        ]

        router_started = time.perf_counter()
        page_prior = document_language_prior or detect_nemotron_page_prior(
            "\n".join(observation.text for observation in valid_observations),
            min_chars=int(self.config.language_min_chars),
            min_words=int(self.config.language_min_words),
        )
        states: list[_CandidateState] = []
        for observation in valid_observations:
            decision = _route_option2_text(
                observation.text,
                prior=page_prior,
                config=self.config,
                document_scoped=document_language_prior is not None,
            )
            candidate = _candidate_from_nemotron(observation, decision)
            states.append(
                _CandidateState(
                    candidate=candidate,
                    unit=observation.unit,
                    decision=decision,
                    debug={
                        "unit_id": observation.unit.unit_id,
                        "kind": observation.unit.kind,
                        "route": decision.route,
                        "language_router": decision.to_dict(),
                        "nemotron": {
                            "text": observation.text,
                            "score": observation.score,
                            "model": observation.model,
                            "language": observation.language,
                            "bbox_local": list(observation.local_bbox)
                            if observation.local_bbox is not None
                            else None,
                            "bbox_xyxy_norm": list(observation.bbox_xyxy_norm),
                        },
                        "selected_backend": "nemotron",
                    },
                )
            )
        language_router_seconds = time.perf_counter() - router_started

        route_counts: Counter[str] = Counter(
            _route_bucket(state.decision.route) for state in states
        )
        fallback_count = 0
        vietnamese_input_count = 0
        vietnamese_batch_count = 0
        vietnamese_seconds = 0.0
        vietnamese_states: list[_CandidateState] = []
        for state in states:
            if state.decision.route != VIETNAMESE:
                continue
            crop = _vietnamese_crop(normalized_page, state, cropper=cropper)
            if crop is None:
                fallback_count += 1
                _reject_vietnamese(state, "vietnamese_crop_unavailable")
                continue
            state.debug["vietnamese_input"] = {
                "bbox_xyxy_norm": list(crop.bbox_xyxy_norm),
                "model": getattr(self.vietnamese_recognizer, "model", None),
            }
            state.unit.metadata["vietnamese_crop_bbox_xyxy_norm"] = list(
                crop.bbox_xyxy_norm
            )
            state.debug["_vietnamese_crop"] = crop.image_b64
            vietnamese_states.append(state)

        if vietnamese_states:
            # Exactly one logical call per page.  HTTPImageBackend may split
            # the transport into configured chunks, but the pipeline never
            # loops over lines with one request per crop.
            vietnamese_input_count = len(vietnamese_states)
            vietnamese_batch_count = 1
            vietnamese_started = time.perf_counter()
            responses: list[Any] = []
            backend_error: Exception | None = None
            try:
                responses = list(
                    self.vietnamese_recognizer.recognize(
                        [state.debug["_vietnamese_crop"] for state in vietnamese_states]
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve Nemotron candidates
                backend_error = exc
                errors.append(_error("vietnamese_recognizer", exc))
            vietnamese_seconds = time.perf_counter() - vietnamese_started
            for index, state in enumerate(vietnamese_states):
                fallback_count += 1
                if backend_error is not None:
                    _reject_vietnamese(
                        state,
                        "vietnamese_backend_error",
                        error=backend_error,
                    )
                    continue
                if index >= len(responses):
                    _reject_vietnamese(state, "vietnamese_missing_response")
                    continue
                try:
                    accepted, reason = _apply_vietnamese_result(
                        state,
                        responses[index],
                        self.vietnamese_recognizer,
                        config=self.config,
                    )
                except Exception as exc:  # noqa: BLE001 - malformed candidate falls back
                    accepted = False
                    reason = "vietnamese_response_parse_error"
                    errors.append(_error("vietnamese_recognizer.parse", exc))
                if accepted:
                    fallback_count -= 1
                else:
                    _reject_vietnamese(state, reason)

        canonical = _merge_option2_candidates([state.candidate for state in states])
        canonical.sort(
            key=lambda candidate: (
                int(candidate.reading_order),
                candidate.bbox_xyxy_norm[1],
                candidate.bbox_xyxy_norm[0],
            )
        )
        blocks = [candidate.to_dict() for candidate in canonical]
        tables = _build_tables(normalized_page, canonical)
        text = "\n".join(
            candidate.text
            for candidate in canonical
            if candidate.content_type != "table_cell" and candidate.text.strip()
        )
        scores = [candidate.score for candidate in canonical if candidate.score is not None]
        selected_backends = Counter(
            str(candidate.provenance.get("selected_backend") or "nemotron")
            for candidate in canonical
        )
        debug_candidates = []
        for state in states:
            state.debug.pop("_vietnamese_crop", None)
            debug = dict(state.debug)
            debug["final"] = state.candidate.to_dict()
            debug_candidates.append(debug)

        total_seconds = time.perf_counter() - started
        timing = _timing(
            total_seconds=total_seconds,
            unit_count=len(units),
            nemotron_seconds=nemotron_seconds,
            language_router_seconds=language_router_seconds,
            vietnamese_recognizer_seconds=vietnamese_seconds,
            nemotron_input_count=sum(
                unit.metadata.get("nemotron_cache_hit") is False for unit in units
            ),
            vietnamese_input_count=vietnamese_input_count,
            route_counts={
                "vietnamese": int(route_counts.get("vietnamese", 0)),
                "english": int(route_counts.get("english", 0)),
                "uncertain": int(route_counts.get("uncertain", 0)),
            },
            fallback_count=fallback_count,
            selected_backend_counts=dict(selected_backends),
            nemotron_batch_count=int(
                any(unit.metadata.get("nemotron_cache_hit") is False for unit in units)
            ),
            nemotron_request_count=int(
                any(unit.metadata.get("nemotron_cache_hit") is False for unit in units)
            ),
            vietnamese_batch_count=vietnamese_batch_count,
            vietnamese_request_count=vietnamese_batch_count,
            canonical_block_count=len(canonical),
        )
        if page_prior is not None:
            timing["page_language_prior"] = dict(page_prior)
        if document_language_prior is not None:
            timing["document_language_sample"] = dict(document_language_prior)
        status = (
            "partial"
            if errors and canonical
            else ("failed" if errors and not canonical else "completed")
        )
        return OCRPageOutput(
            pipeline=self.pipeline_name,
            text=text,
            ocr_text_blocks=blocks,
            bbox_xyxy_norm=union_bbox(canonical),
            score=sum(scores) / len(scores) if scores else None,
            confidence=sum(scores) / len(scores) if scores else None,
            source=self.pipeline_name,
            model=(
                "Nemotron OCR v2 + "
                f"{getattr(self.vietnamese_recognizer, 'model', 'VietOCR')}"
                if any(
                    candidate.provenance.get("selected_backend")
                    == _vietnamese_backend_name(self.vietnamese_recognizer)
                    for candidate in canonical
                )
                else "Nemotron OCR v2"
            ),
            language=self.config.language,
            tables=tables,
            candidates=debug_candidates,
            errors=errors,
            timing=timing,
            status=status,
        )

    def _prepare_page_for_batch(
        self,
        value: OCRPage | Mapping[str, Any] | Any,
        page_index: int,
        *,
        document_language_prior: Mapping[str, Any] | None,
    ) -> _Option2PageWork | OCRPageOutput:
        """Prepare one page without invoking either recognizer.

        Page image decoding/cropping is deliberately separated from model
        calls.  ``process_pages`` can then submit all prepared units in one
        indexed Nemotron request and all Vietnamese units in one indexed
        VietOCR request.
        """

        started = time.perf_counter()
        page = page_value(value)
        if self.config.skip_native_text and _is_native_page(page):
            return _native_output(page, time.perf_counter() - started)
        if not page.image_b64:
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                source=self.pipeline_name,
                model="Nemotron OCR v2",
                language=self.config.language,
                errors=[_error("input", "page image is unavailable")],
                timing=_timing(
                    total_seconds=time.perf_counter() - started,
                    unit_count=0,
                ),
                status="failed",
            )

        cropper = PageImageCropper(page.image_b64)
        units = build_ocr_units(
            page,
            include_table_cells=self.config.include_table_cells,
            include_visual_regions=False,
            pad_table_cells=False,
            cropper=cropper,
        )
        units = _remove_table_text_overlaps(page, units)
        units = _deduplicate_option2_units(units)
        _apply_option2_text_crop_padding(page, units, self.config, cropper=cropper)
        if self.config.scan_page_fallback and _is_scan_page(page) and not units:
            fallback = cropper.crop((0.0, 0.0, 1.0, 1.0))
            if fallback is not None:
                units.append(
                    OCRUnit(
                        unit_id=f"page-{page.page_number or 0}-scan-page",
                        kind="text_block",
                        source="nemotron_scan_page_fallback",
                        bbox_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                        crop_bbox_xyxy_norm=fallback.bbox_xyxy_norm,
                        crop_b64=fallback.image_b64,
                        crop_shape_hw=fallback.shape_hw,
                        reading_order=100000,
                        label="text",
                        metadata={"scan_page_fallback": True},
                    )
                )
        if not units:
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                source=self.pipeline_name,
                model="Nemotron OCR v2",
                language=self.config.language,
                timing=_timing(
                    total_seconds=time.perf_counter() - started,
                    unit_count=0,
                ),
                status="completed",
            )
        return _Option2PageWork(
            page_index=page_index,
            page=page,
            started=started,
            cropper=cropper,
            units=units,
            document_language_prior=document_language_prior,
        )

    def _states_from_observations(
        self,
        observations: Sequence[_NemotronObservation],
        *,
        document_language_prior: Mapping[str, Any] | None,
    ) -> tuple[list[_CandidateState], Mapping[str, Any] | None, float]:
        """Route one page's indexed observations without another model call."""

        valid_observations = [
            observation
            for observation in observations
            if observation.error is None and observation.text.strip()
        ]
        router_started = time.perf_counter()
        page_prior = document_language_prior or detect_nemotron_page_prior(
            "\n".join(observation.text for observation in valid_observations),
            min_chars=int(self.config.language_min_chars),
            min_words=int(self.config.language_min_words),
        )
        states: list[_CandidateState] = []
        for observation in valid_observations:
            decision = _route_option2_text(
                observation.text,
                prior=page_prior,
                config=self.config,
                document_scoped=document_language_prior is not None,
            )
            candidate = _candidate_from_nemotron(observation, decision)
            states.append(
                _CandidateState(
                    candidate=candidate,
                    unit=observation.unit,
                    decision=decision,
                    debug={
                        "unit_id": observation.unit.unit_id,
                        "kind": observation.unit.kind,
                        "route": decision.route,
                        "language_router": decision.to_dict(),
                        "nemotron": {
                            "text": observation.text,
                            "score": observation.score,
                            "model": observation.model,
                            "language": observation.language,
                            "bbox_local": list(observation.local_bbox)
                            if observation.local_bbox is not None
                            else None,
                            "bbox_xyxy_norm": list(observation.bbox_xyxy_norm),
                        },
                        "selected_backend": "nemotron",
                    },
                )
            )
        return states, page_prior, time.perf_counter() - router_started

    def _finalize_batched_page(
        self,
        work: _Option2PageWork,
        *,
        nemotron_seconds: float,
        vietnamese_seconds: float,
        batch_page_count: int,
    ) -> OCRPageOutput:
        """Build the normal page contract after document-wide inference."""

        states = work.states
        canonical = _merge_option2_candidates([state.candidate for state in states])
        canonical.sort(
            key=lambda candidate: (
                int(candidate.reading_order),
                candidate.bbox_xyxy_norm[1],
                candidate.bbox_xyxy_norm[0],
            )
        )
        blocks = [candidate.to_dict() for candidate in canonical]
        tables = _build_tables(work.page, canonical)
        text = "\n".join(
            candidate.text
            for candidate in canonical
            if candidate.content_type != "table_cell" and candidate.text.strip()
        )
        scores = [candidate.score for candidate in canonical if candidate.score is not None]
        selected_backends = Counter(
            str(candidate.provenance.get("selected_backend") or "nemotron")
            for candidate in canonical
        )
        debug_candidates = []
        for state in states:
            state.debug.pop("_vietnamese_crop", None)
            debug = dict(state.debug)
            debug["final"] = state.candidate.to_dict()
            debug_candidates.append(debug)

        total_seconds = time.perf_counter() - work.started
        has_nemotron = any(
            unit.metadata.get("nemotron_cache_hit") is False for unit in work.units
        )
        timing = _timing(
            total_seconds=total_seconds,
            unit_count=len(work.units),
            nemotron_seconds=nemotron_seconds,
            language_router_seconds=work.router_seconds,
            vietnamese_recognizer_seconds=vietnamese_seconds,
            nemotron_input_count=sum(
                unit.metadata.get("nemotron_cache_hit") is False for unit in work.units
            ),
            vietnamese_input_count=len(work.vietnamese_states),
            route_counts={
                "vietnamese": sum(state.decision.route == VIETNAMESE for state in states),
                "english": sum(state.decision.route == ENGLISH for state in states),
                "uncertain": sum(
                    state.decision.route not in {VIETNAMESE, ENGLISH}
                    for state in states
                ),
            },
            fallback_count=work.fallback_count,
            selected_backend_counts=dict(selected_backends),
            nemotron_batch_count=int(has_nemotron),
            nemotron_request_count=int(has_nemotron),
            vietnamese_batch_count=int(bool(work.vietnamese_states)),
            vietnamese_request_count=int(bool(work.vietnamese_states)),
            canonical_block_count=len(canonical),
            language_probe_seconds=work.language_probe_seconds,
            language_probe_input_count=work.language_probe_input_count,
            direct_vietnamese_input_count=work.direct_vietnamese_input_count,
            line_detector_seconds=work.line_detector_seconds,
            line_detector_input_count=work.line_detector_input_count,
            line_count=work.line_count,
            line_split_seconds=work.line_split_seconds,
            line_split_input_count=work.line_split_input_count,
        )
        timing.update(
            {
                "batch_scope": "document",
                "batch_page_count": int(batch_page_count),
                "batch_amortized_seconds": total_seconds / max(1, batch_page_count),
            }
        )
        if work.page_prior is not None:
            timing["page_language_prior"] = dict(work.page_prior)
        if work.document_language_prior is not None:
            timing["document_language_sample"] = dict(work.document_language_prior)
        status = (
            "partial"
            if work.errors and canonical
            else ("failed" if work.errors and not canonical else "completed")
        )
        return OCRPageOutput(
            pipeline=self.pipeline_name,
            text=text,
            ocr_text_blocks=blocks,
            bbox_xyxy_norm=union_bbox(canonical),
            score=sum(scores) / len(scores) if scores else None,
            confidence=sum(scores) / len(scores) if scores else None,
            source=self.pipeline_name,
            model=(
                "Nemotron OCR v2 + "
                f"{getattr(self.vietnamese_recognizer, 'model', 'VietOCR')}"
                if any(
                    candidate.provenance.get("selected_backend")
                    == _vietnamese_backend_name(self.vietnamese_recognizer)
                    for candidate in canonical
                )
                else "Nemotron OCR v2"
            ),
            language=self.config.language,
            tables=tables,
            candidates=debug_candidates,
            errors=work.errors,
            timing=timing,
            status=status,
        )

    def _preprobe_document_language(
        self,
        works: Sequence[_Option2PageWork],
        errors: list[dict[str, Any]],
    ) -> tuple[
        list[OCRUnit],
        list[_NemotronObservation],
        Mapping[str, Any] | None,
        float,
    ]:
        """Probe a small deterministic unit sample before routing the rest.

        The old Option 2 path first sent every unit to Nemotron and only then
        sampled a few returned strings.  That was useful for language
        metadata but did not save an inference call.  This pre-probe sends at
        most ``sample_pages * sample_units_per_page`` crops first.  A decisive
        document prior then lets the remaining Vietnamese units go directly
        to the Vietnamese recognizer; ambiguous documents retain the full
        Nemotron path below.
        """

        started = time.perf_counter()
        eligible = [(work, list(work.units)) for work in works if work.units]
        if not eligible:
            return [], [], None, time.perf_counter() - started

        seed = "|".join(
            str(work.page.page_number if work.page.page_number is not None else work.page_index)
            for work, _units in eligible
        )
        chooser = random.Random(seed)
        selected_pages = chooser.sample(
            eligible,
            k=min(
                len(eligible),
                max(1, int(self.config.document_language_sample_pages)),
            ),
        )
        sampled_units: list[OCRUnit] = []
        for _work, units in selected_pages:
            sampled_units.extend(
                chooser.sample(
                    units,
                    k=min(
                        len(units),
                        max(1, int(self.config.document_language_sample_units_per_page)),
                    ),
                )
            )
        if not sampled_units:
            return [], [], None, time.perf_counter() - started

        observations = self._recognize_nemotron(sampled_units, errors)
        valid = [
            observation
            for observation in observations
            if observation.error is None and observation.text.strip()
        ]
        prior = detect_nemotron_page_prior(
            "\n".join(observation.text for observation in valid),
            min_chars=int(self.config.language_min_chars),
            min_words=int(self.config.language_min_words),
        )
        if prior is not None:
            enriched = dict(prior)
            enriched.update(
                {
                    "scope": "document_preprobe",
                    "sample_pages": sorted(
                        int(work.page.page_number)
                        for work, _units in selected_pages
                        if work.page.page_number is not None
                    ),
                    "sample_units": len(sampled_units),
                }
            )
            prior = enriched
        return sampled_units, observations, prior, time.perf_counter() - started

    def _document_prior_route(self, prior: Mapping[str, Any] | None) -> str | None:
        """Return a route only when the sampled document prior is decisive."""

        if not isinstance(prior, Mapping) or not bool(prior.get("available")):
            return None
        vi = float(prior.get("vi") or 0.0)
        en = float(prior.get("en") or 0.0)
        threshold = float(self.config.document_language_confidence)
        if vi >= threshold and vi - en >= 0.20:
            return VIETNAMESE
        if en >= threshold and vi < 0.20:
            return ENGLISH
        return None

    def _process_pages_batched(
        self,
        page_list: Sequence[OCRPage | Mapping[str, Any] | Any],
        *,
        document_language_prior: Mapping[str, Any] | None,
    ) -> list[OCRPageOutput]:
        """Run detect-derived units through document-wide indexed batches.

        The backend response order is never trusted for placement: every
        observation keeps its ``OCRUnit`` object, and every unit belongs to a
        page work item by object identity.  Results are scattered back to
        those page slots and the returned list stays in input order.
        """

        outputs: list[OCRPageOutput | None] = [None] * len(page_list)
        worker_count = min(
            len(page_list),
            max(1, min(4, int(self.config.page_concurrency or 1))),
        )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="option2-prepare",
        ) as executor:
            prepared = list(
                executor.map(
                    lambda item: self._prepare_page_for_batch(
                        item[1],
                        item[0],
                        document_language_prior=document_language_prior,
                    ),
                    enumerate(page_list),
                )
            )
        works: list[_Option2PageWork] = []
        for index, value in enumerate(prepared):
            if isinstance(value, _Option2PageWork):
                works.append(value)
            else:
                outputs[index] = value
        if not works:
            return [output for output in outputs if output is not None]

        all_units = [unit for work in works for unit in work.units]
        unit_owner = {
            id(unit): work
            for work in works
            for unit in work.units
        }
        global_errors: list[dict[str, Any]] = []
        nemotron_started = time.perf_counter()
        # One Nemotron request serves two purposes: it provides the bounded
        # language sample and, crucially, the local text boxes used to crop
        # VietOCR.  The old fast path sent only three parent crops to the
        # probe, then replaced the model's local boxes with CPU projection
        # lines.  On real forms that loses the detector's precise X span and
        # makes VietOCR read neighboring text.  Sending semantic units once
        # in one batch is only a small incremental cost over the probe and
        # preserves the proven Option 3 geometry without a line-detector
        # sidecar.
        observations = self._recognize_nemotron(all_units, global_errors)
        probe_units: list[OCRUnit] = []
        probe_unit_ids: set[int] = set()
        sample_per_page = max(
            1, int(self.config.document_language_sample_units_per_page or 1)
        )
        for work in works:
            selected = list(work.units[:sample_per_page])
            probe_units.extend(selected)
            probe_unit_ids.update(id(unit) for unit in selected)
        probe_observations = [
            observation
            for observation in observations
            if id(observation.unit) in probe_unit_ids
            and observation.error is None
            and observation.text.strip()
        ]

        effective_document_language_prior = document_language_prior
        probe_started = time.perf_counter()
        if effective_document_language_prior is None and probe_observations:
            prior = detect_nemotron_page_prior(
                "\n".join(observation.text for observation in probe_observations),
                min_chars=int(self.config.language_min_chars),
                min_words=int(self.config.language_min_words),
            )
            if prior is not None:
                enriched = dict(prior)
                enriched.update(
                    {
                        "scope": "document_preprobe",
                        "sample_pages": sorted(
                            int(work.page.page_number)
                            for work in works
                            if work.page.page_number is not None
                        ),
                        "sample_units": len(probe_units),
                    }
                )
                effective_document_language_prior = enriched
        if effective_document_language_prior is None:
            valid_observations = [
                observation
                for observation in observations
                if observation.error is None and observation.text.strip()
            ]
            prior = detect_nemotron_page_prior(
                "\n".join(observation.text for observation in valid_observations),
                min_chars=int(self.config.language_min_chars),
                min_words=int(self.config.language_min_words),
            )
            if prior is not None:
                effective_document_language_prior = dict(prior)
                effective_document_language_prior.update(
                    {
                        "scope": "document_observation_batch",
                        "sample_pages": sorted(
                            int(work.page.page_number)
                            for work in works
                            if work.page.page_number is not None
                        ),
                        "sample_units": len(valid_observations),
                    }
                )
        probe_seconds = time.perf_counter() - probe_started
        document_route = self._document_prior_route(effective_document_language_prior)
        line_detector_seconds = 0.0
        line_detector_input_count = 0
        line_count = 0
        line_split_seconds = 0.0
        line_split_input_count = 0

        # Nemotron local boxes are the line geometry in this branch.  No
        # remote line detector is called; the CPU projection helper remains
        # available only for explicit callers/tests that have no local bbox.
        _LOGGER.info(
            "Option 2 routing: pages=%d semantic_units=%d nemotron_observations=%d "
            "language_probe_units=%d document_route=%s local_bbox=%d "
            "algorithmic_line_split=false",
            len(works),
            len(all_units),
            len(observations),
            len(probe_units),
            document_route,
            sum(
                observation.local_bbox is not None
                for observation in observations
                if observation.error is None
            ),
        )

        for work in works:
            work.observations.clear()
            work.language_probe_seconds = probe_seconds
            work.language_probe_input_count = len(probe_units)
            work.line_detector_seconds = line_detector_seconds
            work.line_detector_input_count = line_detector_input_count
            work.line_count = line_count
            work.line_split_seconds = line_split_seconds
            work.line_split_input_count = line_split_input_count
            work.errors.extend(global_errors)
        for observation in observations:
            owner = unit_owner.get(id(observation.unit))
            if owner is not None:
                owner.observations.append(observation)
        if effective_document_language_prior is None:
            effective_document_language_prior = self._document_prior_from_observations(works)

        # This timer covers the one semantic Nemotron batch only.  It is
        # intentionally not mixed with local language classification.
        nemotron_seconds = max(0.0, time.perf_counter() - nemotron_started)

        for work in works:
            states, page_prior, router_seconds = self._states_from_observations(
                work.observations,
                document_language_prior=effective_document_language_prior,
            )
            work.states = states
            work.page_prior = page_prior
            work.document_language_prior = effective_document_language_prior
            work.router_seconds = router_seconds

            if document_route == VIETNAMESE:
                # Nemotron normally supplies the precise local line boxes.
                # If an adapter response has no local geometry, use the
                # existing CPU projection only for that parent as a bounded
                # recall fallback; never call the legacy remote detector.
                fallback_pairs: list[tuple[_Option2PageWork, OCRUnit]] = []
                seen_fallback_units: set[int] = set()
                for state in states:
                    if state.decision.route != VIETNAMESE:
                        continue
                    nemotron_debug = state.debug.get("nemotron")
                    local_bbox = (
                        nemotron_debug.get("bbox_local")
                        if isinstance(nemotron_debug, Mapping)
                        else None
                    )
                    unit_id = id(state.unit)
                    if local_bbox is None and unit_id not in seen_fallback_units:
                        seen_fallback_units.add(unit_id)
                        fallback_pairs.append((work, state.unit))
                if fallback_pairs and self.config.line_detection:
                    (
                        detected_by_work,
                        split_seconds,
                        split_input_count,
                        split_line_count,
                    ) = self._split_option2_algorithmic_line_units(
                        fallback_pairs,
                        global_errors,
                    )
                    fallback_parent_ids = {id(unit) for _work, unit in fallback_pairs}
                    states = [
                        state
                        for state in states
                        if id(state.unit) not in fallback_parent_ids
                    ]
                    for unit in detected_by_work.get(id(work), []):
                        direct_state = _direct_vietnamese_state(
                            unit,
                            effective_document_language_prior,
                            self.vietnamese_recognizer,
                        )
                        direct_state.debug["document_vietnamese_route"] = True
                        states.append(direct_state)
                    work.line_split_seconds += split_seconds
                    work.line_split_input_count += split_input_count
                    work.line_count += split_line_count

                # The document prior only removes repeated language probing;
                # each Nemotron local observation still owns its bbox and is
                # routed independently so stamps/marks/ambiguous fragments
                # can retain the conservative Nemotron result.
                for state in states:
                    if state.decision.route == VIETNAMESE:
                        state.debug["document_vietnamese_route"] = True
                work.direct_vietnamese_input_count = sum(
                    state.decision.route == VIETNAMESE for state in states
                )
            work.states = states
            for state in states:
                if state.decision.route != VIETNAMESE:
                    continue
                crop = _vietnamese_crop(work.page, state, cropper=work.cropper)
                if crop is None:
                    work.fallback_count += 1
                    _reject_vietnamese(state, "vietnamese_crop_unavailable")
                    continue
                state.debug["vietnamese_input"] = {
                    "bbox_xyxy_norm": list(crop.bbox_xyxy_norm),
                    "model": getattr(self.vietnamese_recognizer, "model", None),
                }
                state.unit.metadata["vietnamese_crop_bbox_xyxy_norm"] = list(
                    crop.bbox_xyxy_norm
                )
                state.debug["_vietnamese_crop"] = crop.image_b64
                work.vietnamese_states.append(state)

        vietnamese_states = [
            state for work in works for state in work.vietnamese_states
        ]
        vietnamese_seconds = 0.0
        if vietnamese_states:
            for work in works:
                work.fallback_count += len(work.vietnamese_states)
            vietnamese_started = time.perf_counter()
            responses: list[Any] = []
            backend_error: Exception | None = None
            try:
                responses = list(
                    self.vietnamese_recognizer.recognize(
                        [state.debug["_vietnamese_crop"] for state in vietnamese_states]
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve Nemotron candidates
                backend_error = exc
                for work in works:
                    if work.vietnamese_states:
                        work.errors.append(_error("vietnamese_recognizer", exc))
            vietnamese_seconds = time.perf_counter() - vietnamese_started
            state_owner = {
                id(state): work
                for work in works
                for state in work.vietnamese_states
            }
            for index, state in enumerate(vietnamese_states):
                work = state_owner[id(state)]
                if backend_error is not None:
                    _reject_vietnamese(
                        state,
                        "vietnamese_backend_error",
                        error=backend_error,
                    )
                    continue
                if index >= len(responses):
                    _reject_vietnamese(state, "vietnamese_missing_response")
                    continue
                try:
                    accepted, reason = _apply_vietnamese_result(
                        state,
                        responses[index],
                        self.vietnamese_recognizer,
                        config=self.config,
                    )
                except Exception as exc:  # noqa: BLE001 - malformed candidate remains local
                    accepted = False
                    reason = "vietnamese_response_parse_error"
                    work.errors.append(_error("vietnamese_recognizer.parse", exc))
                if accepted:
                    work.fallback_count -= 1
                else:
                    _reject_vietnamese(state, reason)

        for work in works:
            outputs[work.page_index] = self._finalize_batched_page(
                work,
                nemotron_seconds=nemotron_seconds,
                vietnamese_seconds=vietnamese_seconds,
                batch_page_count=len(works),
            )
        return [output for output in outputs if output is not None]

    def _split_option2_algorithmic_line_units(
        self,
        pairs: Sequence[tuple[_Option2PageWork, OCRUnit]],
        errors: list[dict[str, Any]],
    ) -> tuple[dict[int, list[OCRUnit]], float, int, int]:
        """Split Vietnamese semantic crops with a local image projection.

        This is intentionally CPU-local and deterministic: no PP-OCR line
        endpoint is called.  The shared projection helper only splits a
        clearly multi-line semantic box; ambiguous boxes remain one direct
        VietOCR unit.  Every generated line is re-cropped through Option 2's
        glyph-safe padding policy before it is sent to the recognizer.
        """

        by_work: dict[int, list[OCRUnit]] = {}
        if not pairs:
            return by_work, 0.0, 0, 0

        started = time.perf_counter()
        line_count = 0
        for work, unit in pairs:
            try:
                projected = split_multiline_units(
                    work.page,
                    [unit],
                    cropper=work.cropper,
                    min_height_ratio=1.65,
                    max_lines=64,
                    line_detector=None,
                    errors=errors,
                )
            except Exception as exc:  # noqa: BLE001 - keep parent recall
                errors.append(_error("option2.line_split", exc))
                projected = [unit]

            output_units: list[OCRUnit] = []
            if len(projected) >= 2:
                for child in projected:
                    crop = _option2_line_crop(
                        work.page,
                        unit,
                        child.bbox_xyxy_norm,
                        cropper=work.cropper,
                        padding_scale=self.config.text_crop_padding_scale,
                    )
                    if crop is None:
                        continue
                    child.crop_bbox_xyxy_norm = crop.bbox_xyxy_norm
                    child.crop_b64 = crop.image_b64
                    child.crop_shape_hw = crop.shape_hw
                    child.metadata.update(
                        {
                            "line_detector": None,
                            "line_detector_score": None,
                            "line_crop_padding_scale": self.config.text_crop_padding_scale,
                        }
                    )
                    output_units.append(child)

            if len(output_units) < 2:
                output_units = [
                    _option2_fallback_line_unit(
                        unit,
                        source="option2_algorithmic_parent",
                    )
                ]
            by_work.setdefault(id(work), []).extend(output_units)
            line_count += len(output_units)

        return by_work, time.perf_counter() - started, len(pairs), line_count

    def process_pages(
        self, pages: Sequence[OCRPage | Mapping[str, Any] | Any]
    ) -> list[OCRPageOutput]:
        """Process one document with stable page order and indexed stage batches."""

        page_list = list(pages)
        if not page_list:
            return []
        with self._nemotron_cache_lock:
            self._nemotron_response_cache.clear()
        # Use the document-batched path even for one page.  It performs the
        # bounded language pre-probe; the former one-page shortcut sent every
        # semantic crop to Nemotron before deciding that the document was
        # Vietnamese.
        document_prior = None
        try:
            return self._process_pages_batched(
                page_list,
                document_language_prior=document_prior,
            )
        except Exception as exc:  # noqa: BLE001 - preserve one output per page
            return [
                self._process_page_safe(page, document_language_prior=document_prior)
                for page in page_list
            ] if not isinstance(exc, KeyboardInterrupt) else []

    def _process_page_safe(
        self,
        page: OCRPage | Mapping[str, Any] | Any,
        *,
        document_language_prior: Mapping[str, Any] | None = None,
    ) -> OCRPageOutput:
        try:
            return self.process_page(
                page, document_language_prior=document_language_prior
            )
        except Exception as exc:  # noqa: BLE001 - preserve page-local failure
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                source=self.pipeline_name,
                model="Nemotron OCR v2",
                language=self.config.language,
                errors=[_error("page", exc)],
                status="failed",
            )

    def _document_prior_from_observations(
        self,
        works: Sequence[_Option2PageWork],
    ) -> Mapping[str, Any] | None:
        """Infer the document language from the already completed OCR batch."""

        eligible: list[tuple[_Option2PageWork, list[_NemotronObservation]]] = []
        for work in works:
            observations = [
                observation
                for observation in work.observations
                if observation.error is None and observation.text.strip()
            ]
            if observations:
                eligible.append((work, observations))
        if not eligible:
            return None

        seed = "|".join(
            str(work.page.page_number if work.page.page_number is not None else work.page_index)
            for work, _observations in eligible
        )
        chooser = random.Random(seed)
        selected = chooser.sample(
            eligible,
            k=min(
                len(eligible),
                max(1, int(self.config.document_language_sample_pages)),
            ),
        )
        sampled: list[_NemotronObservation] = []
        for _work, observations in selected:
            sampled.extend(
                chooser.sample(
                    observations,
                    k=min(
                        len(observations),
                        max(1, int(self.config.document_language_sample_units_per_page)),
                    ),
                )
            )
        if not sampled:
            return None

        prior = detect_nemotron_page_prior(
            "\n".join(observation.text for observation in sampled),
            min_chars=int(self.config.language_min_chars),
            min_words=int(self.config.language_min_words),
        )
        if prior is None:
            return None
        result = dict(prior)
        result.update(
            {
                "scope": "document_sample",
                "sample_pages": sorted(
                    int(work.page.page_number)
                    for work, _observations in selected
                    if work.page.page_number is not None
                ),
                "sample_units": len(sampled),
            }
        )
        return result

    def _sample_document_language_prior(
        self, pages: Sequence[OCRPage | Mapping[str, Any] | Any],
    ) -> Mapping[str, Any] | None:
        """OCR a tiny deterministic sample and infer one document language."""

        if len(pages) < 2:
            return None
        eligible: list[OCRPage] = []
        for value in pages:
            page = page_value(value)
            if page.image_b64 and not _is_native_page(page):
                eligible.append(page)
        if not eligible:
            return None
        seed = "|".join(str(page.page_number or index) for index, page in enumerate(eligible))
        chooser = random.Random(seed)
        selected_pages = chooser.sample(
            eligible,
            k=min(len(eligible), max(1, int(self.config.document_language_sample_pages))),
        )
        sampled_units: list[OCRUnit] = []
        for page in selected_pages:
            cropper = PageImageCropper(page.image_b64)
            units = _remove_table_text_overlaps(
                page,
                build_ocr_units(
                    page,
                    include_table_cells=self.config.include_table_cells,
                    include_visual_regions=False,
                    pad_table_cells=False,
                    cropper=cropper,
                ),
            )
            _apply_option2_text_crop_padding(
                page,
                units,
                self.config,
                cropper=cropper,
            )
            if not units:
                continue
            sampled_units.extend(
                chooser.sample(
                    units,
                    k=min(len(units), max(1, int(self.config.document_language_sample_units_per_page))),
                )
            )
        if not sampled_units:
            return None
        observations = self._recognize_nemotron(sampled_units, [])
        prior = detect_nemotron_page_prior(
            "\n".join(item.text for item in observations if item.text.strip()),
            min_chars=int(self.config.language_min_chars),
            min_words=int(self.config.language_min_words),
        )
        if prior is None:
            return None
        sampled = dict(prior)
        sampled.update(
            {
                "scope": "document_sample",
                "sample_pages": sorted(page.page_number for page in selected_pages if page.page_number is not None),
                "sample_units": len(sampled_units),
            }
        )
        return sampled

    def _recognize_nemotron(
        self,
        units: Sequence[OCRUnit],
        errors: list[dict[str, Any]],
    ) -> list[_NemotronObservation]:
        """Send every semantic crop in one logical Nemotron batch."""

        keys = [hashlib.sha256(unit.crop_b64.encode("ascii")).hexdigest() for unit in units]
        responses: list[Any | None] = [None] * len(units)
        missing_indices: list[int] = []
        with self._nemotron_cache_lock:
            for index, key in enumerate(keys):
                if key in self._nemotron_response_cache:
                    responses[index] = self._nemotron_response_cache[key]
                    units[index].metadata["nemotron_cache_hit"] = True
                else:
                    missing_indices.append(index)
        try:
            if missing_indices:
                missing_responses = list(
                    self.nemotron.recognize(
                        [units[index].crop_b64 for index in missing_indices]
                    )
                )
                with self._nemotron_cache_lock:
                    for offset, index in enumerate(missing_indices):
                        response = (
                            missing_responses[offset]
                            if offset < len(missing_responses)
                            else None
                        )
                        responses[index] = response
                        self._nemotron_response_cache[keys[index]] = response
                        units[index].metadata["nemotron_cache_hit"] = False
        except Exception as exc:  # noqa: BLE001 - record page-local baseline failure
            errors.append(_error("nemotron", exc))
            return [
                _NemotronObservation(
                    unit=unit,
                    text="",
                    score=None,
                    model=str(getattr(self.nemotron, "model", "Nemotron OCR v2")),
                    language=getattr(self.nemotron, "language", None),
                    local_bbox=None,
                    bbox_xyxy_norm=unit.bbox_xyxy_norm,
                    bbox_fallback=True,
                    error=exc,
                )
                for unit in units
            ]

        result: list[_NemotronObservation] = []
        for index, unit in enumerate(units):
            response = responses[index] if index < len(responses) else None
            try:
                items = recognition_items(response)
            except Exception as exc:  # noqa: BLE001 - malformed item remains page-local
                errors.append(_error("nemotron.parse", exc))
                result.append(
                    _NemotronObservation(
                        unit=unit,
                        text="",
                        score=None,
                        model=str(
                            getattr(self.nemotron, "model", "Nemotron OCR v2")
                        ),
                        language=getattr(self.nemotron, "language", None),
                        local_bbox=None,
                        bbox_xyxy_norm=unit.bbox_xyxy_norm,
                        bbox_fallback=True,
                        error=exc,
                    )
                )
                continue
            if not items:
                continue
            for item in items:
                local_bbox = item.raw_bbox if item.raw_bbox is not None else item.bbox
                bbox_fallback = local_bbox is None
                mapped_bbox = (
                    map_local_bbox(
                        local_bbox,
                        unit.crop_bbox_xyxy_norm,
                        unit.crop_shape_hw,
                    )
                    if local_bbox is not None
                    else unit.bbox_xyxy_norm
                )
                result.append(
                    _NemotronObservation(
                        unit=unit,
                        text=str(item.text or "").strip(),
                        score=item.score,
                        model=str(
                            item.model
                            or getattr(self.nemotron, "model", "Nemotron OCR v2")
                        ),
                        language=item.language or getattr(self.nemotron, "language", None),
                        local_bbox=local_bbox,
                        bbox_xyxy_norm=mapped_bbox,
                        bbox_fallback=bbox_fallback,
                    )
                )
        return result


def run_option2_batch(
    batch_df: Any,
    *,
    ocr_invoke_url: str | None,
    vietnamese_ocr_invoke_url: str | None,
    api_key: str | None = None,
    ocr_api_key: str | None = None,
    ocr_lang: str | None = None,
    inference_batch_size: int = 8,
    request_timeout_s: float = 120.0,
    scan_ocr_fallback: bool = True,
    extract_text: bool = True,
    extract_tables: bool = True,
    # Kept as a compatibility argument for old graph specs. Option 2 no
    # longer calls this endpoint: Nemotron local boxes are preferred and
    # multi-line splitting is local projection only as a fallback.
    line_detector_invoke_url: str | None = None,
    ocr_recognizer_invoke_url: str | None = None,
    tesseract_ocr_invoke_url: str | None = None,
) -> Any:
    """Run Option 2 with one semantic Nemotron batch and VietOCR routing."""

    del ocr_recognizer_invoke_url, tesseract_ocr_invoke_url
    import pandas as pd

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    missing = [
        name
        for name, value in (
            ("ocr_invoke_url", ocr_invoke_url),
            ("vietnamese_ocr_invoke_url", vietnamese_ocr_invoke_url),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise ValueError(
            "pipeline-ppocrv6 requires Option 2 endpoints: " + ", ".join(missing)
        )

    _LOGGER.info(
        "Option 2 endpoints: algorithmic_line_split=true "
        "ignored_legacy_line_detector=%s vietnamese=%s",
        bool(str(line_detector_invoke_url or "").strip()),
        bool(str(vietnamese_ocr_invoke_url or "").strip()),
    )

    batch_size = max(1, int(inference_batch_size or 1))
    timeout = max(1.0, float(request_timeout_s or 120.0))
    secret = ocr_api_key or api_key
    option2_nemotron_batch_size = max(
        1,
        int(os.environ.get("NEMO_RETRIEVER_OPTION2_NEMOTRON_BATCH_SIZE", "8") or 8),
    )
    nemotron_backend = make_nemotron_backend(
        str(ocr_invoke_url),
        api_key=secret,
        language=ocr_lang or "multi",
        timeout_s=timeout,
        # Option 2 is the only branch that opts into the wider Nemotron
        # micro-batch.  The NIM service is configured with the same limit;
        # keeping this local prevents the other OCR pipelines from changing
        # their request shape or VRAM profile.
        batch_size=max(option2_nemotron_batch_size, batch_size),
    )
    vietnamese_backend = make_vietnamese_recognizer(
        _option2_batch_ocr_url(str(vietnamese_ocr_invoke_url)),
        api_key=secret,
        timeout_s=timeout,
        # This endpoint uses VietOCR's native GPU batch path. Keep the
        # Nemotron transport at its NIM-safe batch size above, while
        # allowing a document batch to reach VietOCR in wider native GPU
        # chunks.  Batch-32 was measured against the production crop shape:
        # it preserved every response slot/text while beating 8/64/128 on
        # the single-GPU host.  The sidecar admission gate still limits the
        # number of simultaneous native predict_batch calls.
        batch_size=max(32, batch_size),
    )
    runner = Option2Pipeline(
        nemotron_backend,
        vietnamese_backend,
        config=Option2Config(
            language=ocr_lang or "auto",
            include_table_cells=bool(extract_tables),
            scan_page_fallback=bool(scan_ocr_fallback),
            batch_size=batch_size,
            request_timeout_s=timeout,
        ),
    )
    outputs = runner.process_pages([row.to_dict() for _, row in batch_df.iterrows()])
    return pd.DataFrame(
        [
            _apply_option2_output(
                row.to_dict(),
                output,
                extract_text=bool(extract_text),
                extract_tables=bool(extract_tables),
            )
            for (_, row), output in zip(batch_df.iterrows(), outputs)
        ]
    )


def _option2_batch_ocr_url(url: str) -> str:
    """Select the Option-2-only native batch route without changing Option 3."""

    normalized = url.rstrip("/")
    return normalized if normalized.endswith("/batch") else f"{normalized}/batch"


def _apply_option2_output(
    row: dict[str, Any],
    output: OCRPageOutput,
    *,
    extract_text: bool,
    extract_tables: bool,
) -> dict[str, Any]:
    """Restore Option 2 output using the shared dataframe contract."""

    metadata = row.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata.update(
        {
            "ocr_pipeline": OPTION2_SELECTOR,
            "ocr_pipeline_name": output.pipeline,
            "ocr_source": output.source,
            "ocr_model": output.model,
            "ocr_language": output.language,
            "ocr_status": output.status,
            "ocr_timing": dict(output.timing),
        }
    )
    if output.errors:
        metadata["ocr_errors"] = list(output.errors)
    row["metadata"] = metadata
    if output.status == "skipped" and output.source == "native_passthrough":
        row["ocr"] = _ocr_metadata(output, 0, 0)
        if extract_tables:
            row["table"] = list(output.tables)
        return row

    blocks = [
        block
        for block in output.ocr_text_blocks
        if isinstance(block, Mapping)
        and str(block.get("content_type") or "text") != "table_cell"
    ]
    cells = [
        block
        for block in output.ocr_text_blocks
        if isinstance(block, Mapping)
        and str(block.get("content_type") or "text") == "table_cell"
    ]
    row["_ocr_text_blocks"] = list(blocks)
    row["ocr_text_blocks"] = list(output.ocr_text_blocks)
    if extract_text:
        row["text"] = output.text.strip()
    if extract_tables:
        row["table"] = list(output.tables)
    row["ocr"] = _ocr_metadata(
        output,
        len(blocks),
        len(cells),
        blocks=blocks,
        cells=cells,
    )
    row["ocr_v1_num_detections"] = len(blocks) + len(cells)
    return row


def _route_option2_text(
    text: str,
    *,
    prior: Mapping[str, Any] | None,
    config: Option2Config,
    document_scoped: bool,
) -> NemotronLanguageDecision:
    """Use a decisive sampled document language before per-block langdetect."""

    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    # Codes, numbers, and symbols should retain Nemotron irrespective of the
    # document language; sending them to VietOCR only introduces corruption.
    if not compact or re.fullmatch(r"[\W_]*[A-Za-z0-9][A-Za-z0-9._:/+\-#]*[\W_]*", compact):
        return route_nemotron_text(
            compact,
            page_prior=prior,
            min_chars=int(config.language_min_chars),
            min_words=int(config.language_min_words),
        )
    if document_scoped and isinstance(prior, Mapping) and bool(prior.get("available")):
        vi = float(prior.get("vi") or 0.0)
        en = float(prior.get("en") or 0.0)
        threshold = float(config.document_language_confidence)
        if vi >= threshold and vi - en >= 0.20:
            return NemotronLanguageDecision(
                route=VIETNAMESE,
                confidence=vi,
                reason="document_sample_vietnamese_threshold",
                language_probabilities=dict(prior.get("probabilities") or {}),
                page_prior=prior,
                page_prior_used=True,
                raw_text=compact,
            )
        if en >= threshold and vi < 0.20:
            return NemotronLanguageDecision(
                route=ENGLISH,
                confidence=en,
                reason="document_sample_english_threshold",
                language_probabilities=dict(prior.get("probabilities") or {}),
                page_prior=prior,
                page_prior_used=True,
                raw_text=compact,
            )
    return route_nemotron_text(
        compact,
        page_prior=prior,
        min_chars=int(config.language_min_chars),
        min_words=int(config.language_min_words),
    )


def _candidate_from_nemotron(
    observation: _NemotronObservation,
    decision: NemotronLanguageDecision,
) -> OCRCandidate:
    route_language = "vi" if decision.route == VIETNAMESE else (
        "en" if decision.route == ENGLISH else observation.language
    )
    provenance = {
        "selected_backend": "nemotron",
        "route": decision.route,
        "reason": decision.reason,
        "language_confidence": decision.confidence,
        "language_router": decision.to_dict(),
        "page_prior": decision.page_prior,
        "raw_nemotron_text": observation.text[:240],
        "nemotron_original_text": observation.text,
        "vietnamese_candidate_text": None,
        "fallback_reason": None,
        "bbox_source": "nemotron_local" if not observation.bbox_fallback else "parent_semantic_unit",
        "bbox_fallback": observation.bbox_fallback,
        "nemotron": {
            "text": observation.text,
            "score": observation.score,
            "model": observation.model,
            "language": observation.language,
            "bbox_local": list(observation.local_bbox)
            if observation.local_bbox is not None
            else None,
        },
        "ocr_unit": {
            "unit_id": observation.unit.unit_id,
            "kind": observation.unit.kind,
            "source": observation.unit.source,
            "bbox_xyxy_norm": list(observation.unit.bbox_xyxy_norm),
            "crop_bbox_xyxy_norm": list(observation.unit.crop_bbox_xyxy_norm),
        },
        "title_priority": observation.unit.kind == "title",
    }
    return OCRCandidate(
        text=observation.text,
        bbox_xyxy_norm=observation.bbox_xyxy_norm,
        score=observation.score,
        source="option2_nemotron",
        model=observation.model,
        language=route_language,
        content_type=_content_type(observation.unit),
        reading_order=observation.unit.reading_order,
        unit_id=observation.unit.unit_id,
        table_id=observation.unit.table_id,
        cell_id=observation.unit.cell_id,
        provenance=provenance,
        candidates=[
            {
                "backend": "nemotron",
                "text": observation.text,
                "score": observation.score,
                "model": observation.model,
                "language": observation.language,
                "bbox_xyxy_norm": list(observation.bbox_xyxy_norm),
            }
        ],
    )


def _direct_vietnamese_state(
    unit: OCRUnit,
    prior: Mapping[str, Any] | None,
    backend: Any,
) -> _CandidateState:
    """Create a VietOCR candidate without a redundant Nemotron pass."""

    probabilities = dict(prior.get("probabilities") or {}) if isinstance(prior, Mapping) else {}
    confidence = (
        float(prior.get("vi"))
        if isinstance(prior, Mapping) and prior.get("vi") is not None
        else None
    )
    decision = NemotronLanguageDecision(
        route=VIETNAMESE,
        confidence=confidence,
        reason="document_preprobe_vietnamese_direct",
        language_probabilities=probabilities,
        page_prior=prior,
        page_prior_used=True,
        raw_text="",
    )
    model = str(getattr(backend, "model", "VietOCR"))
    provenance = {
        "selected_backend": "vietocr",
        "route": VIETNAMESE,
        "reason": decision.reason,
        "language_confidence": decision.confidence,
        "language_router": decision.to_dict(),
        "page_prior": decision.page_prior,
        "raw_nemotron_text": None,
        "nemotron_original_text": None,
        "vietnamese_candidate_text": None,
        "fallback_reason": None,
        "bbox_source": "parent_semantic_unit",
        "bbox_fallback": True,
        "nemotron_skipped": True,
        "ocr_unit": {
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "source": unit.source,
            "bbox_xyxy_norm": list(unit.bbox_xyxy_norm),
            "crop_bbox_xyxy_norm": list(unit.crop_bbox_xyxy_norm),
        },
        "title_priority": unit.kind == "title",
    }
    candidate = OCRCandidate(
        text="",
        bbox_xyxy_norm=unit.bbox_xyxy_norm,
        score=None,
        source="option2_vietnamese_recognizer",
        model=model,
        language="vi",
        content_type=_content_type(unit),
        reading_order=unit.reading_order,
        unit_id=unit.unit_id,
        table_id=unit.table_id,
        cell_id=unit.cell_id,
        provenance=provenance,
    )
    return _CandidateState(
        candidate=candidate,
        unit=unit,
        decision=decision,
        debug={
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "route": VIETNAMESE,
            "language_router": decision.to_dict(),
            "nemotron": {"skipped": True, "reason": "document_preprobe"},
            "selected_backend": "vietocr",
            "direct_vietnamese_route": True,
        },
    )


def _option2_is_safe_single_line(unit: OCRUnit) -> bool:
    """Allow direct VietOCR only for a geometry-safe single-line crop.

    Page Elements boxes are semantic regions, not guaranteed text lines.  A
    direct recognizer returns one string for one crop, while Nemotron can
    return several local text boxes from a multi-line crop.  Bypassing
    Nemotron for the latter would therefore reduce recall and destroy the
    output block structure.  The detector-derived local text-height estimate
    is the only scale used here, so this remains resolution-independent.

    Table cells are deliberately excluded: even a visually short cell can
    contain wrapped text, and Table Structure's cell geometry must retain the
    conservative Nemotron path until a cell-line benchmark proves otherwise.
    """

    if unit.kind not in {"text_block", "title", "table_cell"}:
        return False
    if not unit.crop_b64 or unit.metadata.get("scan_page_fallback"):
        return False
    bbox = clamp_bbox(unit.bbox_xyxy_norm)
    if bbox is None:
        return False
    try:
        local_height = float(unit.metadata.get("local_text_height_norm") or 0.0)
    except (TypeError, ValueError):
        return False
    if local_height <= 0.0:
        return False
    height = bbox[3] - bbox[1]
    if height <= 0.0:
        return False
    # When a detector returns an isolated tall region, the local-height
    # estimator quite reasonably falls back to that region itself.  A
    # normalized semantic height guard catches that case; it is deliberately
    # generous enough for large one-line headings, which the line detector
    # can still split into one crop without changing their text.
    if height > 0.05:
        return False
    # A 1.25x local-line-height ceiling is intentionally conservative.  It
    # tolerates detector padding/ascenders but rejects likely two-line boxes.
    return height / local_height <= 1.25


def _option2_parent_line_bbox(
    line_bbox: Sequence[float] | None,
    parent_bbox: Sequence[float],
) -> tuple[float, float, float, float] | None:
    """Use detector Y bounds while preserving the semantic parent's X span."""

    line = clamp_bbox(line_bbox)
    parent = clamp_bbox(parent_bbox)
    if line is None or parent is None:
        return None
    return (
        parent[0],
        max(parent[1], min(parent[3], line[1])),
        parent[2],
        max(parent[1], min(parent[3], line[3])),
    ) if line[3] > line[1] else None


def _option2_line_crop(
    page: OCRPage,
    parent: OCRUnit,
    line_bbox: Sequence[float],
    *,
    cropper: PageImageCropper | None = None,
    padding_scale: float = 1.7,
):
    """Crop one detected line with the same glyph-safe padding as text boxes.

    PP line boxes are often tight around the ink.  Cropping them directly can
    remove Vietnamese combining marks, ascenders, or descenders before the
    recognizer sees the image.  Reuse Option 2's adaptive padding policy here
    rather than applying a fixed pixel margin.  Overlapping parent/visual
    boxes remain soft blockers, so padding is not sacrificed just to avoid a
    duplicate detection.  A table-cell parent remains a hard outer boundary.
    """

    line = clamp_bbox(line_bbox)
    parent_bbox = clamp_bbox(parent.bbox_xyxy_norm)
    if line is None or parent_bbox is None:
        return None
    line_height = max(line[3] - line[1], 1e-9)
    safe, _desired = _option2_safe_padded_bbox(
        line,
        local_text_height=line_height,
        scale=max(1.0, float(padding_scale or 1.0)),
        image_shape_hw=_option2_image_shape(page, cropper),
        blockers=_option2_padding_blockers(page, anchor_bbox=line),
    )
    if parent.kind == "table_cell":
        # A line inside a cell may receive padding, but it must never absorb
        # text from the adjacent cell or the table's surrounding content.
        safe = clamp_bbox(
            (
                max(parent_bbox[0], safe[0]),
                max(parent_bbox[1], safe[1]),
                min(parent_bbox[2], safe[2]),
                min(parent_bbox[3], safe[3]),
            )
        ) or line
    if cropper is not None:
        return cropper.crop(safe)
    return crop_image_b64(page.image_b64, safe)


def _option2_fallback_line_unit(
    unit: OCRUnit,
    *,
    source: str = "option2_algorithmic_parent",
) -> OCRUnit:
    """Keep one direct VietOCR slot when projection cannot split a crop."""

    return OCRUnit(
        unit_id=unit.unit_id,
        kind=unit.kind,
        source=source,
        bbox_xyxy_norm=unit.bbox_xyxy_norm,
        crop_bbox_xyxy_norm=unit.crop_bbox_xyxy_norm,
        crop_b64=unit.crop_b64,
        crop_shape_hw=unit.crop_shape_hw,
        reading_order=unit.reading_order,
        detector_score=unit.detector_score,
        label=unit.label,
        table_id=unit.table_id,
        cell_id=unit.cell_id,
        metadata={
            **unit.metadata,
            "line_detector": None,
            "line_detector_fallback": False,
            "line_split_method": "horizontal_projection",
            "line_split_fallback": True,
            "parent_unit_id": unit.unit_id,
        },
    )


def _merge_option2_candidates(
    candidates: Sequence[OCRCandidate],
) -> list[OCRCandidate]:
    """Deduplicate safe same-unit and cross-unit OCR duplicates.

    A Nemotron crop may contain more than one recognition item.  When the
    backend omits local boxes, all of those items intentionally inherit the
    same parent bbox; same-unit merging therefore remains conservative.  A
    second cross-unit pass handles the actual repeated Page Elements boxes
    when both geometry and normalized text agree.
    """

    merged: list[OCRCandidate] = []
    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(merged)
                if _option2_duplicate(candidate, previous)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        previous = merged[duplicate_index]
        winner, loser = (
            (candidate, previous)
            if _option2_candidate_quality(candidate)
            > _option2_candidate_quality(previous)
            else (previous, candidate)
        )
        sources = list(
            dict.fromkeys(
                list(previous.provenance.get("sources", [previous.source]))
                + list(candidate.provenance.get("sources", [candidate.source]))
            )
        )
        winner.provenance = {
            **loser.provenance,
            **winner.provenance,
            "sources": sources,
            "merged_duplicate": True,
            "duplicate_text": loser.text,
            "duplicate_bbox_xyxy_norm": list(loser.bbox_xyxy_norm),
            "merged_unit_ids": sorted(
                set(
                    list(loser.provenance.get("merged_unit_ids") or [])
                    + list(winner.provenance.get("merged_unit_ids") or [])
                    + [str(loser.unit_id), str(winner.unit_id)]
                )
            ),
        }
        winner.candidates = list(previous.candidates) + list(candidate.candidates)
        merged[duplicate_index] = winner
    return merged


def _option2_duplicate(left: OCRCandidate, right: OCRCandidate) -> bool:
    if left.content_type == "table_cell" or right.content_type == "table_cell":
        if not (
            left.content_type == right.content_type == "table_cell"
            and left.table_id == right.table_id
            and left.cell_id == right.cell_id
        ):
            return False
    same_unit = left.unit_id == right.unit_id
    if not same_unit and left.content_type != right.content_type:
        return False
    overlap = bbox_iou(left.bbox_xyxy_norm, right.bbox_xyxy_norm)
    if overlap < (0.80 if same_unit else 0.78):
        return False
    similarity = text_similarity(left.text, right.text)
    if similarity < (0.90 if same_unit else 0.86):
        return False
    if same_unit:
        left_bbox = left.provenance.get("bbox_source")
        right_bbox = right.provenance.get("bbox_source")
        return left_bbox == "nemotron_local" and right_bbox == "nemotron_local"
    return True


def _option2_candidate_quality(candidate: OCRCandidate) -> tuple[int, float, int, int]:
    """Prefer Vietnamese direct OCR over a Nemotron duplicate."""

    selected = str(candidate.provenance.get("selected_backend") or "")
    source = str(candidate.source or "")
    backend_priority = 3 if selected in {"vietocr", "vietnamese_recognizer"} else 1
    if source in {"horizontal_projection", "option2_algorithmic_parent"}:
        backend_priority = max(backend_priority, 3)
    score = 0.0 if candidate.score is None else float(candidate.score)
    return (
        backend_priority,
        score,
        len(str(candidate.text or "").strip()),
        -int(candidate.reading_order),
    )


def _apply_vietnamese_result(
    state: _CandidateState,
    response: Any,
    backend: Any,
    *,
    config: Option2Config,
) -> tuple[bool, str]:
    if isinstance(response, Mapping) and response.get("error"):
        return False, "vietnamese_backend_error_response"
    items = recognition_items(response)
    if not items:
        return False, "vietnamese_empty_output"
    item = items[0]
    text = _clean_candidate_text(item.text, table_cell=state.unit.kind == "table_cell")
    score = item.score
    accepted, reason = _vietnamese_quality_gate(
        text,
        score,
        state.decision,
        config=config,
        allow_scoreless_override=(
            config.allow_scoreless_line_vietnamese
            and state.unit.source in {
                "horizontal_projection",
                "option2_algorithmic_parent",
                "option2_algorithmic_split_disabled",
            }
        ),
    )
    state.debug["vietnamese"] = {
        "text": text,
        "score": score,
        "model": item.model or getattr(backend, "model", "VietOCR"),
        "language": item.language or "vi",
        "accepted": accepted,
        "reason": reason,
    }
    if not accepted:
        return False, reason
    backend_name = _vietnamese_backend_name(backend)
    state.candidate.text = text
    state.candidate.score = score
    state.candidate.model = str(item.model or getattr(backend, "model", backend_name))
    state.candidate.language = item.language or "vi"
    state.candidate.source = "option2_vietnamese_recognizer"
    state.candidate.provenance.update(
        {
            "selected_backend": backend_name,
            "vietnamese_candidate_text": text,
            "vietnamese": {
                "text": text,
                "score": score,
                "model": state.candidate.model,
                "language": state.candidate.language,
                "quality_gate": "accepted",
            },
            "fallback_reason": None,
        }
    )
    state.candidate.candidates.append(
        {
            "backend": backend_name,
            "text": text,
            "score": score,
            "model": state.candidate.model,
            "language": state.candidate.language,
        }
    )
    state.debug["selected_backend"] = backend_name
    return True, "accepted"


def _reject_vietnamese(
    state: _CandidateState,
    reason: str,
    *,
    error: Exception | None = None,
) -> None:
    direct_line_attempt = bool(
        state.debug.get("direct_vietnamese_route")
        or state.unit.source == "ppocrv6_line_detector"
    )
    state.candidate.provenance.update(
        {
            # A direct line route has no Nemotron observation to fall back to.
            # Keep the provenance truthful when VietOCR rejects/returns an
            # empty result; the generic semantic-unit route still records its
            # real Nemotron fallback below.
            "selected_backend": "vietocr" if direct_line_attempt else "nemotron",
            "fallback_reason": reason,
            "fallback_to_nemotron": not direct_line_attempt,
            "vietnamese_candidate_text": (
                state.debug.get("vietnamese", {}).get("text")
                if isinstance(state.debug.get("vietnamese"), Mapping)
                else None
            ),
        }
    )
    state.debug["selected_backend"] = "vietocr" if direct_line_attempt else "nemotron"
    state.debug["vietnamese_fallback"] = {
        "reason": reason,
        "error": str(error) if error else None,
    }


def _vietnamese_quality_gate(
    text: str,
    score: float | None,
    decision: NemotronLanguageDecision,
    *,
    config: Option2Config,
    allow_scoreless_override: bool = False,
) -> tuple[bool, str]:
    if not text.strip():
        return False, "vietnamese_empty_output"
    visible = [character for character in text if not character.isspace()]
    if not any(character.isalnum() for character in visible):
        return False, "vietnamese_no_letter_or_number"
    if any(
        character in {"�", "□"}
        or (ord(character) < 32 and character not in "\n\t\r")
        for character in visible
    ):
        return False, "vietnamese_replacement_or_control_character"
    if score is None:
        if not (config.allow_scoreless_vietnamese or allow_scoreless_override):
            return False, "vietnamese_score_missing"
        if decision.route != VIETNAMESE or (
            decision.confidence or 0.0
        ) < float(config.scoreless_language_confidence):
            return False, "vietnamese_scoreless_language_confidence_too_low"
        if _abnormal_or_prompt_like(text):
            return False, "vietnamese_scoreless_output_suspect"
        return True, (
            "accepted_scoreless_line_policy"
            if allow_scoreless_override and not config.allow_scoreless_vietnamese
            else "accepted_scoreless_server_policy"
        )
    try:
        if float(score) < float(config.vietnamese_score_threshold):
            return False, "vietnamese_score_below_threshold"
    except (TypeError, ValueError):
        return False, "vietnamese_score_invalid"
    return True, "accepted_score_threshold"


def _abnormal_or_prompt_like(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip().lower()
    if re.search(r"(.)\1{5,}", compact):
        return True
    return any(
        phrase in compact
        for phrase in (
            "as an ai",
            "language model",
            "prompt",
            "explanation:",
            "here is the",
        )
    )


def _vietnamese_crop(
    page: OCRPage,
    state: _CandidateState,
    *,
    cropper: PageImageCropper | None = None,
):
    padding = state.candidate.content_type != "table_cell"
    local_height = state.unit.metadata.get("local_text_height_norm")
    try:
        local_height = float(local_height) if local_height is not None else None
    except (TypeError, ValueError):
        local_height = None
    kwargs = {
        "local_text_height": (local_height * 1.7) if local_height and padding else local_height,
        "add_padding": padding,
    }
    if padding:
        blockers = _option2_padding_blockers(
            page,
            anchor_bbox=state.unit.bbox_xyxy_norm,
        )
        safe_bbox, _desired_bbox = _option2_safe_padded_bbox(
            state.candidate.bbox_xyxy_norm,
            local_text_height=local_height,
            scale=1.7,
            image_shape_hw=_option2_image_shape(page, cropper),
            blockers=blockers,
        )
        if cropper is not None:
            return cropper.crop(safe_bbox)
        return crop_image_b64(page.image_b64, safe_bbox)
    if cropper is not None:
        return cropper.crop(state.candidate.bbox_xyxy_norm, **kwargs)
    return crop_image_b64(page.image_b64, state.candidate.bbox_xyxy_norm, **kwargs)


_OPTION2_TEXT_LABELS = frozenset({"text", "title", "header_footer"})
_OPTION2_VISUAL_LABELS = frozenset({"image", "chart", "infographic", "stamp"})


def _option2_image_shape(
    page: OCRPage,
    cropper: PageImageCropper | None,
) -> tuple[int, int]:
    if cropper is not None:
        return (max(1, int(cropper.height)), max(1, int(cropper.width)))
    return image_size(page.image_b64) or (1, 1)


def _option2_same_bbox(left: Sequence[float], right: Sequence[float]) -> bool:
    # ``containment`` is intentionally not used here: it returns 1.0 when a
    # large chart/table contains a small text box, but that is not a duplicate
    # bbox and must remain a visible soft boundary candidate.
    return bbox_iou(left, right) >= 0.92


def _option2_padding_blockers(
    page: OCRPage,
    units: Sequence[OCRUnit] = (),
    *,
    current_unit: OCRUnit | None = None,
    anchor_bbox: Sequence[float] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Collect nearby geometry without treating overlaps as exclusions.

    Overlapping detections are intentionally retained: they may be duplicate
    views of the same glyphs, and shrinking one crop to avoid the other can
    remove Vietnamese combining marks.  The caller only uses clearly
    separated boxes as a soft padding boundary.
    """

    result: list[tuple[float, float, float, float]] = []
    anchor = clamp_bbox(anchor_bbox) if anchor_bbox is not None else None
    for unit in units:
        if current_unit is not None and unit is current_unit:
            continue
        box = clamp_bbox(unit.bbox_xyxy_norm)
        if box is not None and (anchor is None or not _option2_same_bbox(box, anchor)):
            result.append(box)

    payload = page.page_elements_v3
    detections = payload.get("detections") if isinstance(payload, Mapping) else []
    for detection in detections or []:
        if not isinstance(detection, Mapping):
            continue
        label = str(detection.get("label_name") or "").strip().lower()
        if label not in (_OPTION2_TEXT_LABELS | _OPTION2_VISUAL_LABELS):
            continue
        box = clamp_bbox(detection.get("bbox_xyxy_norm"))
        if box is not None and (anchor is None or not _option2_same_bbox(box, anchor)):
            result.append(box)

    # Table regions are hard semantic boundaries for generic text, but they
    # do not override the minimum glyph padding of an existing text crop.
    result.extend(_table_structure_boxes(page))
    return result


def _option2_safe_padded_bbox(
    bbox: Sequence[float],
    *,
    local_text_height: float | None,
    scale: float,
    image_shape_hw: Sequence[int],
    blockers: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Add glyph-safe padding and softly limit it only across clear gaps.

    The minimum margins are deliberately preserved.  A blocker that overlaps
    the current box (including a duplicate bbox) never shrinks the crop.  For
    separated boxes, a midpoint boundary is used only when it still leaves
    enough vertical/horizontal margin for ascenders, descenders, and combining
    marks.
    """

    current = clamp_bbox(bbox) or (0.0, 0.0, 1.0, 1.0)
    height, width = max(1, int(image_shape_hw[0])), max(1, int(image_shape_hw[1]))
    current_height = current[3] - current[1]
    current_width = current[2] - current[0]
    local_height = max(float(local_text_height or 0.0), current_height * 0.25, 1e-9)
    desired = expand_bbox_adaptive(
        current,
        local_text_height=local_height * max(1.0, float(scale or 1.0)),
        image_shape_hw=(height, width),
    )
    left, top, right, bottom = desired
    # Never allow a neighboring detector to remove the glyph-safe border.
    min_horizontal = max(local_height * 0.10, 1.0 / width)
    min_vertical = max(local_height * 0.30, 1.0 / height)
    base_x0, base_y0, base_x1, base_y1 = current
    base_center_x = (base_x0 + base_x1) / 2.0
    base_center_y = (base_y0 + base_y1) / 2.0

    for candidate in blockers:
        other = clamp_bbox(candidate)
        if other is None or _option2_same_bbox(current, other):
            continue
        ox0, oy0, ox1, oy1 = other
        other_height = oy1 - oy0
        other_width = ox1 - ox0
        x_overlap = max(0.0, min(base_x1, ox1) - max(base_x0, ox0))
        y_overlap = max(0.0, min(base_y1, oy1) - max(base_y0, oy0))

        # Overlapping/containing boxes are not a reason to throw away crop
        # pixels.  Keep the requested padding and let downstream canonical
        # merge handle duplicate recognition results.
        if x_overlap > 0.0 and y_overlap > 0.0:
            continue

        same_row = y_overlap > 0.0 or abs((oy0 + oy1) / 2.0 - base_center_y) <= max(
            other_height, current_height
        ) * 0.75
        same_column = x_overlap > 0.0 or abs((ox0 + ox1) / 2.0 - base_center_x) <= max(
            other_width, current_width
        ) * 0.75

        if same_row:
            if ox1 <= base_x0:
                gap_boundary = (ox1 + base_x0) / 2.0
                if base_x0 - gap_boundary >= min_horizontal:
                    left = max(left, gap_boundary)
            elif ox0 >= base_x1:
                gap_boundary = (ox0 + base_x1) / 2.0
                if gap_boundary - base_x1 >= min_horizontal:
                    right = min(right, gap_boundary)

        if same_column:
            if oy1 <= base_y0:
                gap_boundary = (oy1 + base_y0) / 2.0
                if base_y0 - gap_boundary >= min_vertical:
                    top = max(top, gap_boundary)
            elif oy0 >= base_y1:
                gap_boundary = (oy0 + base_y1) / 2.0
                if gap_boundary - base_y1 >= min_vertical:
                    bottom = min(bottom, gap_boundary)

    safe = clamp_bbox((left, top, right, bottom)) or current
    return safe, desired


def _apply_option2_text_crop_padding(
    page: OCRPage,
    units: Sequence[OCRUnit],
    config: Option2Config,
    *,
    cropper: PageImageCropper | None = None,
) -> None:
    """Expand only Option 2's text OCR crops without moving semantic boxes."""

    scale = max(1.0, float(config.text_crop_padding_scale or 1.0))
    if scale <= 1.0:
        return
    blockers = _option2_padding_blockers(page, units)
    for unit in units:
        if unit.kind not in {"text_block", "title"}:
            continue
        try:
            local_height = float(unit.metadata.get("local_text_height_norm") or 0.0)
        except (TypeError, ValueError):
            local_height = 0.0
        safe_bbox, desired_bbox = _option2_safe_padded_bbox(
            unit.bbox_xyxy_norm,
            local_text_height=local_height,
            scale=scale,
            image_shape_hw=_option2_image_shape(page, cropper),
            blockers=blockers,
        )
        crop = (
            cropper.crop(safe_bbox)
            if cropper is not None
            else crop_image_b64(page.image_b64, safe_bbox)
        )
        if crop is None:
            continue
        unit.crop_bbox_xyxy_norm = crop.bbox_xyxy_norm
        unit.crop_b64 = crop.image_b64
        unit.crop_shape_hw = crop.shape_hw
        unit.metadata["option2_text_crop_padding_scale"] = scale
        unit.metadata["option2_text_crop_padding_clipped"] = safe_bbox != desired_bbox


def _build_tables(
    page: OCRPage,
    candidates: Sequence[OCRCandidate],
) -> list[dict[str, Any]]:
    tables = table_payload(page)
    for table in tables:
        cells = [
            candidate.to_dict()
            for candidate in candidates
            if candidate.content_type == "table_cell"
            and candidate.table_id == table.get("table_id")
        ]
        cells.sort(
            key=lambda cell: (
                int(cell.get("reading_order") or 0),
                float((cell.get("bbox_xyxy_norm") or [0, 0, 0, 0])[1]),
                float((cell.get("bbox_xyxy_norm") or [0, 0, 0, 0])[0]),
            )
        )
        table["cells"] = cells
        table["text"] = _table_markdown(cells)
        table["table_text_format"] = "markdown"
    return tables


def _remove_table_text_overlaps(page: OCRPage, units: Sequence[OCRUnit]) -> list[OCRUnit]:
    """Keep Table Structure cells authoritative over generic text blocks."""

    table_boxes = _table_structure_boxes(page)
    if not table_boxes:
        return list(units)
    return [
        unit
        for unit in units
        if unit.kind == "table_cell"
        or not any(
            containment(unit.bbox_xyxy_norm, table_box) >= 0.55
            or bbox_iou(unit.bbox_xyxy_norm, table_box) >= 0.35
            for table_box in table_boxes
        )
    ]


def _deduplicate_option2_units(units: Sequence[OCRUnit]) -> list[OCRUnit]:
    """Remove exact/near-exact semantic boxes before any OCR call.

    Page Elements can expose the same visual line through overlapping
    semantic detections.  This pass is intentionally strict (IoU >= 0.92)
    so adjacent lines and columns remain independent.  Table cells are only
    merged when their table/cell identity is identical.
    """

    result: list[OCRUnit] = []
    for unit in units:
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(result)
                if _option2_unit_duplicate(unit, previous)
            ),
            None,
        )
        if duplicate_index is None:
            result.append(unit)
            continue
        previous = result[duplicate_index]
        winner, loser = (
            (unit, previous)
            if _option2_unit_quality(unit) > _option2_unit_quality(previous)
            else (previous, unit)
        )
        winner.metadata = {
            **loser.metadata,
            **winner.metadata,
            "merged_semantic_unit_ids": sorted(
                set(
                    list(loser.metadata.get("merged_semantic_unit_ids") or [])
                    + [loser.unit_id, winner.unit_id]
                )
            ),
        }
        result[duplicate_index] = winner
    return sorted(
        result,
        key=lambda unit: (
            int(unit.reading_order),
            unit.bbox_xyxy_norm[1],
            unit.bbox_xyxy_norm[0],
        ),
    )


def _option2_unit_duplicate(left: OCRUnit, right: OCRUnit) -> bool:
    if left.kind == "table_cell" or right.kind == "table_cell":
        return (
            left.kind == right.kind == "table_cell"
            and left.table_id == right.table_id
            and left.cell_id == right.cell_id
            and bbox_iou(left.bbox_xyxy_norm, right.bbox_xyxy_norm) >= 0.92
        )
    return bbox_iou(left.bbox_xyxy_norm, right.bbox_xyxy_norm) >= 0.92


def _option2_unit_quality(unit: OCRUnit) -> tuple[float, float, int]:
    score = unit.detector_score
    return (
        1.0 if score is None else float(score),
        1.0 if unit.kind == "title" else 0.0,
        -int(unit.reading_order),
    )


def _table_structure_boxes(page: OCRPage) -> list[tuple[float, float, float, float]]:
    payload = page.table_structure_v1
    if not isinstance(payload, Mapping):
        return []
    result: list[tuple[float, float, float, float]] = []
    for region in payload.get("regions") or []:
        if not isinstance(region, Mapping):
            continue
        table_bbox = clamp_bbox(region.get("bbox_xyxy_norm"))
        if table_bbox is not None:
            result.append(table_bbox)
        shape = region.get("orig_shape_hw") or (1, 1)
        try:
            shape_hw = (max(1, int(shape[0])), max(1, int(shape[1])))
        except (TypeError, ValueError, IndexError):
            shape_hw = (1, 1)
        for cell in region.get("detections") or []:
            if not isinstance(cell, Mapping) or str(cell.get("label_name") or "").lower() != "cell":
                continue
            mapped = map_local_bbox(
                cell.get("bbox_xyxy_norm") or cell.get("bbox"),
                table_bbox or (0.0, 0.0, 1.0, 1.0),
                shape_hw,
            )
            if mapped is not None:
                result.append(mapped)
    return result


def _table_markdown(cells: Sequence[Mapping[str, Any]]) -> str:
    if not cells:
        return ""
    rows: list[list[Mapping[str, Any]]] = []
    for cell in cells:
        bbox = cell.get("bbox_xyxy_norm") or [0, 0, 0, 0]
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        height = max(1e-6, float(bbox[3]) - float(bbox[1]))
        target = next(
            (
                row
                for row in rows
                if abs(center_y - _row_center(row)) <= max(height, _row_height(row)) * 0.6
            ),
            None,
        )
        if target is None:
            rows.append([cell])
        else:
            target.append(cell)
    for row in rows:
        row.sort(key=lambda item: float((item.get("bbox_xyxy_norm") or [0, 0, 0, 0])[0]))
    width = max(len(row) for row in rows)
    rendered = [
        "| "
        + " | ".join(str(cell.get("text") or "").replace("|", "/").strip() for cell in row)
        + " |"
        for row in rows
    ]
    rendered.insert(1, "| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(rendered)


def _row_center(row: Sequence[Mapping[str, Any]]) -> float:
    return sum(
        (
            float((item.get("bbox_xyxy_norm") or [0, 0, 0, 0])[1])
            + float((item.get("bbox_xyxy_norm") or [0, 0, 0, 0])[3])
        )
        / 2.0
        for item in row
    ) / max(1, len(row))


def _row_height(row: Sequence[Mapping[str, Any]]) -> float:
    values = [
        float((item.get("bbox_xyxy_norm") or [0, 0, 0, 0])[3])
        - float((item.get("bbox_xyxy_norm") or [0, 0, 0, 0])[1])
        for item in row
    ]
    return max(values, default=1e-6)


def _content_type(unit: OCRUnit) -> str:
    if unit.kind == "table_cell":
        return "table_cell"
    if unit.kind == "title":
        return "title"
    return "text"


def _clean_candidate_text(text: str, *, table_cell: bool) -> str:
    lines = []
    for raw in str(text or "").splitlines() or [str(text or "")]:
        line = " ".join(raw.split()).strip()
        if table_cell:
            line = line.strip("|").strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _vietnamese_backend_name(backend: Any) -> str:
    return str(getattr(backend, "backend", None) or "vietnamese_recognizer")


def _route_bucket(route: str) -> str:
    if route == VIETNAMESE:
        return "vietnamese"
    if route == ENGLISH:
        return "english"
    return "uncertain"


def _is_scan_page(page: OCRPage) -> bool:
    metadata = page.metadata
    if bool(metadata.get("needs_ocr_for_text") or metadata.get("needs_ocr")):
        return True
    if metadata.get("has_text") is False:
        return True
    return str(metadata.get("reader_backend") or "").lower() in {
        "scan",
        "image",
        "raster",
        "ocr",
    }


def _is_native_page(page: OCRPage) -> bool:
    metadata = page.metadata
    if bool(metadata.get("needs_ocr_for_text") or metadata.get("needs_ocr")):
        return False
    if str(metadata.get("reader_backend") or "").lower() in {
        "native_pdf",
        "native_spreadsheet",
        "openpyxl",
        "python_csv",
    }:
        return True
    return bool(metadata.get("has_text") is True and page.native_text)


def _native_output(page: OCRPage, elapsed: float) -> OCRPageOutput:
    return OCRPageOutput(
        pipeline=OPTION2_PIPELINE_NAME,
        text=page.native_text,
        source="native_passthrough",
        model="PDFium native text",
        language="native",
        timing=_timing(total_seconds=elapsed, unit_count=0, skipped_native=True),
        status="skipped",
    )


def _timing(
    *,
    total_seconds: float,
    unit_count: int,
    nemotron_seconds: float = 0.0,
    language_router_seconds: float = 0.0,
    vietnamese_recognizer_seconds: float = 0.0,
    nemotron_input_count: int = 0,
    vietnamese_input_count: int = 0,
    route_counts: Mapping[str, int] | None = None,
    fallback_count: int = 0,
    selected_backend_counts: Mapping[str, int] | None = None,
    nemotron_batch_count: int = 0,
    nemotron_request_count: int = 0,
    vietnamese_batch_count: int = 0,
    vietnamese_request_count: int = 0,
    canonical_block_count: int = 0,
    language_probe_seconds: float = 0.0,
    language_probe_input_count: int = 0,
    direct_vietnamese_input_count: int = 0,
    line_detector_seconds: float = 0.0,
    line_detector_input_count: int = 0,
    line_count: int = 0,
    line_split_seconds: float = 0.0,
    line_split_input_count: int = 0,
    skipped_native: bool = False,
) -> dict[str, Any]:
    return {
        "seconds": total_seconds,
        "total_seconds": total_seconds,
        "nemotron_seconds": nemotron_seconds,
        "language_router_seconds": language_router_seconds,
        "vietnamese_recognizer_seconds": vietnamese_recognizer_seconds,
        "nemotron_input_count": nemotron_input_count,
        "vietnamese_input_count": vietnamese_input_count,
        "route_counts": dict(
            route_counts
            or {"vietnamese": 0, "english": 0, "uncertain": 0}
        ),
        "fallback_count": int(fallback_count),
        "selected_backend_counts": dict(selected_backend_counts or {}),
        "nemotron_batch_count": nemotron_batch_count,
        "nemotron_request_count": nemotron_request_count,
        "vietnamese_batch_count": vietnamese_batch_count,
        "vietnamese_request_count": vietnamese_request_count,
        "unit_count": unit_count,
        "canonical_block_count": canonical_block_count,
        "language_probe_seconds": language_probe_seconds,
        "language_probe_input_count": int(language_probe_input_count),
        "direct_vietnamese_input_count": int(direct_vietnamese_input_count),
        "line_detector_seconds": float(line_detector_seconds),
        "line_detector_input_count": int(line_detector_input_count),
        "line_count": int(line_count),
        "line_split_seconds": float(line_split_seconds),
        "line_split_input_count": int(line_split_input_count),
        "skipped_native": skipped_native,
    }


def _ocr_metadata(
    output: OCRPageOutput,
    block_count: int,
    cell_count: int,
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
    cells: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "pipeline": OPTION2_SELECTOR,
        "pipeline_name": output.pipeline,
        "source": output.source,
        "model": output.model,
        "language": output.language,
        "score": output.score,
        "confidence": output.confidence,
        "status": output.status,
        "num_detections": block_count + cell_count,
        "candidates": list(output.candidates),
        "errors": list(output.errors),
        "timing": dict(output.timing),
        "output": {
            "bbox_xyxy_norm": list(output.bbox_xyxy_norm)
            if output.bbox_xyxy_norm
            else None,
            "block_count": block_count,
            "table_cell_count": cell_count,
            "candidate_blocks": list(blocks or []),
            "candidate_cells": list(cells or []),
        },
    }


def _error(stage: str, error: Any) -> dict[str, Any]:
    if isinstance(error, str):
        return {"stage": stage, "type": "RuntimeError", "message": error}
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}
