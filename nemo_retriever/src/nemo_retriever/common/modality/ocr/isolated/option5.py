# SPDX-License-Identifier: Apache-2.0

"""Option 5: Nemotron baseline with raw-text language routing to VietOCR.

Page Elements v3 and Table Structure v1 provide semantic OCR units.  Tall
text/title boxes are split with a bounded PP-OCRv6 line-detector batch and a
cheap CPU projection fallback before the line-oriented Vietnamese recognizer.
Nemotron OCR v2 probes the document and remains the authoritative selective
fallback after the Vietnamese quality gate.  The selector remains
``pipeline-option5`` for API compatibility while the internal pipeline name
describes the actual architecture.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.adapters import (
    HTTPDetectorBackend,
    OCRBackend,
    OCRDetectorBackend,
    detector_boxes,
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
    CroppedImage,
    PageImageCropper,
    bbox_area,
    bbox_iou,
    clamp_bbox,
    containment,
    crop_image_b64,
    map_local_bbox,
    text_similarity,
    union_bbox,
)
from nemo_retriever.common.modality.ocr.isolated.language_router import (
    ENGLISH,
    VIETNAMESE,
    NemotronLanguageDecision,
    _OPTION3_VIETNAMESE_UNICODE,
    detect_nemotron_page_prior,
    route_nemotron_text,
)
from nemo_retriever.common.modality.ocr.isolated.multiline import (
    multiline_detector_candidates,
    split_multiline_units,
)
from nemo_retriever.common.modality.ocr.isolated.units import (
    build_ocr_units,
    table_payload,
)

OPTION5_SELECTOR = "pipeline-option5"
OPTION5_PIPELINE_NAME = "option5_nemotron_language_routed_vietnamese_ocr"
# The detector is only used after a document-level Vietnamese decision and
# only for semantic boxes that look multi-line. One-line boxes, English
# documents, and table cells do not pay for this remote call. The detector
# service itself micro-batches to its GPU-safe limit (16 by default); 100 here
# is the client logical batch so a document does not create one HTTP request
# per page.
OPTION5_LINE_DETECTOR_ENABLED = True
OPTION5_LINE_DETECTOR_BATCH_SIZE = 100
OPTION5_LINE_DETECTOR_MAX_POOL_WORKERS = 1
# These are deliberately Option-5-only presets.  The existing pipelines keep
# their public ``inference_batch_size`` and Ray tuning values unchanged.  The
# historical constant name is retained because the graph uses it for the
# Page Elements/Table Structure detection stages.
OPTION5_DETECTOR_BATCH_SIZE = 100
# The bundled VietOCR service performs its own conservative width buckets;
# keeping the transport batch at 100 removes per-request HTTP/model warmup for
# document pages with many line crops.  The service still controls its GPU
# admission/concurrency independently.
OPTION5_OCR_BATCH_SIZE = 100
OPTION5_MAX_REQUEST_WORKERS = 1
OPTION5_LANGUAGE_SAMPLE_PAGES = 5
OPTION5_LANGUAGE_SAMPLE_UNITS_PER_PAGE = 3
OPTION5_MAX_MULTILINE_LINES = 64
# A line detector cannot recover text outside semantic Page Elements boxes.
# Option 5 therefore performs a bounded full-page detector recall pass only on
# sparse/scanned pages.  It reuses the same PP-OCRv6 sidecar and batches these
# images together with the existing multiline crops.
OPTION5_FULL_PAGE_RECALL_ENABLED = True
OPTION5_RECALL_MIN_PRIMARY_UNITS = 2
OPTION5_RECALL_MAX_PRIMARY_UNITS = 32
OPTION5_RECALL_MIN_DETECTOR_SCORE = 0.35
# Keep recall additive but bounded on long documents. Semantic Page Elements
# and the multiline detector already cover most lines; a very large cap makes
# a long scan pay for a second full-page OCR stream.
OPTION5_RECALL_MAX_NEW_UNITS_PER_PAGE = 32


def option5_vietnamese_endpoint(endpoint: str | None) -> str:
    """Prefer the native batch route exposed by the Option 5 VietOCR sidecar.

    Option 3 deliberately keeps its historical ``/v1/ocr`` endpoint.  The
    bundled VietOCR service exposes ``/v1/ocr/batch`` for the speed-optimised
    path, so only Option 5 upgrades the conventional endpoint spelling.  A
    caller that already supplies a custom path is left untouched.
    """

    value = str(endpoint or "").strip().rstrip("/")
    if value.endswith("/v1/ocr"):
        return f"{value}/batch"
    return value


def option5_line_detector_endpoint(endpoint: str | None) -> str:
    """Use the true batched PP-OCRv6 detector route when available."""

    value = str(endpoint or "").strip().rstrip("/")
    if value.endswith("/v1/detect"):
        return f"{value}-batch"
    return value


@dataclass(frozen=True)
class Option5Config:
    """Configuration for the semantic-crop, language-routed Option 5 branch."""

    language: str | None = "auto"
    skip_native_text: bool = True
    include_table_cells: bool = True
    # Optional Page Elements-only table fallback.  The default stays false so
    # Pipeline 5 keeps its existing Table Structure cell behavior unchanged.
    include_page_element_table_regions: bool = False
    scan_page_fallback: bool = True
    batch_size: int = OPTION5_OCR_BATCH_SIZE
    # Option 5 keeps page outputs in input order while allowing a bounded
    # number of page-local Nemotron/VietOCR calls to overlap.  The single-GPU
    # development preset sets the same limit at the VietOCR sidecar.
    page_concurrency: int = 4
    request_timeout_s: float = 120.0
    vietnamese_score_threshold: float = 0.80
    allow_scoreless_vietnamese: bool = False
    scoreless_language_confidence: float = 0.90
    language_min_chars: int = 24
    language_min_words: int = 4
    language_sample_pages: int = OPTION5_LANGUAGE_SAMPLE_PAGES
    language_sample_units_per_page: int = OPTION5_LANGUAGE_SAMPLE_UNITS_PER_PAGE
    max_request_workers: int = OPTION5_MAX_REQUEST_WORKERS
    # A high-confidence Vietnamese document can skip the expensive full
    # Nemotron pass.  Nemotron remains the probe and selective fallback.
    direct_vietnamese: bool = True
    direct_language_confidence: float = 0.85
    split_multiline: bool = True
    multiline_height_ratio: float = 1.65
    # A full-page semantic text box can legitimately contain a few dozen
    # visual lines.  Capping this at 16 silently sends the whole paragraph
    # back to Nemotron, which is both slower and less accurate for Vietnamese.
    # The OCR request is still one bounded batch (normally <= 64 crops), not
    # 64 independent HTTP requests.
    max_multiline_lines: int = OPTION5_MAX_MULTILINE_LINES
    # The endpoint remains optional for local/test deployments. When it is
    # configured, the runtime supplies the batched detector backend; callers
    # can still turn the detector off explicitly and retain projection split.
    line_detection: bool = True
    # Full-page recall is selective: it is considered only for sparse pages
    # where Page Elements is most likely to have missed short lines.  The
    # normal semantic-crop path remains unchanged for dense/native pages.
    full_page_recall: bool = OPTION5_FULL_PAGE_RECALL_ENABLED
    recall_min_primary_units: int = OPTION5_RECALL_MIN_PRIMARY_UNITS
    recall_max_primary_units: int = OPTION5_RECALL_MAX_PRIMARY_UNITS
    recall_min_detector_score: float = OPTION5_RECALL_MIN_DETECTOR_SCORE
    recall_max_new_units_per_page: int = OPTION5_RECALL_MAX_NEW_UNITS_PER_PAGE


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


class Option5Pipeline:
    """Run one page's semantic crops through Nemotron, then VietOCR in batch."""

    pipeline_name = OPTION5_PIPELINE_NAME

    def __init__(
        self,
        nemotron: OCRBackend | Any,
        vietnamese_recognizer: VietnameseRecognizerBackend | Any,
        *,
        config: Option5Config | None = None,
        line_detector: OCRDetectorBackend | Any | None = None,
    ) -> None:
        if not hasattr(nemotron, "recognize"):
            raise TypeError("nemotron backend must expose recognize(images)")
        if not hasattr(vietnamese_recognizer, "recognize"):
            raise TypeError(
                "vietnamese_recognizer backend must expose recognize(images)"
            )
        if line_detector is not None and not hasattr(line_detector, "detect"):
            raise TypeError("line_detector backend must expose detect(images)")
        self.config = config or Option5Config()
        self.nemotron = nemotron
        self.vietnamese_recognizer = vietnamese_recognizer
        self.line_detector = (
            line_detector
            if OPTION5_LINE_DETECTOR_ENABLED and self.config.line_detection
            else None
        )
        self.last_document_diagnostics: dict[str, Any] = {}

    def process_page(self, page: OCRPage | Mapping[str, Any] | Any) -> OCRPageOutput:
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

        cropper = _page_cropper(normalized_page)
        units = build_ocr_units(
            normalized_page,
            include_table_cells=self.config.include_table_cells,
            # Visual regions are deliberately not OCR units in Option 5.
            include_visual_regions=False,
            # Padding normal text crops is useful; table cells stay inside
            # their structure box so adjacent cells cannot be duplicated.
            pad_table_cells=False,
            cropper=cropper,
        )
        units = _remove_table_text_overlaps(normalized_page, units)
        units = _suppress_nested_text_units(units)
        if (
            self.config.scan_page_fallback
            and _is_scan_page(normalized_page)
            and not units
        ):
            fallback = (
                cropper.crop((0.0, 0.0, 1.0, 1.0))
                if cropper is not None
                else crop_image_b64(
                    normalized_page.image_b64,
                    (0.0, 0.0, 1.0, 1.0),
                )
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
        nemotron_request_count = _backend_request_count(self.nemotron, len(units))
        valid_observations = [
            observation
            for observation in observations
            if observation.error is None and observation.text.strip()
        ]

        router_started = time.perf_counter()
        page_prior = detect_nemotron_page_prior(
            "\n".join(observation.text for observation in valid_observations),
            min_chars=int(self.config.language_min_chars),
            min_words=int(self.config.language_min_words),
        )
        states: list[_CandidateState] = []
        for observation in valid_observations:
            decision = route_nemotron_text(
                observation.text,
                page_prior=page_prior,
                min_chars=int(self.config.language_min_chars),
                min_words=int(self.config.language_min_words),
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
        vietnamese_request_count = 0
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
            vietnamese_request_count = _backend_request_count(
                self.vietnamese_recognizer,
                len(vietnamese_states),
            )
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

        canonical = _merge_option5_candidates([state.candidate for state in states])
        canonical.sort(
            key=lambda candidate: (
                int(candidate.reading_order),
                candidate.bbox_xyxy_norm[1],
                candidate.bbox_xyxy_norm[0],
            )
        )
        blocks = [candidate.to_dict() for candidate in canonical]
        tables = _build_tables(
            normalized_page,
            canonical,
            include_page_element_table_regions=bool(
                self.config.include_page_element_table_regions
            ),
        )
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
            nemotron_input_count=len(units),
            vietnamese_input_count=vietnamese_input_count,
            route_counts={
                "vietnamese": int(route_counts.get("vietnamese", 0)),
                "english": int(route_counts.get("english", 0)),
                "uncertain": int(route_counts.get("uncertain", 0)),
            },
            fallback_count=fallback_count,
            selected_backend_counts=dict(selected_backends),
            nemotron_batch_count=1,
            nemotron_request_count=nemotron_request_count,
            vietnamese_batch_count=vietnamese_batch_count,
            vietnamese_request_count=vietnamese_request_count,
            canonical_block_count=len(canonical),
        )
        if page_prior is not None:
            timing["page_language_prior"] = dict(page_prior)
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

    def process_document(
        self,
        pages: Sequence[OCRPage | Mapping[str, Any] | Any],
        *,
        document_key: str | None = None,
    ) -> list[OCRPageOutput]:
        """Process one document with a probe-first, language-aware plan.

        A high-confidence Vietnamese document takes the fast path: Nemotron
        only probes the sample and handles selective VietOCR failures.  Other
        documents retain the conservative Nemotron baseline and route only the
        candidates that are actually Vietnamese to VietOCR.
        """

        page_list = list(pages)
        if not page_list:
            self.last_document_diagnostics = {}
            return []

        started = time.perf_counter()
        errors: list[dict[str, Any]] = []
        line_split_stats: dict[str, Any] = {
            "line_detector_seconds": 0.0,
            "line_detector_input_count": 0,
            "line_detector_line_count": 0,
            "full_page_recall_pages": 0,
            "full_page_recall_input_count": 0,
            "full_page_recall_new_unit_count": 0,
        }
        normalized_pages = [page_value(page) for page in page_list]
        outputs: list[OCRPageOutput | None] = [None] * len(normalized_pages)
        contexts: list[tuple[int, OCRPage, list[OCRUnit], PageImageCropper | None]] = []
        all_units: list[OCRUnit] = []
        page_index_by_unit: dict[int, int] = {}
        line_split_unit_count = 0

        for page_index, page in enumerate(normalized_pages):
            if self.config.skip_native_text and _is_native_page(page):
                outputs[page_index] = _native_output(page, 0.0)
                continue
            if not page.image_b64:
                outputs[page_index] = OCRPageOutput(
                    pipeline=self.pipeline_name,
                    source=self.pipeline_name,
                    model="Nemotron OCR v2",
                    language=self.config.language,
                    errors=[_error("input", "page image is unavailable")],
                    status="failed",
                )
                continue
            cropper = _page_cropper(page)
            units = self._build_units_for_page(page, cropper=cropper)
            if not units:
                outputs[page_index] = OCRPageOutput(
                    pipeline=self.pipeline_name,
                    source=self.pipeline_name,
                    model="Nemotron OCR v2",
                    language=self.config.language,
                    status="completed",
                )
                continue
            contexts.append((page_index, page, units, cropper))
            all_units.extend(units)
            for unit in units:
                page_index_by_unit[id(unit)] = page_index

        if all_units:
            page_by_index = {
                page_index: (page, cropper)
                for page_index, page, _units, cropper in contexts
            }

            # Probe first.  Only the selected sample goes to Nemotron before
            # we know whether the document can take the fast Vietnamese path.
            probe_started = time.perf_counter()
            probe_units, probe_pages = _select_document_probe_units(
                contexts,
                document_key=document_key,
                max_pages=max(1, int(self.config.language_sample_pages)),
                max_units_per_page=max(
                    1, int(self.config.language_sample_units_per_page)
                ),
            )
            probe_observations = self._recognize_nemotron(probe_units, errors)
            valid_probe_observations = [
                observation
                for observation in probe_observations
                if observation.error is None and observation.text.strip()
            ]
            document_language, document_prior, language_probe = _infer_document_language(
                valid_probe_observations,
                language_min_chars=int(self.config.language_min_chars),
                language_min_words=int(self.config.language_min_words),
            )
            language_probe["strategy"] = "probe_first"
            direct_vietnamese = _should_direct_vietnamese(
                self.config,
                document_language=document_language,
                document_prior=document_prior,
                probe_observation_count=len(valid_probe_observations),
            )
            language_probe["direct_vietnamese"] = direct_vietnamese
            language_probe_seconds = time.perf_counter() - probe_started

            routing_prior = None if document_language == "mixed" else document_prior
            states_by_page: dict[int, list[_CandidateState]] = defaultdict(list)
            nemotron_seconds = language_probe_seconds
            nemotron_input_count = len(probe_units)
            nemotron_request_count = _backend_request_count(
                self.nemotron, len(probe_units)
            )
            nemotron_logical_batches = 1 if probe_units else 0
            valid_observations: list[_NemotronObservation]
            language_router_seconds = 0.0
            vietnamese_fallback_count = 0
            vietnamese_input_count = 0
            vietnamese_seconds = 0.0
            vietnamese_batch_count = 0
            vietnamese_request_count = 0

            if direct_vietnamese:
                # The document is now known to be Vietnamese. Collect every
                # tall text/title crop across all pages first, then issue one
                # bounded PP-OCRv6 batch. The projection splitter remains the
                # per-unit fallback when the detector is unavailable or
                # returns no usable lines.
                detector_responses: dict[int, Any] = {}
                full_page_recall_responses: dict[int, Any] = {}
                if self.config.split_multiline:
                    detector_responses = self._detect_multiline_responses(
                        contexts,
                        stats=line_split_stats,
                        errors=errors,
                        recall_responses=full_page_recall_responses,
                    )
                    expanded_contexts: list[
                        tuple[int, OCRPage, list[OCRUnit], PageImageCropper | None]
                    ] = []
                    expanded_all_units: list[OCRUnit] = []
                    page_index_by_unit.clear()
                    for page_index, page, units, cropper in contexts:
                        expanded_units = split_multiline_units(
                            page,
                            units,
                            cropper=cropper,
                            min_height_ratio=float(self.config.multiline_height_ratio),
                            max_lines=max(2, int(self.config.max_multiline_lines)),
                            line_detector=None,
                            detector_responses=detector_responses,
                            stats=line_split_stats,
                            errors=errors,
                        )
                        line_split_unit_count += max(
                            0, len(expanded_units) - len(units)
                        )
                        expanded_contexts.append(
                            (page_index, page, expanded_units, cropper)
                        )
                        expanded_all_units.extend(expanded_units)
                        for unit in expanded_units:
                            page_index_by_unit[id(unit)] = page_index
                    contexts = expanded_contexts
                    all_units = expanded_all_units

                    # The semantic detector remains the primary source of
                    # layout boxes.  Only add line boxes that the full-page
                    # recall detector found outside those boxes; this keeps
                    # the fast path stable and avoids duplicate OCR crops.
                    contexts, recall_unit_count = self._append_full_page_recall_units(
                        contexts,
                        full_page_recall_responses,
                        stats=line_split_stats,
                    )
                    line_split_stats["full_page_recall_new_unit_count"] = int(
                        line_split_stats.get("full_page_recall_new_unit_count") or 0
                    ) + recall_unit_count
                    all_units = [
                        unit
                        for _page_index, _page, units, _cropper in contexts
                        for unit in units
                    ]
                    page_index_by_unit.clear()
                    for page_index, _page, units, _cropper in contexts:
                        for unit in units:
                            page_index_by_unit[id(unit)] = page_index

                # Vietnamese fast path: trust the document-level decision for
                # normal text units.  This is the critical change that avoids
                # paying for a full Nemotron OCR pass before VietOCR.
                for page_index, _page, units, _cropper in contexts:
                    for unit in units:
                        states_by_page[page_index].append(
                            _direct_vietnamese_state(
                                unit,
                                document_language=document_language,
                                document_prior=document_prior,
                            )
                        )
                state_pages = [
                    (state, page_by_index[page_index][0], page_by_index[page_index][1])
                    for page_index, states in states_by_page.items()
                    for state in states
                ]
                (
                    vietnamese_fallback_count,
                    vietnamese_input_count,
                    vietnamese_seconds,
                    vietnamese_batch_count,
                    vietnamese_request_count,
                ) = self._run_vietnamese_batch(state_pages, errors)

                failed_vietnamese_states = [
                    state
                    for states in states_by_page.values()
                    for state in states
                    if state.debug.get("selected_backend")
                    != _vietnamese_backend_name(self.vietnamese_recognizer)
                ]
                if failed_vietnamese_states:
                    fallback_started = time.perf_counter()
                    fallback_observations = self._recognize_nemotron(
                        [state.unit for state in failed_vietnamese_states],
                        errors,
                    )
                    nemotron_input_count += len(failed_vietnamese_states)
                    nemotron_seconds += time.perf_counter() - fallback_started
                    nemotron_request_count += _backend_request_count(
                        self.nemotron, len(failed_vietnamese_states)
                    )
                    nemotron_logical_batches += 1
                    _apply_direct_nemotron_fallback(
                        failed_vietnamese_states,
                        fallback_observations,
                        document_language=document_language,
                        document_prior=document_prior,
                    )
            else:
                # Conservative path for English, mixed, unknown, or a weak
                # probe.  Reuse probe observations and OCR only the remaining
                # units, so probe crops are never called twice.
                probe_ids = {id(unit) for unit in probe_units}
                remaining_units = [
                    unit for unit in all_units if id(unit) not in probe_ids
                ]
                full_started = time.perf_counter()
                full_observations = self._recognize_nemotron(remaining_units, errors)
                nemotron_input_count += len(remaining_units)
                nemotron_seconds += time.perf_counter() - full_started
                nemotron_request_count += _backend_request_count(
                    self.nemotron, len(remaining_units)
                )
                if remaining_units:
                    nemotron_logical_batches += 1
                valid_observations = [
                    observation
                    for observation in [*probe_observations, *full_observations]
                    if observation.error is None and observation.text.strip()
                ]

                router_started = time.perf_counter()
                for observation in valid_observations:
                    decision = route_nemotron_text(
                        observation.text,
                        page_prior=routing_prior,
                        min_chars=int(self.config.language_min_chars),
                        min_words=int(self.config.language_min_words),
                    )
                    # A strong document prior should rescue long Vietnamese
                    # boxes whose Nemotron text lost its diacritics.  Codes,
                    # numbers, and mixed documents remain conservative.
                    decision = _apply_document_route_prior(
                        decision,
                        observation.text,
                        document_language=document_language,
                        document_prior=document_prior,
                    )
                    candidate = _candidate_from_nemotron(observation, decision)
                    candidate.provenance.update(
                        {
                            "language_scope": "document",
                            "document_language": document_language,
                        }
                    )
                    state = _CandidateState(
                        candidate=candidate,
                        unit=observation.unit,
                        decision=decision,
                        debug={
                            "unit_id": observation.unit.unit_id,
                            "kind": observation.unit.kind,
                            "route": decision.route,
                            "language_scope": "document",
                            "document_language": document_language,
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
                    page_index = page_index_by_unit.get(id(observation.unit))
                    if page_index is not None:
                        states_by_page[page_index].append(state)
                language_router_seconds = time.perf_counter() - router_started

                state_pages = [
                    (state, page_by_index[page_index][0], page_by_index[page_index][1])
                    for page_index, states in states_by_page.items()
                    for state in states
                ]
                (
                    vietnamese_fallback_count,
                    vietnamese_input_count,
                    vietnamese_seconds,
                    vietnamese_batch_count,
                    vietnamese_request_count,
                ) = self._run_vietnamese_batch(state_pages, errors)

            route_counts: Counter[str] = Counter(
                _route_bucket(state.decision.route)
                for states in states_by_page.values()
                for state in states
            )
            valid_observation_count = len(valid_probe_observations)
            if not direct_vietnamese:
                valid_observation_count = len(valid_observations)
            diagnostics = _document_diagnostics(
                language=document_language,
                document_key=document_key,
                page_count=len(normalized_pages),
                processed_page_count=len(contexts),
                unit_count=len(all_units),
                valid_observation_count=valid_observation_count,
                probe_pages=probe_pages,
                probe_unit_count=len(probe_units),
                language_probe=language_probe,
                language_probe_seconds=language_probe_seconds,
                nemotron_seconds=nemotron_seconds,
                nemotron_input_count=nemotron_input_count,
                language_router_seconds=language_router_seconds,
                vietnamese_seconds=vietnamese_seconds,
                route_counts=route_counts,
                nemotron_batch_count=nemotron_logical_batches,
                nemotron_request_count=nemotron_request_count,
                vietnamese_batch_count=vietnamese_batch_count,
                vietnamese_request_count=vietnamese_request_count,
                vietnamese_input_count=vietnamese_input_count,
                fallback_count=vietnamese_fallback_count,
                cache_hit_count=len(valid_probe_observations),
                direct_vietnamese=direct_vietnamese,
                line_split_unit_count=line_split_unit_count,
                line_detector_seconds=float(
                    line_split_stats.get("line_detector_seconds") or 0.0
                ),
                line_detector_input_count=int(
                    line_split_stats.get("line_detector_input_count") or 0
                ),
                line_detector_line_count=int(
                    line_split_stats.get("line_detector_line_count") or 0
                ),
                full_page_recall_pages=int(
                    line_split_stats.get("full_page_recall_pages") or 0
                ),
                full_page_recall_input_count=int(
                    line_split_stats.get("full_page_recall_input_count") or 0
                ),
                full_page_recall_new_unit_count=int(
                    line_split_stats.get("full_page_recall_new_unit_count") or 0
                ),
                total_seconds=time.perf_counter() - started,
                errors=errors,
            )

            for page_index, page, units, _cropper in contexts:
                page_states = states_by_page.get(page_index, [])
                page_fallback_count = sum(
                    1
                    for state in page_states
                    if state.decision.route == VIETNAMESE
                    and state.debug.get("selected_backend")
                    != _vietnamese_backend_name(self.vietnamese_recognizer)
                )
                outputs[page_index] = self._finalize_page_output(
                    page,
                    units,
                    page_states,
                    errors,
                    language=document_language,
                    total_seconds=time.perf_counter() - started,
                    nemotron_seconds=nemotron_seconds,
                    language_router_seconds=language_router_seconds,
                    vietnamese_seconds=vietnamese_seconds,
                    nemotron_request_count=nemotron_request_count,
                    vietnamese_request_count=vietnamese_request_count,
                    vietnamese_input_count=vietnamese_input_count,
                    vietnamese_batch_count=vietnamese_batch_count,
                    fallback_count=page_fallback_count,
                    document_diagnostics=diagnostics,
                )
        else:
            diagnostics = _document_diagnostics(
                language="unknown",
                document_key=document_key,
                page_count=len(normalized_pages),
                processed_page_count=0,
                unit_count=0,
                valid_observation_count=0,
                probe_pages=[],
                probe_unit_count=0,
                language_probe={},
                language_probe_seconds=0.0,
                nemotron_seconds=0.0,
                language_router_seconds=0.0,
                vietnamese_seconds=0.0,
                route_counts={},
                nemotron_batch_count=0,
                nemotron_request_count=0,
                vietnamese_batch_count=0,
                vietnamese_request_count=0,
                vietnamese_input_count=0,
                fallback_count=0,
                cache_hit_count=0,
                direct_vietnamese=False,
                line_split_unit_count=0,
                total_seconds=time.perf_counter() - started,
                errors=[],
            )

        for output in outputs:
            if output is None:
                output = OCRPageOutput(
                    pipeline=self.pipeline_name,
                    source=self.pipeline_name,
                    model="Nemotron OCR v2",
                    language=diagnostics["language"],
                    status="failed",
                )
            output.timing["document"] = dict(diagnostics)
            if output.source != "native_passthrough":
                output.language = diagnostics["language"]

        final_outputs = [output for output in outputs if output is not None]
        self.last_document_diagnostics = dict(diagnostics)
        return final_outputs

    def _build_units_for_page(
        self,
        page: OCRPage,
        *,
        cropper: PageImageCropper | None = None,
    ) -> list[OCRUnit]:
        units = build_ocr_units(
            page,
            include_table_cells=self.config.include_table_cells,
            include_page_element_table_regions=bool(
                self.config.include_page_element_table_regions
            ),
            include_visual_regions=False,
            pad_table_cells=False,
            cropper=cropper,
        )
        units = _remove_table_text_overlaps(page, units)
        units = _suppress_nested_text_units(units)
        if self.config.scan_page_fallback and _is_scan_page(page) and not units:
            fallback = crop_image_b64(page.image_b64, (0.0, 0.0, 1.0, 1.0))
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
        return units

    def _detect_multiline_responses(
        self,
        contexts: Sequence[
            tuple[int, OCRPage, list[OCRUnit], PageImageCropper | None]
        ],
        *,
        stats: dict[str, Any],
        errors: list[dict[str, Any]],
        recall_responses: dict[int, Any] | None = None,
    ) -> dict[int, Any]:
        """Run one document-level PP-OCRv6 detector batch.

        The old page loop made one HTTP call for every page. This method
        flattens likely multi-line candidates across the document and, for
        sparse/scanned pages, appends the page image as a selective recall
        request. Both request types share one logical batch (100), while the
        sidecar still controls its own GPU microbatch. Responses are mapped
        back to unit ids and page indexes separately.
        """

        detector = self.line_detector
        if (
            detector is None
            or not OPTION5_LINE_DETECTOR_ENABLED
            or not self.config.line_detection
        ):
            return {}
        candidates: list[OCRUnit] = []
        recall_pages: list[tuple[int, OCRPage]] = []
        for page_index, page, units, _cropper in contexts:
            candidates.extend(
                multiline_detector_candidates(
                    units,
                    min_height_ratio=float(self.config.multiline_height_ratio),
                )
            )
            if self._should_full_page_recall(page, units):
                recall_pages.append((page_index, page))
        if not candidates and not recall_pages:
            return {}

        started = time.perf_counter()
        stats["line_detector_input_count"] = int(
            stats.get("line_detector_input_count") or 0
        ) + len(candidates)
        stats["full_page_recall_pages"] = int(
            stats.get("full_page_recall_pages") or 0
        ) + len(recall_pages)
        stats["full_page_recall_input_count"] = int(
            stats.get("full_page_recall_input_count") or 0
        ) + len(recall_pages)
        request_images = [unit.crop_b64 for unit in candidates]
        request_images.extend(page.image_b64 for _page_index, page in recall_pages)
        try:
            responses = list(detector.detect(request_images))
        except Exception as exc:  # noqa: BLE001 - projection is safe fallback
            errors.append(_error("option5.line_detector", exc))
            responses = []
        stats["line_detector_seconds"] = float(
            stats.get("line_detector_seconds") or 0.0
        ) + (time.perf_counter() - started)
        semantic_responses = {
            id(unit): responses[index]
            for index, unit in enumerate(candidates)
            if index < len(responses)
        }
        if recall_responses is not None:
            # Keep recall responses separate from semantic-crop responses; the
            # page image has no parent crop geometry to map through.
            recall_responses.update(
                {
                    page_index: responses[len(candidates) + offset]
                    for offset, (page_index, _page) in enumerate(recall_pages)
                    if len(candidates) + offset < len(responses)
                }
            )
        return semantic_responses

    def _should_full_page_recall(
        self,
        page: OCRPage,
        units: Sequence[OCRUnit],
    ) -> bool:
        """Return whether this page is worth a cheap detector recall pass.

        The lower bound avoids changing the existing one-box test/scan
        fallback behavior. The upper bound protects long dense pages from an
        unnecessary whole-page detector call. Native pages are never sent to
        the recall detector because their text path is already authoritative.
        """

        if (
            not OPTION5_FULL_PAGE_RECALL_ENABLED
            or not bool(self.config.full_page_recall)
            or self.line_detector is None
            or not self.config.line_detection
            or not page.image_b64
            or _is_native_page(page)
        ):
            return False
        unit_count = len(
            [
                unit
                for unit in units
                if unit.kind in {"text_block", "title"}
                and not unit.metadata.get("scan_page_fallback")
            ]
        )
        minimum = max(0, int(self.config.recall_min_primary_units))
        maximum = max(minimum, int(self.config.recall_max_primary_units))
        return minimum <= unit_count <= maximum

    def _append_full_page_recall_units(
        self,
        contexts: Sequence[
            tuple[int, OCRPage, list[OCRUnit], PageImageCropper | None]
        ],
        responses: Mapping[int, Any],
        *,
        stats: dict[str, Any],
    ) -> tuple[list[tuple[int, OCRPage, list[OCRUnit], PageImageCropper | None]], int]:
        """Materialize only new full-page line boxes as VietOCR units."""

        if not responses:
            return list(contexts), 0
        result: list[tuple[int, OCRPage, list[OCRUnit], PageImageCropper | None]] = []
        total_added = 0
        for page_index, page, units, cropper in contexts:
            response = responses.get(page_index)
            if response is None:
                result.append((page_index, page, units, cropper))
                continue
            added = _full_page_recall_units(
                page,
                units,
                response,
                cropper=cropper,
                min_detector_score=float(self.config.recall_min_detector_score),
                max_new_units=max(1, int(self.config.recall_max_new_units_per_page)),
            )
            if added:
                # If a broad semantic parent yielded at least two reliable
                # full-page lines, replace that parent with line crops. This
                # avoids returning the first line twice while recovering the
                # short second line that Page Elements often misses.
                parent_counts: Counter[int] = Counter()
                for added_unit in added:
                    for existing_unit in units:
                        if (
                            existing_unit.kind == "table_cell"
                            or existing_unit.metadata.get("multiline_split")
                            or existing_unit.source == "ppocrv6_line_detector"
                        ):
                            continue
                        existing_bbox = clamp_bbox(existing_unit.bbox_xyxy_norm)
                        if existing_bbox is None:
                            continue
                        if containment(
                            added_unit.bbox_xyxy_norm,
                            existing_bbox,
                        ) < 0.80:
                            continue
                        if bbox_area(existing_bbox) <= bbox_area(
                            added_unit.bbox_xyxy_norm
                        ) * 2.5:
                            continue
                        parent_counts[id(existing_unit)] += 1
                replace_ids = {
                    unit_id
                    for unit_id, count in parent_counts.items()
                    if count >= 2
                }
                if replace_ids:
                    units = [unit for unit in units if id(unit) not in replace_ids]
                units = _remove_table_text_overlaps(page, [*units, *added])
                total_added += len(added)
            result.append((page_index, page, units, cropper))
        return result, total_added

    def _run_vietnamese_batch(
        self,
        state_pages: Sequence[
            tuple[_CandidateState, OCRPage, PageImageCropper | None]
        ],
        errors: list[dict[str, Any]],
    ) -> tuple[int, int, float, int, int]:
        vietnamese_states: list[_CandidateState] = []
        fallback_count = 0
        for state, page, cropper in state_pages:
            if state.decision.route != VIETNAMESE:
                continue
            crop = _vietnamese_crop(page, state, cropper=cropper)
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

        if not vietnamese_states:
            return fallback_count, 0, 0.0, 0, 0

        started = time.perf_counter()
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
        elapsed = time.perf_counter() - started
        for index, state in enumerate(vietnamese_states):
            fallback_count += 1
            if backend_error is not None:
                _reject_vietnamese(state, "vietnamese_backend_error", error=backend_error)
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
            except Exception as exc:  # noqa: BLE001 - malformed candidate fallback
                accepted = False
                reason = "vietnamese_response_parse_error"
                errors.append(_error("vietnamese_recognizer.parse", exc))
            if accepted:
                fallback_count -= 1
            else:
                _reject_vietnamese(state, reason)
        return (
            fallback_count,
            len(vietnamese_states),
            elapsed,
            1,
            _backend_request_count(
                self.vietnamese_recognizer,
                len(vietnamese_states),
            ),
        )

    def _finalize_page_output(
        self,
        page: OCRPage,
        units: Sequence[OCRUnit],
        states: Sequence[_CandidateState],
        errors: list[dict[str, Any]],
        *,
        language: str | None,
        total_seconds: float,
        nemotron_seconds: float,
        language_router_seconds: float,
        vietnamese_seconds: float,
        nemotron_request_count: int,
        vietnamese_request_count: int,
        vietnamese_input_count: int,
        vietnamese_batch_count: int,
        fallback_count: int,
        document_diagnostics: Mapping[str, Any] | None = None,
    ) -> OCRPageOutput:
        canonical = _merge_option5_candidates([state.candidate for state in states])
        canonical.sort(
            key=lambda candidate: (
                int(candidate.reading_order),
                candidate.bbox_xyxy_norm[1],
                candidate.bbox_xyxy_norm[0],
            )
        )
        blocks = [candidate.to_dict() for candidate in canonical]
        tables = _build_tables(
            page,
            canonical,
            include_page_element_table_regions=bool(
                self.config.include_page_element_table_regions
            ),
        )
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
        route_counts: Counter[str] = Counter(
            _route_bucket(state.decision.route) for state in states
        )
        timing = _timing(
            total_seconds=total_seconds,
            unit_count=len(units),
            nemotron_seconds=nemotron_seconds,
            language_router_seconds=language_router_seconds,
            vietnamese_recognizer_seconds=vietnamese_seconds,
            nemotron_input_count=len(units),
            vietnamese_input_count=vietnamese_input_count,
            route_counts={
                "vietnamese": int(route_counts.get("vietnamese", 0)),
                "english": int(route_counts.get("english", 0)),
                "uncertain": int(route_counts.get("uncertain", 0)),
            },
            fallback_count=fallback_count,
            selected_backend_counts=dict(selected_backends),
            nemotron_batch_count=1 if units else 0,
            nemotron_request_count=nemotron_request_count if units else 0,
            vietnamese_batch_count=vietnamese_batch_count,
            vietnamese_request_count=vietnamese_request_count,
            canonical_block_count=len(canonical),
        )
        if document_diagnostics is not None:
            timing["document"] = dict(document_diagnostics)
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
            language=language,
            tables=tables,
            candidates=debug_candidates,
            errors=list(errors),
            timing=timing,
            status=status,
        )

    def process_pages(
        self, pages: Sequence[OCRPage | Mapping[str, Any] | Any]
    ) -> list[OCRPageOutput]:
        """Process pages concurrently, returning outputs in input order.

        Page-local routing and one Vietnamese logical batch per page remain
        intentional.  Only the page jobs overlap; the bounded executor and
        the VietOCR sidecar admission gate keep the single GPU from seeing an
        unbounded request fan-out.
        """

        page_list = list(pages)
        if not page_list:
            return []
        worker_count = min(
            len(page_list),
            max(1, min(4, int(self.config.page_concurrency or 1))),
        )
        if worker_count == 1:
            return [self._process_page_safe(page) for page in page_list]
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="option5-page",
        ) as executor:
            # executor.map preserves input order even when page durations
            # differ, which keeps downstream dataframe/page numbering stable.
            return list(executor.map(self._process_page_safe, page_list))

    def _process_page_safe(
        self, page: OCRPage | Mapping[str, Any] | Any
    ) -> OCRPageOutput:
        try:
            return self.process_page(page)
        except Exception as exc:  # noqa: BLE001 - preserve page-local failure
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                source=self.pipeline_name,
                model="Nemotron OCR v2",
                language=self.config.language,
                errors=[_error("page", exc)],
                status="failed",
            )

    def _recognize_nemotron(
        self,
        units: Sequence[OCRUnit],
        errors: list[dict[str, Any]],
    ) -> list[_NemotronObservation]:
        """Send every semantic crop in one logical Nemotron batch."""

        try:
            responses = list(
                self.nemotron.recognize([unit.crop_b64 for unit in units])
            )
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


def run_option5_batch(
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
    # Kept in the public adapter signature for configuration compatibility.
    # Option 5 uses this endpoint only for candidate multiline boxes; the
    # other legacy recognizer endpoint remains unused.
    line_detector_invoke_url: str | None = None,
    ocr_recognizer_invoke_url: str | None = None,
    tesseract_ocr_invoke_url: str | None = None,
) -> Any:
    """Run Option 5 over a dataframe batch."""

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
            "pipeline-option5 requires the same endpoints as pipeline-option3: "
            + ", ".join(missing)
        )

    batch_size = max(OPTION5_OCR_BATCH_SIZE, int(inference_batch_size or 1))
    timeout = max(1.0, float(request_timeout_s or 120.0))
    secret = ocr_api_key or api_key
    line_detector = (
        HTTPDetectorBackend(
            endpoint=option5_line_detector_endpoint(line_detector_invoke_url),
            timeout_s=timeout,
            batch_size=OPTION5_LINE_DETECTOR_BATCH_SIZE,
            max_retries=1,
            max_429_retries=0,
            max_pool_workers=OPTION5_LINE_DETECTOR_MAX_POOL_WORKERS,
        )
        if OPTION5_LINE_DETECTOR_ENABLED
        and str(line_detector_invoke_url or "").strip()
        else None
    )
    runner = Option5Pipeline(
        make_nemotron_backend(
            str(ocr_invoke_url),
            api_key=secret,
            language=ocr_lang or "multi",
            timeout_s=timeout,
            batch_size=batch_size,
            max_pool_workers=OPTION5_MAX_REQUEST_WORKERS,
        ),
        make_vietnamese_recognizer(
            option5_vietnamese_endpoint(vietnamese_ocr_invoke_url),
            api_key=secret,
            timeout_s=timeout,
            batch_size=batch_size,
            max_pool_workers=OPTION5_MAX_REQUEST_WORKERS,
        ),
        config=Option5Config(
            language=ocr_lang or "auto",
            include_table_cells=bool(extract_tables),
            scan_page_fallback=bool(scan_ocr_fallback),
            batch_size=batch_size,
            request_timeout_s=timeout,
            max_request_workers=OPTION5_MAX_REQUEST_WORKERS,
            line_detection=line_detector is not None,
        ),
        line_detector=line_detector,
    )
    source_rows = [row.to_dict() for _, row in batch_df.iterrows()]
    outputs: list[OCRPageOutput | None] = [None] * len(source_rows)
    for document_key, entries in _group_document_rows(source_rows):
        indices = [index for index, _row in entries]
        try:
            document_outputs = runner.process_document(
                [source_rows[index] for index in indices],
                document_key=document_key,
            )
        except Exception as exc:  # noqa: BLE001 - preserve dataframe batch shape
            document_outputs = [
                OCRPageOutput(
                    pipeline=runner.pipeline_name,
                    source=runner.pipeline_name,
                    model="Nemotron OCR v2",
                    errors=[_error("document", exc)],
                    status="failed",
                )
                for _ in indices
            ]
        for index, output in zip(indices, document_outputs):
            outputs[index] = output
    outputs = [
        output
        if output is not None
        else OCRPageOutput(
            pipeline=runner.pipeline_name,
            source=runner.pipeline_name,
            model="Nemotron OCR v2",
            errors=[_error("document", "missing document output")],
            status="failed",
        )
        for output in outputs
    ]
    return pd.DataFrame(
        [
            _apply_option5_output(
                row.to_dict(),
                output,
                extract_text=bool(extract_text),
                extract_tables=bool(extract_tables),
            )
            for (_, row), output in zip(batch_df.iterrows(), outputs)
        ]
    )


def _group_document_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[tuple[int, Mapping[str, Any]]]]]:
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        key = _document_key_for_row(row, index)
        grouped.setdefault(key, []).append((index, row))
    return list(grouped.items())


def _document_key_for_row(row: Mapping[str, Any], index: int) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("source_path", "document_path", "document_id", "filename"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
    for key in ("source_path", "document_path", "path", "document_id", "filename"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    # A missing source key is unsafe to merge with another page.  The normal
    # PDF splitter always supplies metadata.source_path; this fallback keeps
    # ad-hoc image rows isolated rather than silently mixing documents.
    return f"unkeyed-page-{index}"


def _select_document_probe(
    contexts: Sequence[
        tuple[int, OCRPage, Sequence[OCRUnit], PageImageCropper | None]
    ],
    observations: Sequence[_NemotronObservation],
    *,
    document_key: str | None,
    max_pages: int,
    max_units_per_page: int,
) -> tuple[list[_NemotronObservation], list[int]]:
    """Select a deterministic-random five-page probe from reused observations."""

    observations_by_unit: dict[int, list[_NemotronObservation]] = defaultdict(list)
    for observation in observations:
        observations_by_unit[id(observation.unit)].append(observation)

    eligible: list[tuple[int, OCRPage, Sequence[OCRUnit]]] = []
    for context in contexts:
        if any(
            observations_by_unit.get(id(unit))
            for unit in context[2]
        ):
            eligible.append(context)
    if not eligible:
        return [], []

    seed_bytes = hashlib.sha256(
        str(document_key or "option5-document").encode("utf-8", "replace")
    ).digest()
    sampler = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    sample_count = min(max(1, int(max_pages)), len(eligible))
    selected_positions = sorted(sampler.sample(range(len(eligible)), sample_count))
    selected = [eligible[position] for position in selected_positions]
    probe: list[_NemotronObservation] = []
    probe_pages: list[int] = []
    for page_index, page, units, _cropper in selected:
        probe_pages.append(int(page.page_number or page_index + 1))
        ordered_units = sorted(
            units,
            key=lambda unit: (
                0 if unit.kind == "title" else (1 if unit.kind == "text_block" else 2),
                int(unit.reading_order),
                unit.bbox_xyxy_norm[1],
                unit.bbox_xyxy_norm[0],
            ),
        )
        taken = 0
        for unit in ordered_units:
            for observation in observations_by_unit.get(id(unit), []):
                if not observation.text.strip():
                    continue
                probe.append(observation)
                taken += 1
                if taken >= max(1, int(max_units_per_page)):
                    break
            if taken >= max(1, int(max_units_per_page)):
                break
    return probe, probe_pages


def _select_document_probe_units(
    contexts: Sequence[
        tuple[int, OCRPage, Sequence[OCRUnit], PageImageCropper | None]
    ],
    *,
    document_key: str | None,
    max_pages: int,
    max_units_per_page: int,
) -> tuple[list[OCRUnit], list[int]]:
    """Select probe units before full OCR so Vietnamese can short-circuit it."""

    eligible = [context for context in contexts if context[2]]
    if not eligible:
        return [], []
    seed_bytes = hashlib.sha256(
        str(document_key or "option5-document").encode("utf-8", "replace")
    ).digest()
    sampler = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    sample_count = min(max(1, int(max_pages)), len(eligible))
    selected_positions = sorted(sampler.sample(range(len(eligible)), sample_count))
    selected = [eligible[position] for position in selected_positions]
    probe_units: list[OCRUnit] = []
    probe_pages: list[int] = []
    for page_index, page, units, _cropper in selected:
        probe_pages.append(int(page.page_number or page_index + 1))
        ordered_units = sorted(
            units,
            key=lambda unit: (
                0 if unit.kind == "title" else (1 if unit.kind == "text_block" else 2),
                int(unit.reading_order),
                unit.bbox_xyxy_norm[1],
                unit.bbox_xyxy_norm[0],
            ),
        )
        probe_units.extend(ordered_units[: max(1, int(max_units_per_page))])
    return probe_units, probe_pages


def _should_direct_vietnamese(
    config: Option5Config,
    *,
    document_language: str,
    document_prior: Mapping[str, Any] | None,
    probe_observation_count: int,
) -> bool:
    """Return whether Option 5 can avoid a full Nemotron OCR pass."""

    if not bool(config.direct_vietnamese):
        return False
    forced = str(config.language or "").strip().lower()
    if forced in {"vi", "vie", "vietnamese"}:
        return True
    if document_language != "vietnamese" or probe_observation_count <= 0:
        return False
    prior = document_prior if isinstance(document_prior, Mapping) else {}
    confidence = float(prior.get("confidence") or prior.get("vi") or 0.0)
    return confidence >= float(config.direct_language_confidence)


def _apply_document_route_prior(
    decision: NemotronLanguageDecision,
    text: str,
    *,
    document_language: str,
    document_prior: Mapping[str, Any] | None,
) -> NemotronLanguageDecision:
    """Rescue long Vietnamese boxes when the OCR lost its diacritics."""

    if document_language != "vietnamese":
        return decision
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact or _is_numeric_or_code_text(compact):
        return decision
    prior = document_prior if isinstance(document_prior, Mapping) else {}
    confidence = float(prior.get("confidence") or prior.get("vi") or 0.0)
    if confidence < 0.80:
        return decision
    if decision.route == VIETNAMESE:
        return decision
    return NemotronLanguageDecision(
        route=VIETNAMESE,
        confidence=max(confidence, float(decision.confidence or 0.0)),
        reason="document_vietnamese_prior_override",
        language_probabilities=dict(decision.language_probabilities or {}),
        page_prior=document_prior,
        page_prior_used=True,
        raw_text=compact,
    )


def _direct_vietnamese_state(
    unit: OCRUnit,
    *,
    document_language: str,
    document_prior: Mapping[str, Any] | None,
) -> _CandidateState:
    prior = document_prior if isinstance(document_prior, Mapping) else {}
    confidence = float(prior.get("confidence") or prior.get("vi") or 0.95)
    decision = NemotronLanguageDecision(
        route=VIETNAMESE,
        confidence=max(0.85, min(1.0, confidence)),
        reason="document_vietnamese_direct_route",
        language_probabilities={"vi": max(0.85, min(1.0, confidence))},
        page_prior=document_prior,
        page_prior_used=True,
        raw_text="",
    )
    backend_name = "vietnamese_recognizer"
    candidate = OCRCandidate(
        text="",
        bbox_xyxy_norm=unit.bbox_xyxy_norm,
        score=None,
        source="option5_vietnamese_recognizer",
        model="VietOCR",
        language="vi",
        content_type=_content_type(unit),
        reading_order=unit.reading_order,
        unit_id=unit.unit_id,
        table_id=unit.table_id,
        cell_id=unit.cell_id,
        provenance={
            "selected_backend": backend_name,
            "route": VIETNAMESE,
            "reason": decision.reason,
            "language_scope": "document",
            "document_language": document_language,
            "language_confidence": decision.confidence,
            "page_prior": document_prior,
            "ocr_unit": {
                "unit_id": unit.unit_id,
                "kind": unit.kind,
                "source": unit.source,
                "bbox_xyxy_norm": list(unit.bbox_xyxy_norm),
                "crop_bbox_xyxy_norm": list(unit.crop_bbox_xyxy_norm),
            },
        },
    )
    if unit.metadata.get("multiline_split"):
        candidate.provenance["multiline_split"] = True
        candidate.provenance["parent_unit_id"] = unit.metadata.get("parent_unit_id")
        candidate.provenance["line_index"] = unit.metadata.get("line_index")
    if unit.metadata.get("full_page_recall"):
        candidate.provenance["full_page_recall"] = True
        candidate.provenance["line_detector"] = unit.metadata.get("line_detector")
        candidate.provenance["line_detector_score"] = unit.metadata.get(
            "line_detector_score"
        )
    return _CandidateState(
        candidate=candidate,
        unit=unit,
        decision=decision,
        debug={
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "route": VIETNAMESE,
            "language_scope": "document",
            "document_language": document_language,
            "language_router": decision.to_dict(),
            "selected_backend": backend_name,
            "direct_vietnamese": True,
            "multiline_split": bool(unit.metadata.get("multiline_split")),
            "full_page_recall": bool(unit.metadata.get("full_page_recall")),
        },
    )


def _apply_direct_nemotron_fallback(
    states: Sequence[_CandidateState],
    observations: Sequence[_NemotronObservation],
    *,
    document_language: str,
    document_prior: Mapping[str, Any] | None,
) -> None:
    """Fill only failed direct-Vietnamese units with Nemotron output."""

    by_unit: dict[int, list[_NemotronObservation]] = defaultdict(list)
    for observation in observations:
        if observation.error is None and observation.text.strip():
            by_unit[id(observation.unit)].append(observation)
    for state in states:
        candidates = by_unit.get(id(state.unit)) or []
        if not candidates:
            state.debug["selected_backend"] = "nemotron"
            state.debug["vietnamese_fallback"] = {
                "reason": "nemotron_fallback_empty"
            }
            continue
        observation = candidates[0]
        prior_confidence = (
            float(document_prior.get("confidence") or document_prior.get("vi") or 0.85)
            if isinstance(document_prior, Mapping)
            else 0.85
        )
        decision = NemotronLanguageDecision(
            route=VIETNAMESE,
            confidence=max(0.85, min(1.0, prior_confidence)),
            reason="direct_vietnamese_quality_fallback_to_nemotron",
            language_probabilities={"vi": 0.85},
            page_prior=document_prior,
            page_prior_used=True,
            raw_text=observation.text,
        )
        candidate = _candidate_from_nemotron(observation, decision)
        rejected_vietnamese = state.debug.get("vietnamese")
        candidate.provenance.update(
            {
                "language_scope": "document",
                "document_language": document_language,
                "fallback_reason": "vietnamese_quality_gate",
                "vietnamese_candidate_text": (
                    rejected_vietnamese.get("text")
                    if isinstance(rejected_vietnamese, Mapping)
                    else None
                ),
                "vietnamese_candidate": (
                    dict(rejected_vietnamese)
                    if isinstance(rejected_vietnamese, Mapping)
                    else None
                ),
            }
        )
        state.candidate = candidate
        state.decision = decision
        state.debug.update(
            {
                "route": VIETNAMESE,
                "language_router": decision.to_dict(),
                "selected_backend": "nemotron",
                "nemotron_fallback": {
                    "text": observation.text,
                    "score": observation.score,
                },
            }
        )


def _is_numeric_or_code_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact or not any(character.isalpha() for character in compact):
        return True
    words = re.findall(r"[A-Za-zÀ-ỹĐđ]+", compact)
    if len(words) >= 4:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._:/+\-#\s]+", compact)) and any(
        character.isdigit() or character in "-_/.:+#" for character in compact
    )


def _infer_document_language(
    observations: Sequence[_NemotronObservation],
    *,
    language_min_chars: int,
    language_min_words: int,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    texts = [str(observation.text or "").strip() for observation in observations]
    combined = "\n".join(text for text in texts if text)
    prior = detect_nemotron_page_prior(
        combined,
        min_chars=language_min_chars,
        min_words=language_min_words,
    )
    decisions = [
        route_nemotron_text(
            text,
            page_prior=None,
            min_chars=language_min_chars,
            min_words=language_min_words,
        )
        for text in texts
        if text
    ]
    vi_weight = sum(
        len(text)
        for text, decision in zip(
            (text for text in texts if text), decisions
        )
        if decision.route == VIETNAMESE
    )
    en_weight = sum(
        len(text)
        for text, decision in zip(
            (text for text in texts if text), decisions
        )
        if decision.route == ENGLISH
    )
    total_weight = max(1, vi_weight + en_weight)
    prior_vi = float((prior or {}).get("vi") or 0.0)
    prior_en = float((prior or {}).get("en") or 0.0)
    if (
        vi_weight > 0
        and en_weight > 0
        and vi_weight / total_weight >= 0.18
        and en_weight / total_weight >= 0.18
    ):
        language = "mixed"
    elif vi_weight > 0 and (en_weight == 0 or vi_weight >= en_weight * 1.15):
        language = "vietnamese"
    elif en_weight > 0 and (vi_weight == 0 or en_weight >= vi_weight * 1.15):
        language = "english"
    elif prior_vi >= 0.80 and prior_vi - prior_en >= 0.20:
        language = "vietnamese"
    elif prior_en >= 0.80 and prior_en >= prior_vi:
        language = "english"
    else:
        language = "unknown"

    prior_probabilities = (
        prior.get("probabilities")
        if isinstance(prior, Mapping)
        else None
    )
    prior_has_signal = bool(
        isinstance(prior_probabilities, Mapping)
        and any(
            isinstance(value, (int, float)) and float(value) > 0.0
            for value in prior_probabilities.values()
        )
    )
    if language in {"vietnamese", "english"} and not prior_has_signal:
        # Give short, unaccented units the same document-level context that a
        # long probe would have supplied, without affecting mixed documents.
        vi = 0.92 if language == "vietnamese" else 0.03
        en = 0.92 if language == "english" else 0.03
        prior = {
            "available": True,
            "source": "document_probe_vote",
            "probabilities": {"vi": vi, "en": en},
            "vi": vi,
            "en": en,
            "confidence": max(vi, en),
        }
    probe = {
        "sample_unit_count": len(observations),
        "sample_text_count": len(texts),
        "route_counts": dict(
            Counter(_route_bucket(decision.route) for decision in decisions)
        ),
        "weighted_vietnamese_chars": vi_weight,
        "weighted_english_chars": en_weight,
        "prior": dict(prior) if prior else None,
    }
    return language, prior, probe


def _document_diagnostics(
    *,
    language: str,
    document_key: str | None,
    page_count: int,
    processed_page_count: int,
    unit_count: int,
    valid_observation_count: int,
    probe_pages: Sequence[int],
    probe_unit_count: int,
    language_probe: Mapping[str, Any],
    language_probe_seconds: float,
    nemotron_seconds: float,
    language_router_seconds: float,
    vietnamese_seconds: float,
    route_counts: Mapping[str, int],
    nemotron_batch_count: int,
    nemotron_request_count: int,
    vietnamese_batch_count: int,
    vietnamese_request_count: int,
    vietnamese_input_count: int,
    fallback_count: int,
    cache_hit_count: int,
    total_seconds: float,
    errors: Sequence[Mapping[str, Any]],
    nemotron_input_count: int = 0,
    direct_vietnamese: bool = False,
    line_split_unit_count: int = 0,
    line_detector_seconds: float = 0.0,
    line_detector_input_count: int = 0,
    line_detector_line_count: int = 0,
    full_page_recall_pages: int = 0,
    full_page_recall_input_count: int = 0,
    full_page_recall_new_unit_count: int = 0,
) -> dict[str, Any]:
    return {
        "scope": "document",
        "document_key": str(document_key) if document_key else None,
        "language": language,
        "page_count": int(page_count),
        "processed_page_count": int(processed_page_count),
        "unit_count": int(unit_count),
        "valid_observation_count": int(valid_observation_count),
        "probe_pages": [int(page) for page in probe_pages],
        "probe_unit_count": int(probe_unit_count),
        "probe_page_limit": OPTION5_LANGUAGE_SAMPLE_PAGES,
        "probe_units_per_page_limit": OPTION5_LANGUAGE_SAMPLE_UNITS_PER_PAGE,
        "language_probe": dict(language_probe),
        "cache_hits": int(cache_hit_count),
        "nemotron_input_count": int(nemotron_input_count),
        "route_counts": {
            "vietnamese": int(route_counts.get("vietnamese", 0)),
            "english": int(route_counts.get("english", 0)),
            "uncertain": int(route_counts.get("uncertain", 0)),
        },
        "fallback_count": int(fallback_count),
        "nemotron_logical_batches": int(nemotron_batch_count),
        "nemotron_request_count": int(nemotron_request_count),
        "vietnamese_logical_batches": int(vietnamese_batch_count),
        "vietnamese_request_count": int(vietnamese_request_count),
        "vietnamese_input_count": int(vietnamese_input_count),
        "error_count": len(errors),
        "direct_vietnamese": bool(direct_vietnamese),
        "line_split_unit_count": int(line_split_unit_count),
        "line_detector_input_count": int(line_detector_input_count),
        "line_detector_line_count": int(line_detector_line_count),
        "full_page_recall_pages": int(full_page_recall_pages),
        "full_page_recall_input_count": int(full_page_recall_input_count),
        "full_page_recall_new_unit_count": int(full_page_recall_new_unit_count),
        "timing": {
            "total_seconds": float(total_seconds),
            "nemotron_seconds": float(nemotron_seconds),
            "language_probe_seconds": float(language_probe_seconds),
            "language_router_seconds": float(language_router_seconds),
            "vietnamese_recognizer_seconds": float(vietnamese_seconds),
            "line_detector_seconds": float(line_detector_seconds),
        },
    }


def _apply_option5_output(
    row: dict[str, Any],
    output: OCRPageOutput,
    *,
    extract_text: bool,
    extract_tables: bool,
) -> dict[str, Any]:
    """Restore Option 5 output using the shared dataframe contract."""

    metadata = row.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata.update(
        {
            "ocr_pipeline": OPTION5_SELECTOR,
            "ocr_pipeline_name": output.pipeline,
            "ocr_source": output.source,
            "ocr_model": output.model,
            "ocr_language": output.language,
            "ocr_status": output.status,
            "ocr_timing": dict(output.timing),
        }
    )
    document_diagnostics = output.timing.get("document")
    if isinstance(document_diagnostics, Mapping):
        metadata["ocr_document_diagnostics"] = dict(document_diagnostics)
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
        source="option5_nemotron",
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


def _merge_option5_candidates(
    candidates: Sequence[OCRCandidate],
) -> list[OCRCandidate]:
    """Deduplicate only safe same-unit duplicates for Option 5.

    A Nemotron crop may contain more than one recognition item.  When the
    backend omits local boxes, all of those items intentionally inherit the
    same parent bbox; the shared generic merger would mistake that geometry
    for a duplicate and drop a distinct line.  Conservative same-unit,
    same-local-box merging keeps output stable without sacrificing nearby
    lines.
    """

    merged: list[OCRCandidate] = []
    for candidate in candidates:
        # A failed direct recognizer can leave an empty placeholder when its
        # selective Nemotron fallback has no usable response.  Empty blocks
        # are not useful to downstream chunking and should never shadow a
        # valid neighboring line.
        if not str(candidate.text or "").strip():
            continue
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(merged)
                if _option5_duplicate(candidate, previous)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        previous = merged[duplicate_index]
        previous_score = previous.score if previous.score is not None else 0.0
        candidate_score = candidate.score if candidate.score is not None else 0.0
        winner, loser = (
            (candidate, previous)
            if candidate_score > previous_score
            else (previous, candidate)
        )
        sources = list(previous.provenance.get("sources", [previous.source]))
        if candidate.source not in sources:
            sources.append(candidate.source)
        winner.provenance = {
            **loser.provenance,
            **winner.provenance,
            "sources": sources,
            "merged_duplicate": True,
            "duplicate_text": loser.text,
            "duplicate_bbox_xyxy_norm": list(loser.bbox_xyxy_norm),
        }
        winner.candidates = list(previous.candidates) + list(candidate.candidates)
        merged[duplicate_index] = winner
    return merged


def _option5_duplicate(left: OCRCandidate, right: OCRCandidate) -> bool:
    if left.unit_id != right.unit_id or left.content_type != right.content_type:
        return False
    if left.content_type == "table_cell" and (
        left.table_id != right.table_id or left.cell_id != right.cell_id
    ):
        return False
    left_bbox = left.provenance.get("bbox_source")
    right_bbox = right.provenance.get("bbox_source")
    if left_bbox != "nemotron_local" or right_bbox != "nemotron_local":
        return False
    if bbox_iou(left.bbox_xyxy_norm, right.bbox_xyxy_norm) < 0.80:
        return False
    return text_similarity(left.text, right.text) >= 0.90


def _apply_vietnamese_result(
    state: _CandidateState,
    response: Any,
    backend: Any,
    *,
    config: Option5Config,
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
        multiline_split=bool(state.unit.metadata.get("multiline_split")),
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
    state.candidate.source = "option5_vietnamese_recognizer"
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
    state.candidate.provenance.update(
        {
            "selected_backend": "nemotron",
            "fallback_reason": reason,
            "vietnamese_candidate_text": (
                state.debug.get("vietnamese", {}).get("text")
                if isinstance(state.debug.get("vietnamese"), Mapping)
                else None
            ),
        }
    )
    state.debug["selected_backend"] = "nemotron"
    state.debug["vietnamese_fallback"] = {
        "reason": reason,
        "error": str(error) if error else None,
    }


def _vietnamese_quality_gate(
    text: str,
    score: float | None,
    decision: NemotronLanguageDecision,
    *,
    config: Option5Config,
    multiline_split: bool = False,
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
        if not config.allow_scoreless_vietnamese:
            return False, "vietnamese_score_missing"
        if decision.route != VIETNAMESE or (
            decision.confidence or 0.0
        ) < float(config.scoreless_language_confidence):
            return False, "vietnamese_scoreless_language_confidence_too_low"
        if _abnormal_or_prompt_like(text):
            return False, "vietnamese_scoreless_output_suspect"
        return True, "accepted_scoreless_server_policy"
    try:
        numeric_score = float(score)
        if numeric_score < float(config.vietnamese_score_threshold):
            # A line detector fixes the recognizer's input geometry.  On
            # long Vietnamese paragraphs VietOCR's confidence is sometimes
            # conservatively below 0.80 even when the returned line is
            # readable; sending that corrected line back to a paragraph-level
            # Nemotron crop is both slower and less faithful.  Relax only for
            # detector/projection line units, require visible Vietnamese
            # diacritics, and keep a hard lower bound to reject garbage.
            if (
                multiline_split
                and numeric_score >= 0.35
                and _plausible_split_vietnamese(text)
            ):
                return True, "accepted_multiline_relaxed_score"
            return False, "vietnamese_score_below_threshold"
    except (TypeError, ValueError):
        return False, "vietnamese_score_invalid"
    return True, "accepted_score_threshold"


def _plausible_split_vietnamese(text: str) -> bool:
    """Conservative language/shape check for the multiline score relaxation."""

    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) < 6 or _abnormal_or_prompt_like(compact):
        return False
    if not any(character.isalpha() for character in compact):
        return False
    digit_count = sum(character.isdigit() for character in compact)
    if digit_count / max(1, len(compact)) > 0.75:
        return False
    # Do not accept low-confidence English labels or ASCII diagram noise just
    # because the document-level prior is Vietnamese.
    return any(character in _OPTION3_VIETNAMESE_UNICODE for character in compact)


def _abnormal_or_prompt_like(text: str) -> bool:
    raw = str(text or "")
    # A recognizer failure can return an unbounded repeated stream.  Do not
    # feed megabytes of it through regex/Counter: real line crops are short,
    # and rejecting an output over this bound is both safer and much faster.
    if len(raw) > 4096:
        return True
    compact = re.sub(r"\s+", " ", raw).strip().lower()
    if len(compact) > 2048:
        return True
    if re.search(r"(.)\1{5,}", compact):
        return True
    # VietOCR failures on oversized/incorrect crops often look superficially
    # fluent but repeat the same word or short phrase.  Treat those as a
    # quality-gate failure so the selective Nemotron fallback preserves
    # accuracy instead of accepting a fast, corrupted result.
    tokens = re.findall(r"[\wÀ-ỹĐđ]+", compact, flags=re.UNICODE)
    if len(tokens) >= 8:
        counts = Counter(tokens)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= 4 and most_common_count / len(tokens) >= 0.34:
            return True
        bigrams = Counter(zip(tokens, tokens[1:]))
        if bigrams and bigrams.most_common(1)[0][1] >= 3:
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


def _page_cropper(page: OCRPage) -> PageImageCropper | None:
    """Decode a page image once when it is valid, else keep old fallbacks."""
    try:
        # Option 5 sends many line crops in one document batch.  JPEG crops
        # preserve OCR-relevant detail while avoiding the expensive PNG
        # encoder; the default cropper format remains PNG for the other
        # isolated pipelines.
        return PageImageCropper(page.image_b64, output_format="JPEG")
    except Exception:  # noqa: BLE001 - malformed image remains page-local
        return None


def _full_page_recall_units(
    page: OCRPage,
    existing_units: Sequence[OCRUnit],
    response: Any,
    *,
    cropper: PageImageCropper | None,
    min_detector_score: float,
    max_new_units: int,
) -> list[OCRUnit]:
    """Turn uncovered full-page PP-OCRv6 lines into bounded OCR units.

    Page Elements remains authoritative for semantic regions.  This helper is
    deliberately conservative: a detector line that is already covered by a
    semantic unit is ignored, while uncovered lines get a padded crop and are
    routed through the normal Vietnamese recognizer.  Recognition is still
    responsible for deciding whether a detector false positive is useful.
    """

    if not page.image_b64:
        return []
    existing_text_units = [
        unit
        for unit in existing_units
        if unit.kind != "table_cell" and clamp_bbox(unit.bbox_xyxy_norm) is not None
    ]
    accepted: list[tuple[tuple[float, float, float, float], float | None]] = []
    for detection in detector_boxes(response):
        bbox = clamp_bbox(detection.bbox)
        if bbox is None:
            continue
        score = detection.score
        if score is not None and score < float(min_detector_score):
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < 0.008 or height < 0.002:
            continue
        if cropper is not None:
            if width * cropper.width < 10 or height * cropper.height < 5:
                continue
        # Do not duplicate an existing line crop.  A semantic parent can be
        # intentionally broad, however: the two problematic fields in the
        # certificate are inside broad Page Elements boxes but were never
        # materialized as line units.  Permit a detector line inside such a
        # parent when the parent is materially larger than the line.  Already
        # split units remain authoritative and are still suppressed.
        covered_by_existing = False
        for existing_unit in existing_text_units:
            existing = clamp_bbox(existing_unit.bbox_xyxy_norm)
            if existing is None:
                continue
            overlap = bbox_iou(bbox, existing)
            inside = containment(bbox, existing)
            if inside >= 0.80:
                is_line_unit = bool(
                    existing_unit.metadata.get("multiline_split")
                    or existing_unit.source == "ppocrv6_line_detector"
                )
                parent_area = bbox_area(existing)
                line_area = bbox_area(bbox)
                if is_line_unit or parent_area <= line_area * 2.5:
                    covered_by_existing = True
                    break
                # This is a broad semantic parent. Let the full-page
                # detector expose individual lines inside it; the caller may
                # replace the parent when at least two such lines exist.
                continue
            if overlap >= 0.35:
                covered_by_existing = True
                break
        if covered_by_existing:
            continue
        if any(
            bbox_iou(bbox, previous) >= 0.65
            or containment(bbox, previous) >= 0.85
            for previous, _previous_score in accepted
        ):
            continue
        accepted.append((bbox, score))

    if not accepted:
        return []
    if len(accepted) > max(1, int(max_new_units)):
        accepted = sorted(
            accepted,
            key=lambda item: (
                -(float(item[1]) if item[1] is not None else 0.0),
                item[0][1],
                item[0][0],
            ),
        )[: max(1, int(max_new_units))]
    accepted.sort(key=lambda item: (item[0][1], item[0][0]))

    all_boxes = [
        unit.bbox_xyxy_norm
        for unit in existing_units
        if clamp_bbox(unit.bbox_xyxy_norm) is not None
    ]
    result: list[OCRUnit] = []
    page_number = int(page.page_number or 0)
    for index, (bbox, score) in enumerate(accepted):
        line_height = bbox[3] - bbox[1]
        if cropper is not None:
            crop = cropper.crop(
                bbox,
                local_text_height=line_height,
                add_padding=True,
            )
        else:
            crop = crop_image_b64(
                page.image_b64,
                bbox,
                local_text_height=line_height,
                add_padding=True,
            )
        if crop is None:
            continue
        center = ((bbox[1] + bbox[3]) / 2.0, (bbox[0] + bbox[2]) / 2.0)
        reading_order = sum(
            1
            for other in all_boxes
            if ((other[1] + other[3]) / 2.0, (other[0] + other[2]) / 2.0) < center
        )
        unit = OCRUnit(
            unit_id=f"page-{page_number}-full-page-recall-{index}",
            kind="text_block",
            source="ppocrv6_full_page_recall",
            bbox_xyxy_norm=bbox,
            crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
            crop_b64=crop.image_b64,
            crop_shape_hw=crop.shape_hw,
            reading_order=reading_order,
            detector_score=score,
            label="text",
            metadata={
                "full_page_recall": True,
                "line_detector": "PP-OCRv6_medium_det",
                "line_detector_score": score,
                "local_text_height_norm": line_height,
                "padding_applied": crop.bbox_xyxy_norm != bbox,
                "reading_order_inferred": True,
            },
        )
        result.append(unit)
        all_boxes.append(bbox)
    return result


def _vietnamese_crop(
    page: OCRPage,
    state: _CandidateState,
    *,
    cropper: PageImageCropper | None = None,
):
    # ``build_ocr_units`` and the multiline splitter already materialize the
    # exact crop that will be sent to the recognizer.  Reusing it avoids a
    # second PIL/PNG encode per unit (the old path encoded every line twice),
    # and is especially important for a 20–60 line semantic parent.  Do not
    # reuse it when Nemotron supplied a tighter local bbox.
    if (
        state.unit.crop_b64
        and bbox_iou(state.candidate.bbox_xyxy_norm, state.unit.bbox_xyxy_norm)
        >= 0.999
    ):
        return CroppedImage(
            bbox_xyxy_norm=state.unit.crop_bbox_xyxy_norm,
            image_b64=state.unit.crop_b64,
            shape_hw=state.unit.crop_shape_hw,
        )
    padding = state.candidate.content_type != "table_cell"
    local_height = state.unit.metadata.get("local_text_height_norm")
    try:
        local_height = float(local_height) if local_height is not None else None
    except (TypeError, ValueError):
        local_height = None
    if cropper is not None:
        return cropper.crop(
            state.candidate.bbox_xyxy_norm,
            local_text_height=local_height,
            add_padding=padding,
        )
    return crop_image_b64(
        page.image_b64,
        state.candidate.bbox_xyxy_norm,
        local_text_height=local_height,
        add_padding=padding,
    )


def _build_tables(
    page: OCRPage,
    candidates: Sequence[OCRCandidate],
    *,
    include_page_element_table_regions: bool = False,
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
    if include_page_element_table_regions and not tables:
        # Without Table Structure there are no cell boxes to assemble. Keep
        # the Page Elements table bbox and the VLM's whole-table transcript as
        # a coarse table artifact instead of fabricating cell coordinates.
        for candidate in candidates:
            if candidate.content_type != "table" or not candidate.table_id:
                continue
            tables.append(
                {
                    "table_id": candidate.table_id,
                    "bbox_xyxy_norm": list(candidate.bbox_xyxy_norm),
                    "cells": [],
                    "text": candidate.text,
                    "table_text_format": "plain",
                    "structure_source": "page_elements_v3",
                    "cell_structure_available": False,
                }
            )
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


def _suppress_nested_text_units(units: Sequence[OCRUnit]) -> list[OCRUnit]:
    """Suppress near-duplicate low-confidence semantic boxes.

    On noisy scans Page Elements can return multiple nested recall boxes over
    the same paragraph.  Keep a broad parent when it is materially larger
    than its children (it may contain text that the detector missed), but
    remove a contained near-duplicate so OCR does not repeat the same region.
    This is an Option-5-local pass; table cells and isolated paragraphs stay
    untouched.
    """

    text_units = [
        unit
        for unit in units
        if unit.kind in {"text_block", "title"}
        and clamp_bbox(unit.bbox_xyxy_norm) is not None
    ]
    if len(text_units) < 3:
        return list(units)

    suppressed: set[int] = set()
    for left_index, left in enumerate(text_units):
        left_bbox = clamp_bbox(left.bbox_xyxy_norm)
        if left_bbox is None:
            continue
        left_area = bbox_area(left_bbox)
        if left_area <= 0.0:
            continue
        try:
            left_score = float(left.detector_score or 0.0)
        except (TypeError, ValueError):
            left_score = 0.0
        for right in text_units[left_index + 1 :]:
            right_bbox = clamp_bbox(right.bbox_xyxy_norm)
            if right_bbox is None:
                continue
            right_area = bbox_area(right_bbox)
            if right_area <= 0.0:
                continue
            if containment(left_bbox, right_bbox) < 0.90 and containment(
                right_bbox, left_bbox
            ) < 0.90:
                continue
            if left_area >= right_area:
                larger, smaller = left, right
                larger_area, smaller_area = left_area, right_area
                larger_score, smaller_score = left_score, float(
                    right.detector_score or 0.0
                )
            else:
                larger, smaller = right, left
                larger_area, smaller_area = right_area, left_area
                larger_score, smaller_score = float(
                    right.detector_score or 0.0
                ), left_score
            area_ratio = larger_area / max(smaller_area, 1e-9)
            if area_ratio > 4.0:
                # The parent may contain additional text; do not sacrifice
                # recall merely because it encloses a precise child.
                continue
            if larger_score > 0.45 and smaller_score > 0.45:
                continue
            # Prefer the larger low-score region for recall.  When the small
            # region is substantially more confident, retain both because the
            # parent may cover text that the child does not.
            if smaller_score >= larger_score + 0.20:
                continue
            suppressed.add(id(smaller))

    return [unit for unit in units if id(unit) not in suppressed]


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
    if unit.kind == "table_region":
        return "table"
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
        pipeline=OPTION5_PIPELINE_NAME,
        text=page.native_text,
        source="native_passthrough",
        model="PDFium native text",
        language="native",
        timing=_timing(total_seconds=elapsed, unit_count=0, skipped_native=True),
        status="skipped",
    )


def _backend_request_count(backend: Any, input_count: int) -> int:
    """Return transport-chunk count exposed by an adapter, if available."""

    if not input_count:
        return 0
    value = getattr(backend, "last_request_count", None)
    try:
        if value is not None and int(value) > 0:
            return int(value)
    except (TypeError, ValueError):
        pass
    # Test doubles and custom adapters are one logical invocation from the
    # coordinator's perspective.  HTTPImageBackend records the exact chunk
    # count above, so production diagnostics remain transport-aware.
    return 1


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
        "pipeline": OPTION5_SELECTOR,
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
