# SPDX-License-Identifier: Apache-2.0

"""Option 3: Nemotron baseline with raw-text language routing to VietOCR.

This branch intentionally has no detector.  Page Elements v3 and Table
Structure v1 provide the semantic OCR units; Nemotron OCR v2 is authoritative
for every unit and a Vietnamese-only recognizer may replace its text only
after the quality gate succeeds.  The selector remains ``pipeline-option3``
for API compatibility while the internal pipeline name describes the actual
architecture.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    detect_nemotron_page_prior,
    route_nemotron_text,
)
from nemo_retriever.common.modality.ocr.isolated.units import (
    build_ocr_units,
    table_payload,
)

OPTION3_SELECTOR = "pipeline-option3"
OPTION3_PIPELINE_NAME = "option3_nemotron_language_routed_vietnamese_ocr"


@dataclass(frozen=True)
class Option3Config:
    """Configuration for the detector-free Option 3 branch."""

    language: str | None = "auto"
    skip_native_text: bool = True
    include_table_cells: bool = True
    scan_page_fallback: bool = True
    batch_size: int = 8
    # Option 3 keeps page outputs in input order while allowing a bounded
    # number of page-local Nemotron/VietOCR calls to overlap.  The single-GPU
    # development preset sets the same limit at the VietOCR sidecar.
    page_concurrency: int = 4
    request_timeout_s: float = 120.0
    vietnamese_score_threshold: float = 0.80
    allow_scoreless_vietnamese: bool = False
    scoreless_language_confidence: float = 0.90
    language_min_chars: int = 24
    language_min_words: int = 4


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


class Option3Pipeline:
    """Run one page's semantic crops through Nemotron, then VietOCR in batch."""

    pipeline_name = OPTION3_PIPELINE_NAME

    def __init__(
        self,
        nemotron: OCRBackend | Any,
        vietnamese_recognizer: VietnameseRecognizerBackend | Any,
        *,
        config: Option3Config | None = None,
    ) -> None:
        if not hasattr(nemotron, "recognize"):
            raise TypeError("nemotron backend must expose recognize(images)")
        if not hasattr(vietnamese_recognizer, "recognize"):
            raise TypeError(
                "vietnamese_recognizer backend must expose recognize(images)"
            )
        self.nemotron = nemotron
        self.vietnamese_recognizer = vietnamese_recognizer
        self.config = config or Option3Config()

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

        units = build_ocr_units(
            normalized_page,
            include_table_cells=self.config.include_table_cells,
            # Visual regions are deliberately not OCR units in Option 3.
            include_visual_regions=False,
            # Padding normal text crops is useful; table cells stay inside
            # their structure box so adjacent cells cannot be duplicated.
            pad_table_cells=False,
        )
        units = _remove_table_text_overlaps(normalized_page, units)
        if (
            self.config.scan_page_fallback
            and _is_scan_page(normalized_page)
            and not units
        ):
            fallback = crop_image_b64(
                normalized_page.image_b64,
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
        vietnamese_seconds = 0.0
        vietnamese_states: list[_CandidateState] = []
        for state in states:
            if state.decision.route != VIETNAMESE:
                continue
            crop = _vietnamese_crop(normalized_page, state)
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

        canonical = _merge_option3_candidates([state.candidate for state in states])
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
            nemotron_request_count=1,
            vietnamese_batch_count=vietnamese_batch_count,
            vietnamese_request_count=vietnamese_batch_count,
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
            thread_name_prefix="option3-page",
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


def run_option3_batch(
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
    # Kept in the public call shape for graph compatibility.  Option 3 never
    # reads or requires these legacy detector/recognizer endpoints.
    line_detector_invoke_url: str | None = None,
    ocr_recognizer_invoke_url: str | None = None,
    tesseract_ocr_invoke_url: str | None = None,
) -> Any:
    """Run Option 3 over a dataframe batch without a detector sidecar."""

    del line_detector_invoke_url, ocr_recognizer_invoke_url, tesseract_ocr_invoke_url
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
            "pipeline-option3 requires Option 3 endpoints: " + ", ".join(missing)
        )

    batch_size = max(1, int(inference_batch_size or 1))
    timeout = max(1.0, float(request_timeout_s or 120.0))
    secret = ocr_api_key or api_key
    runner = Option3Pipeline(
        make_nemotron_backend(
            str(ocr_invoke_url),
            api_key=secret,
            language=ocr_lang or "multi",
            timeout_s=timeout,
            batch_size=batch_size,
        ),
        make_vietnamese_recognizer(
            str(vietnamese_ocr_invoke_url),
            api_key=secret,
            timeout_s=timeout,
            batch_size=batch_size,
        ),
        config=Option3Config(
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
            _apply_option3_output(
                row.to_dict(),
                output,
                extract_text=bool(extract_text),
                extract_tables=bool(extract_tables),
            )
            for (_, row), output in zip(batch_df.iterrows(), outputs)
        ]
    )


def _apply_option3_output(
    row: dict[str, Any],
    output: OCRPageOutput,
    *,
    extract_text: bool,
    extract_tables: bool,
) -> dict[str, Any]:
    """Restore Option 3 output using the shared dataframe contract."""

    metadata = row.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata.update(
        {
            "ocr_pipeline": OPTION3_SELECTOR,
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
        source="option3_nemotron",
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


def _merge_option3_candidates(
    candidates: Sequence[OCRCandidate],
) -> list[OCRCandidate]:
    """Deduplicate only safe same-unit duplicates for Option 3.

    A Nemotron crop may contain more than one recognition item.  When the
    backend omits local boxes, all of those items intentionally inherit the
    same parent bbox; the shared generic merger would mistake that geometry
    for a duplicate and drop a distinct line.  Conservative same-unit,
    same-local-box merging keeps output stable without sacrificing nearby
    lines.
    """

    merged: list[OCRCandidate] = []
    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(merged)
                if _option3_duplicate(candidate, previous)
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


def _option3_duplicate(left: OCRCandidate, right: OCRCandidate) -> bool:
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
    config: Option3Config,
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
    state.candidate.source = "option3_vietnamese_recognizer"
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
    config: Option3Config,
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


def _vietnamese_crop(page: OCRPage, state: _CandidateState):
    padding = state.candidate.content_type != "table_cell"
    local_height = state.unit.metadata.get("local_text_height_norm")
    try:
        local_height = float(local_height) if local_height is not None else None
    except (TypeError, ValueError):
        local_height = None
    return crop_image_b64(
        page.image_b64,
        state.candidate.bbox_xyxy_norm,
        local_text_height=local_height,
        add_padding=padding,
    )


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
        pipeline=OPTION3_PIPELINE_NAME,
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
        "pipeline": OPTION3_SELECTOR,
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
