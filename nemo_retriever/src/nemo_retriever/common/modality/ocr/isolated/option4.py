# SPDX-License-Identifier: Apache-2.0

"""Isolated Option 4: Tesseract-first OCR with Nemotron fallback fusion."""

from __future__ import annotations

import base64
import concurrent.futures
import io
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.adapters import (
    OCRBackend,
    OCRDetectorBackend,
    detector_boxes,
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
    clamp_bbox,
    crop_image_b64,
    inside_or_overlaps,
    language_quality,
    map_local_bbox,
    merge_candidates,
    numeric_tokens,
    text_similarity,
    union_bbox,
)
from nemo_retriever.common.modality.ocr.isolated.language_router import (
    detect_probe_language,
)
from nemo_retriever.common.modality.ocr.isolated.units import (
    VISUAL_LABELS,
    build_ocr_units,
    page_element_detections,
    table_payload,
    visual_exclusion_boxes,
)


@dataclass(frozen=True)
class Option4Config:
    """Tuning for Tesseract-first recognition and fallback fusion."""

    # Option 4 is bilingual: a small vie+eng probe routes Vietnamese to
    # Tesseract vie and English/uncertain crops to multilingual Nemotron.
    language: str | None = "auto"
    skip_native_text: bool = True
    max_workers: int = 2
    include_table_cells: bool = True
    scan_page_fallback: bool = True
    request_timeout_s: float = 120.0
    line_detection: bool = True
    preserve_parent_horizontal_bounds: bool = True
    projection_fallback: bool = True
    tesseract_first: bool = True
    tesseract_language: str | None = "vie"
    language_probe_language: str | None = "vie+eng"
    language_routing: bool = True
    language_probe_min_score: float = 0.70
    tesseract_psm: int = 7
    tesseract_min_score: float = 0.80


class Option4Pipeline:
    """Run Tesseract first and use Nemotron only for uncertain crops.

    Page Elements and Table Structure provide semantic regions. PP-OCRv6
    medium det refines each region into line crops; Nemotron OCR v2 and
    Tesseract 5 then receive the exact same line image. The two response
    shapes are aligned after recognition. A sufficiently confident Tesseract
    result is accepted immediately; Nemotron is invoked for empty or
    low-quality Tesseract results and the fusion path is retained for those
    disagreements.
    """

    pipeline_name = "option4_parallel_nemotron_tesseract_fusion"

    def __init__(
        self,
        nemotron: OCRBackend | Any,
        tesseract: OCRBackend | Any,
        *,
        line_detector: OCRDetectorBackend | Any | None = None,
        language_probe: OCRBackend | Any | None = None,
        config: Option4Config | None = None,
    ) -> None:
        for name, backend in (("nemotron", nemotron), ("tesseract", tesseract)):
            if not hasattr(backend, "recognize"):
                raise TypeError(f"{name} backend must expose recognize(images)")
        self.nemotron = nemotron
        self.tesseract = tesseract
        if line_detector is not None and not hasattr(line_detector, "detect"):
            raise TypeError("line_detector backend must expose detect(images)")
        self.line_detector = line_detector
        if language_probe is not None and not hasattr(language_probe, "recognize"):
            raise TypeError("language_probe backend must expose recognize(images)")
        self.language_probe = language_probe
        self.config = config or Option4Config()
        self._tesseract_disabled_for_page = False

    def process_page(self, page: OCRPage | Mapping[str, Any] | Any) -> OCRPageOutput:
        started = time.perf_counter()
        # A failed sidecar must not make every later line wait through the
        # same network failure.  Reset the small circuit breaker per page so a
        # restarted Tesseract service can be used on the next page.
        self._tesseract_disabled_for_page = False
        normalized_page = page_value(page)
        if self.config.skip_native_text and _is_native_page(normalized_page):
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                text=normalized_page.native_text,
                source="native_passthrough",
                timing={
                    "seconds": time.perf_counter() - started,
                    "skipped_native": True,
                    "unit_count": 0,
                },
                status="skipped",
            )
        if not normalized_page.image_b64:
            return OCRPageOutput(
                pipeline=self.pipeline_name,
                source="option4",
                model="Nemotron OCR v2 + Tesseract 5",
                language=self.config.language,
                errors=[_error("input", "page image is unavailable")],
                timing={"seconds": time.perf_counter() - started, "unit_count": 0},
                status="failed",
            )

        units = build_ocr_units(
            normalized_page,
            include_table_cells=self.config.include_table_cells,
            include_visual_regions=False,
            # Table Structure already supplies cell geometry. Do not expand
            # cell crops back over neighboring grid lines in Option 4.
            pad_table_cells=False,
        )
        has_visual_regions = any(
            str(item.get("label_name") or "").strip().lower() in VISUAL_LABELS
            for item in page_element_detections(normalized_page.page_elements_v3)
        )
        if (
            self.config.scan_page_fallback
            and _is_scan_page(normalized_page)
            and not units
            and not has_visual_regions
        ):
            crop = crop_image_b64(normalized_page.image_b64, (0.0, 0.0, 1.0, 1.0))
            if crop is not None:
                units.append(
                    OCRUnit(
                        unit_id=f"page-{normalized_page.page_number or 0}-scan-page",
                        kind="text_block",
                        source="scan_page_fallback",
                        bbox_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                        crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                        crop_b64=crop.image_b64,
                        crop_shape_hw=crop.shape_hw,
                        reading_order=100000,
                        label="text",
                        metadata={"scan_page_fallback": True},
                    )
                )

        errors: list[dict[str, Any]] = []
        line_units = self._build_line_units(normalized_page, units, errors)
        canonical: list[OCRCandidate] = []
        debug_candidates: list[dict[str, Any]] = []
        for unit in line_units:
            observations = self._observations(unit, errors)
            selected, debug = _fuse_unit(
                unit,
                observations,
                language=self.config.language,
                prefer_tesseract=self.config.tesseract_first,
            )
            debug_candidates.append(debug)
            if selected is not None:
                canonical.append(selected)

        # Block fallback and line recall can overlap. Keep one canonical line
        # while retaining the discarded backend/crop in provenance.
        canonical = merge_candidates(canonical)
        canonical.sort(
            key=lambda candidate: (
                candidate.reading_order,
                candidate.bbox_xyxy_norm[1],
                candidate.bbox_xyxy_norm[0],
            )
        )
        blocks = [candidate.to_dict() for candidate in canonical]
        text = "\n".join(
            candidate.text
            for candidate in canonical
            if candidate.content_type != "table_cell" and candidate.text.strip()
        )
        scores = [
            candidate.score for candidate in canonical if candidate.score is not None
        ]
        tables = table_payload(normalized_page)
        for table in tables:
            table["cells"] = [
                candidate.to_dict()
                for candidate in canonical
                if candidate.content_type == "table_cell"
                and candidate.table_id == table["table_id"]
            ]
            table["text"] = _table_markdown(table["cells"])
            table["table_text_format"] = "markdown"
        return OCRPageOutput(
            pipeline=self.pipeline_name,
            text=text,
            ocr_text_blocks=blocks,
            bbox_xyxy_norm=union_bbox(canonical),
            score=sum(scores) / len(scores) if scores else None,
            confidence=sum(scores) / len(scores) if scores else None,
            source="option4_fusion",
            model="Nemotron OCR v2 + Tesseract 5",
            language=self.config.language,
            tables=tables,
            candidates=debug_candidates,
            errors=errors,
            timing={
                "seconds": time.perf_counter() - started,
                "unit_count": len(units),
                "line_unit_count": len(line_units),
                "canonical_block_count": len(canonical),
                "parallel_backends": ["nemotron", "tesseract"],
                "language_router": bool(
                    self.language_probe and self.config.language_routing
                ),
                "language_probe": self.config.language_probe_language,
                "recognition_strategy": (
                    "language_router_tesseract_vie_or_nemotron"
                    if self.config.language_routing and self.language_probe
                    else (
                        "tesseract_first_nemotron_fallback"
                        if self.config.tesseract_first
                        else "parallel_fusion"
                    )
                ),
                "tesseract_language": self.config.tesseract_language,
                "tesseract_psm": int(self.config.tesseract_psm),
                "tesseract_min_score": float(self.config.tesseract_min_score),
                "line_detector": bool(self.line_detector and self.config.line_detection),
            },
            status="partial"
            if errors and canonical
            else ("failed" if errors else "completed"),
        )

    def _build_line_units(
        self,
        page: OCRPage,
        units: Sequence[OCRUnit],
        errors: list[dict[str, Any]],
    ) -> list[OCRUnit]:
        """Split each shared block/cell crop into line-level OCR units."""
        if not units or not self.line_detector or not self.config.line_detection:
            return list(units)
        try:
            responses = list(
                self.line_detector.detect([unit.crop_b64 for unit in units])
            )
        except Exception as exc:  # noqa: BLE001 - page-local detector failure.
            errors.append(_error("option4.line_detector", exc))
            if self.config.projection_fallback:
                projected = [
                    fallback
                    for unit in units
                    for fallback in _projection_line_units(page, unit)
                ]
                if projected:
                    return sorted(
                        projected,
                        key=lambda unit: (
                            unit.reading_order,
                            unit.bbox_xyxy_norm[1],
                            unit.bbox_xyxy_norm[0],
                        ),
                    )
            return [_fallback_line_unit(unit) for unit in units]

        line_units: list[OCRUnit] = []
        excluded_regions = visual_exclusion_boxes(page)
        for index, unit in enumerate(units):
            response = responses[index] if index < len(responses) else None
            detections = detector_boxes(response)
            accepted = 0
            for line_index, detection in enumerate(detections):
                mapped = map_local_bbox(
                    detection.bbox,
                    unit.crop_bbox_xyxy_norm,
                    unit.crop_shape_hw,
                )
                if unit.kind != "table_cell" and inside_or_overlaps(
                    mapped, excluded_regions, threshold=0.60
                ):
                    continue
                if self.config.preserve_parent_horizontal_bounds:
                    # The line detector is authoritative for the vertical
                    # slice, not for the horizontal extent.  Keeping the
                    # parent region's left/right edges prevents the common
                    # `Pursuant -> ursuant` failure when a detector trims the
                    # first or last glyph of a line.
                    mapped = _preserve_parent_horizontal_bounds(
                        mapped, unit.bbox_xyxy_norm
                    )
                local_height = max(mapped[3] - mapped[1], 1e-9)
                crop = _crop_line_unit(
                    page,
                    mapped,
                    unit,
                    local_text_height=local_height,
                )
                if crop is None:
                    continue
                line_units.append(
                    OCRUnit(
                        unit_id=f"{unit.unit_id}:line-{line_index}",
                        kind=unit.kind,
                        source="ppocrv6_line_detector",
                        bbox_xyxy_norm=mapped,
                        crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                        crop_b64=crop.image_b64,
                        crop_shape_hw=crop.shape_hw,
                        reading_order=int(unit.reading_order) * 1000 + line_index,
                        detector_score=detection.score,
                        label=unit.label,
                        table_id=unit.table_id,
                        cell_id=unit.cell_id,
                        metadata={
                            **unit.metadata,
                            "line_detector": "PP-OCRv6_medium_det",
                            "line_detector_score": detection.score,
                            "parent_unit_id": unit.unit_id,
                            "line_index": line_index,
                        },
                    )
                )
                accepted += 1
            if not accepted:
                projected = (
                    _projection_line_units(page, unit)
                    if self.config.projection_fallback
                    else []
                )
                line_units.extend(projected or [_fallback_line_unit(unit)])
        return sorted(
            line_units,
            key=lambda unit: (
                unit.reading_order,
                unit.bbox_xyxy_norm[1],
                unit.bbox_xyxy_norm[0],
            ),
        )

    def process_pages(
        self, pages: Sequence[OCRPage | Mapping[str, Any] | Any]
    ) -> list[OCRPageOutput]:
        return [self.process_page(page) for page in pages]

    def _observations(
        self, unit: OCRUnit, errors: list[dict[str, Any]]
    ) -> dict[str, BackendObservation]:
        """Route Vietnamese to Tesseract and other crops to Nemotron."""
        if self.config.language_routing and self.language_probe is not None:
            return self._routed_observations(unit, errors)

        if self.config.tesseract_first:
            if self._tesseract_disabled_for_page:
                tesseract = BackendObservation(
                    backend="tesseract",
                    model=str(getattr(self.tesseract, "model", "tesseract-5")),
                    language=getattr(self.tesseract, "language", None),
                    error=RuntimeError(
                        "Tesseract sidecar unavailable; skipped until the next page"
                    ),
                )
            else:
                tesseract = _observe_backend("tesseract", self.tesseract, unit)
                if tesseract.error:
                    # Tesseract is optional in this fusion branch.  Keep the
                    # failure inside candidates[*].error for diagnostics, but
                    # do not turn a successful Nemotron fallback into a row
                    # error that aborts the whole graph.
                    self._tesseract_disabled_for_page = True
            if _tesseract_is_usable(
                tesseract, min_score=float(self.config.tesseract_min_score)
            ):
                return {"tesseract": tesseract}

            nemotron = _observe_backend("nemotron", self.nemotron, unit)
            if nemotron.error:
                errors.append(_error("nemotron.recognizer", nemotron.error))
            return {"tesseract": tesseract, "nemotron": nemotron}

        return self._parallel_observations(unit, errors)

    def _routed_observations(
        self, unit: OCRUnit, errors: list[dict[str, Any]]
    ) -> dict[str, BackendObservation]:
        """Use a bilingual probe before selecting the recognition backend."""
        probe = _observe_backend("language_probe", self.language_probe, unit)
        decision = detect_probe_language(
            probe.text,
            probe.score,
            min_probe_score=float(self.config.language_probe_min_score),
        )
        unit.metadata["language_router"] = decision.to_dict()
        if probe.error is not None:
            unit.metadata["language_router"]["probe_error"] = str(probe.error)

        if decision.is_vietnamese:
            tesseract = _observe_backend("tesseract", self.tesseract, unit)
            if _tesseract_is_usable(
                tesseract, min_score=float(self.config.tesseract_min_score)
            ):
                unit.metadata["language_router"]["selected_backend"] = "tesseract"
                return {"tesseract": tesseract}
            unit.metadata["language_router"]["tesseract_rejected"] = {
                "text": tesseract.text[:240],
                "score": tesseract.score,
                "error": str(tesseract.error) if tesseract.error else None,
            }

        # English, mixed, uncertain, probe failure, and weak Vietnamese all
        # use Nemotron. Do not return a weak Tesseract candidate to fusion.
        nemotron = _observe_backend("nemotron", self.nemotron, unit)
        if nemotron.error:
            errors.append(_error("nemotron.recognizer", nemotron.error))
        unit.metadata["language_router"]["selected_backend"] = "nemotron"
        return {"nemotron": nemotron}

    def _parallel_observations(
        self, unit: OCRUnit, errors: list[dict[str, Any]]
    ) -> dict[str, BackendObservation]:
        """Run both backends concurrently when explicitly requested."""
        backends = {"nemotron": self.nemotron, "tesseract": self.tesseract}
        result = {}
        worker_count = max(2, int(self.config.max_workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {name: executor.submit(_observe_backend, name, backend, unit) for name, backend in backends.items()}
            for name, future in futures.items():
                observation = future.result()
                result[name] = observation
                if name == "tesseract" and observation.error:
                    self._tesseract_disabled_for_page = True
                if observation.error and name != "tesseract":
                    errors.append(_error(f"{name}.recognizer", observation.error))
        return result


@dataclass
class BackendObservation:
    backend: str
    model: str
    language: str | None
    text: str = ""
    score: float | None = None
    bbox_xyxy_norm: tuple[float, float, float, float] | None = None
    error: Exception | None = None
    raw_candidates: list[dict[str, Any]] = field(default_factory=list)


def _observe_backend(name: str, backend: Any, unit: OCRUnit) -> BackendObservation:
    model = str(
        getattr(
            backend, "model", "Nemotron OCR v2" if name == "nemotron" else "tesseract-5"
        )
    )
    language = getattr(backend, "language", None)
    try:
        responses = list(backend.recognize([unit.crop_b64]))
        response = responses[0] if responses else None
        items = recognition_items(response)
        if not items:
            return BackendObservation(backend=name, model=model, language=language)
        mapped_boxes = [
            map_local_bbox(item.bbox, unit.crop_bbox_xyxy_norm, unit.crop_shape_hw)
            for item in items
            if item.bbox is not None
        ]
        bbox = _union_boxes(mapped_boxes) or unit.bbox_xyxy_norm
        # Recognition backends often return only the ink-tight bbox.  Keep
        # the canonical line geometry at least as wide as the protected OCR
        # unit so the UI/result cannot regress to a crop that lost edge text.
        bbox = _union_boxes([bbox, unit.bbox_xyxy_norm]) or bbox
        text = _clean_ocr_text(
            "\n".join(item.text.strip() for item in items if item.text.strip()),
            table_cell=unit.kind == "table_cell",
        )
        if not text:
            return BackendObservation(backend=name, model=model, language=language)
        scores = [item.score for item in items if item.score is not None]
        item_language = next(
            (item.language for item in items if item.language), language
        )
        return BackendObservation(
            backend=name,
            model=str(next((item.model for item in items if item.model), model)),
            language=str(item_language) if item_language else None,
            text=text,
            score=sum(scores) / len(scores) if scores else None,
            bbox_xyxy_norm=bbox,
            raw_candidates=[
                {
                    "text": item.text,
                    "score": item.score,
                    "bbox_xyxy_norm": list(
                        map_local_bbox(
                            item.bbox, unit.crop_bbox_xyxy_norm, unit.crop_shape_hw
                        )
                    )
                    if item.bbox is not None
                    else list(unit.bbox_xyxy_norm),
                    "model": item.model or model,
                    "language": item.language or language,
                }
                for item in items
            ],
        )
    except Exception as exc:  # noqa: BLE001 - backend failure is page-local.
        return BackendObservation(
            backend=name, model=model, language=language, error=exc
        )


def _tesseract_is_usable(
    observation: BackendObservation, *, min_score: float
) -> bool:
    """Accept only sufficiently confident, non-garbage Tesseract output."""
    if observation.error is not None or not observation.text.strip():
        return False
    if observation.score is None or float(observation.score) < float(min_score):
        return False
    visible = [char for char in observation.text if not char.isspace()]
    if not any(char.isalnum() for char in visible):
        return False
    if any(char in {"�", "□"} or (ord(char) < 32 and char not in "\n\t") for char in visible):
        return False
    return True


def _fuse_unit(
    unit: OCRUnit,
    observations: Mapping[str, BackendObservation],
    *,
    language: str | None,
    prefer_tesseract: bool = False,
) -> tuple[OCRCandidate | None, dict[str, Any]]:
    available = [
        observation
        for observation in observations.values()
        if observation.text.strip() and observation.error is None
    ]
    debug = {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "bbox_xyxy_norm": list(unit.bbox_xyxy_norm),
        "selected_backend": None,
        "decision": "empty",
        "candidates": [
            _observation_dict(observation) for observation in observations.values()
        ],
    }
    language_router = unit.metadata.get("language_router")
    if isinstance(language_router, Mapping):
        debug["language_router"] = dict(language_router)
    if not available:
        debug["decision"] = "both_empty_or_failed"
        return None, debug
    if len(available) == 1:
        selected = available[0]
        debug["selected_backend"] = selected.backend
        debug["decision"] = (
            "tesseract_first_accepted"
            if selected.backend == "tesseract"
            else "fallback_other_backend"
        )
        return _candidate_from_observation(unit, selected, debug), debug

    left, right = available[0], available[1]
    similarity = text_similarity(left.text, right.text)
    left_quality = _fusion_quality(left, unit, language)
    right_quality = _fusion_quality(right, unit, language)
    if similarity >= 0.78:
        decision = "near_duplicate_keep_quality"
    else:
        decision = "different_keep_quality"
    selected = left if left_quality >= right_quality else right
    # When metrics are genuinely tied, prefer the configured first backend.
    # Tesseract is preferred by the default Vietnamese-first strategy; the
    # alternate parallel mode keeps Nemotron's historical tie-break behavior.
    if abs(left_quality - right_quality) <= 0.015:
        preferred_backend = "tesseract" if prefer_tesseract else "nemotron"
        selected = next(
            (item for item in available if item.backend == preferred_backend), left
        )
        decision = f"{decision}_{preferred_backend}_tiebreak"
    debug["selected_backend"] = selected.backend
    debug["decision"] = decision
    debug["text_similarity"] = similarity
    debug["quality"] = {
        item.backend: _fusion_quality(item, unit, language) for item in available
    }
    return _candidate_from_observation(unit, selected, debug), debug


def _candidate_from_observation(
    unit: OCRUnit,
    observation: BackendObservation,
    debug: Mapping[str, Any],
) -> OCRCandidate:
    provenance = {
        "ocr_unit": {
            "unit_id": unit.unit_id,
            "kind": unit.kind,
            "source": unit.source,
            "bbox_xyxy_norm": list(unit.bbox_xyxy_norm),
            "crop_bbox_xyxy_norm": list(unit.crop_bbox_xyxy_norm),
        },
        "selected_backend": observation.backend,
        "backend_candidates": list(debug.get("candidates") or []),
        "fusion_decision": debug.get("decision"),
        "language_router": debug.get("language_router"),
        "sources": ["nemotron", "tesseract"],
        "line_detector": unit.metadata.get("line_detector"),
        "line_detector_score": unit.metadata.get("line_detector_score"),
        "line_index": unit.metadata.get("line_index"),
        "line_detector_fallback": bool(
            unit.metadata.get("line_detector_fallback")
        ),
    }
    return OCRCandidate(
        text=observation.text.strip(),
        bbox_xyxy_norm=observation.bbox_xyxy_norm or unit.bbox_xyxy_norm,
        score=observation.score,
        source="option4_fusion",
        model=observation.model,
        language=observation.language,
        content_type="table_cell"
        if unit.kind == "table_cell"
        else ("title" if unit.kind == "title" else "text"),
        reading_order=unit.reading_order,
        unit_id=unit.unit_id,
        table_id=unit.table_id,
        cell_id=unit.cell_id,
        provenance=provenance,
        candidates=list(debug.get("candidates") or []),
    )


def _observation_dict(observation: BackendObservation) -> dict[str, Any]:
    value = {
        "backend": observation.backend,
        "model": observation.model,
        "language": observation.language,
        "text": observation.text,
        "score": observation.score,
        "confidence": observation.score,
        "bbox_xyxy_norm": list(observation.bbox_xyxy_norm)
        if observation.bbox_xyxy_norm
        else None,
        "raw_candidates": list(observation.raw_candidates),
    }
    if observation.error is not None:
        value["error"] = {
            "type": type(observation.error).__name__,
            "message": str(observation.error),
        }
    return value


def _fusion_quality(
    observation: BackendObservation, unit: OCRUnit, language: str | None
) -> float:
    text = observation.text.strip()
    if not text:
        return 0.0
    confidence = (
        0.5 if observation.score is None else max(0.0, min(1.0, observation.score))
    )
    visible = [char for char in text if not char.isspace()]
    replacement = sum(char in {"�", "□"} for char in visible) / max(1, len(visible))
    control = sum(ord(char) < 32 and char not in "\n\t" for char in visible) / max(
        1, len(visible)
    )
    glyph_quality = max(0.0, 1.0 - replacement - control)
    lang_quality = language_quality(text, observation.language or language)
    tokens = numeric_tokens(text)
    numeric_quality = 1.0 if tokens else 0.55
    # Numeric strings are deliberately rewarded for retaining complete tokens;
    # malformed replacement glyphs and missing Vietnamese marks are penalized.
    return max(
        0.0,
        min(
            1.0,
            0.46 * confidence
            + 0.22 * glyph_quality
            + 0.17 * lang_quality
            + 0.15 * numeric_quality,
        ),
    )


def _clean_ocr_text(value: str, *, table_cell: bool = False) -> str:
    """Remove grid-line artifacts without stripping valid OCR punctuation."""
    lines: list[str] = []
    for raw_line in str(value or "").splitlines() or [str(value or "")]:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if table_cell:
            line = re.sub(r"^\s*\|+\s*", "", line)
            line = re.sub(r"\s*\|+\s*$", "", line).strip()
        visible = "".join(char for char in line if not char.isspace())
        if not visible:
            continue
        # Keep `.vn`, `A-01`, and `0`; reject border-only results such as
        # `||||` or `-----`.
        if not any(char.isalnum() for char in visible):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _fallback_line_unit(unit: OCRUnit) -> OCRUnit:
    return OCRUnit(
        unit_id=unit.unit_id,
        kind=unit.kind,
        source=unit.source,
        bbox_xyxy_norm=unit.bbox_xyxy_norm,
        crop_bbox_xyxy_norm=unit.crop_bbox_xyxy_norm,
        crop_b64=unit.crop_b64,
        crop_shape_hw=unit.crop_shape_hw,
        reading_order=unit.reading_order,
        detector_score=unit.detector_score,
        label=unit.label,
        table_id=unit.table_id,
        cell_id=unit.cell_id,
        metadata={**unit.metadata, "line_detector_fallback": True},
    )


def _preserve_parent_horizontal_bounds(
    line_bbox: Sequence[float], parent_bbox: Sequence[float]
) -> tuple[float, float, float, float]:
    """Keep detector Y bounds while retaining the parent region's X span."""
    line = clamp_bbox(line_bbox)
    parent = clamp_bbox(parent_bbox)
    if line is None:
        return parent or (0.0, 0.0, 1.0, 1.0)
    if parent is None:
        return line
    return (
        min(line[0], parent[0]),
        line[1],
        max(line[2], parent[2]),
        line[3],
    )


def _crop_line_unit(
    page: OCRPage,
    line_bbox: Sequence[float],
    unit: OCRUnit,
    *,
    local_text_height: float,
):
    """Create a line crop without crossing a table-cell boundary."""
    if unit.kind != "table_cell":
        return crop_image_b64(
            page.image_b64,
            line_bbox,
            local_text_height=local_text_height,
            add_padding=True,
        )

    parent = clamp_bbox(unit.bbox_xyxy_norm)
    line = clamp_bbox(line_bbox)
    if parent is None or line is None:
        return None
    # Add a small vertical margin for ascenders/descenders, but clamp it to
    # the cell.  Horizontal padding is intentionally not added: neighboring
    # cells must never leak into this OCR crop.
    vertical_padding = min(
        max((line[3] - line[1]) * 0.35, 1e-9),
        max((parent[3] - parent[1]) * 0.25, 1e-9),
    )
    safe_bbox = (
        parent[0],
        max(parent[1], line[1] - vertical_padding),
        parent[2],
        min(parent[3], line[3] + vertical_padding),
    )
    return crop_image_b64(page.image_b64, safe_bbox)


def _projection_line_units(page: OCRPage, unit: OCRUnit) -> list[OCRUnit]:
    """Fallback line splitter for a detector-empty multi-line region.

    This deliberately uses only horizontal ink projection.  It is a recall
    fallback, not a replacement for PP-OCRv6: when no usable bands are found,
    the original unit is retained so the pipeline does not silently lose text.
    """
    bands = _horizontal_ink_bands(unit.crop_b64)
    if not bands:
        return []
    height = max(1, int(unit.crop_shape_hw[0]))
    line_units: list[OCRUnit] = []
    for line_index, (top, bottom) in enumerate(bands):
        local_bbox = (0.0, top / height, 1.0, bottom / height)
        mapped = map_local_bbox(
            local_bbox,
            unit.crop_bbox_xyxy_norm,
            unit.crop_shape_hw,
        )
        mapped = _preserve_parent_horizontal_bounds(mapped, unit.bbox_xyxy_norm)
        crop = _crop_line_unit(
            page,
            mapped,
            unit,
            local_text_height=max(mapped[3] - mapped[1], 1e-9),
        )
        if crop is None:
            continue
        line_units.append(
            OCRUnit(
                unit_id=f"{unit.unit_id}:projection-line-{line_index}",
                kind=unit.kind,
                source="horizontal_projection_fallback",
                bbox_xyxy_norm=mapped,
                crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                crop_b64=crop.image_b64,
                crop_shape_hw=crop.shape_hw,
                reading_order=int(unit.reading_order) * 1000 + line_index,
                detector_score=unit.detector_score,
                label=unit.label,
                table_id=unit.table_id,
                cell_id=unit.cell_id,
                metadata={
                    **unit.metadata,
                    "line_detector_fallback": True,
                    "line_detector_fallback_method": "horizontal_projection",
                    "line_index": line_index,
                },
            )
        )
    return line_units


def _horizontal_ink_bands(image_b64: str) -> list[tuple[int, int]]:
    """Return conservative contiguous rows containing dark foreground ink."""
    if not image_b64:
        return []
    try:
        from PIL import Image

        value = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
        with Image.open(io.BytesIO(base64.b64decode(value))) as image:
            grayscale = image.convert("L")
            width, height = grayscale.size
            if width < 2 or height < 2:
                return []
            pixels = grayscale.load()
            # A low density threshold keeps thin antialiased glyphs while
            # avoiding isolated JPEG/background noise.
            min_ink = max(2, int(round(width * 0.006)))
            row_stats: list[tuple[int, int]] = []
            for y in range(height):
                ink_count = 0
                longest_run = 0
                current_run = 0
                for x in range(width):
                    if pixels[x, y] < 210:
                        ink_count += 1
                        current_run += 1
                        longest_run = max(longest_run, current_run)
                    else:
                        current_run = 0
                row_stats.append((ink_count, longest_run))
            active = [count >= min_ink for count, _ in row_stats]
    except Exception:  # noqa: BLE001 - malformed crop is page-local.
        return []

    bands: list[tuple[int, int]] = []
    index = 0
    gap_tolerance = max(1, int(round(height * 0.012)))
    while index < height:
        if not active[index]:
            index += 1
            continue
        start = index
        end = index
        gap = 0
        index += 1
        while index < height:
            if active[index]:
                end = index
                gap = 0
            else:
                gap += 1
                if gap > gap_tolerance:
                    break
            index += 1
        band_max_run = max(
            (row_stats[row][1] for row in range(start, end + 1)),
            default=0,
        )
        # Dotted leaders can activate many rows despite containing no real
        # glyph.  Require at least one short continuous stroke so projection
        # fallback does not turn a blank form field into fake OCR text.
        min_stroke = max(3, int(round(width * 0.012)))
        if (
            end - start + 1 >= max(2, int(round(height * 0.01)))
            and band_max_run >= min_stroke
        ):
            bands.append((start, end + 1))
        index = max(index, end + 1)
    return bands


def _table_markdown(cells: Sequence[Mapping[str, Any]]) -> str:
    """Rebuild a lightweight row/column table from cell geometry."""
    usable = [
        dict(cell)
        for cell in cells
        if isinstance(cell, Mapping) and str(cell.get("text") or "").strip()
    ]
    if not usable:
        return ""
    usable.sort(
        key=lambda cell: (
            (float(cell["bbox_xyxy_norm"][1]) + float(cell["bbox_xyxy_norm"][3])) / 2.0,
            float(cell["bbox_xyxy_norm"][0]),
        )
    )
    rows: list[list[dict[str, Any]]] = []
    for cell in usable:
        box = cell.get("bbox_xyxy_norm") or [0.0, 0.0, 1.0, 1.0]
        center_y = (float(box[1]) + float(box[3])) / 2.0
        height = max(1e-6, float(box[3]) - float(box[1]))
        target = next(
            (
                row
                for row in rows
                if abs(center_y - row[0]["_center_y"])
                <= 0.55 * max(height, row[0]["_height"])
            ),
            None,
        )
        item = {"_center_y": center_y, "_height": height, "cell": cell}
        if target is None:
            rows.append([item])
        else:
            target.append(item)
    rows.sort(key=lambda row: row[0]["_center_y"])
    rendered: list[str] = []
    for row in rows:
        row.sort(key=lambda item: float(item["cell"]["bbox_xyxy_norm"][0]))
        values = [
            str(item["cell"].get("text") or "").replace("|", "/").strip()
            for item in row
        ]
        rendered.append("| " + " | ".join(values) + " |")
    return "\n".join(rendered)


def _union_boxes(
    boxes: Sequence[Sequence[float]],
) -> tuple[float, float, float, float] | None:
    normalized = [clamp_bbox(box) for box in boxes]
    valid = [box for box in normalized if box is not None]
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


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


def _error(stage: str, error: Any) -> dict[str, Any]:
    if isinstance(error, str):
        return {"stage": stage, "type": "RuntimeError", "message": error}
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}
