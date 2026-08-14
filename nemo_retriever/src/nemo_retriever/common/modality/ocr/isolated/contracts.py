# SPDX-License-Identifier: Apache-2.0

"""Data contracts for the opt-in OCR pipelines.

These contracts deliberately do not depend on Ray, pandas, the default OCR
operator, or the service router.  A future selector can adapt one page row to
``OCRPage`` and pass the resulting dictionaries to the existing clean/chunk
consumer without changing those stages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

BBox = tuple[float, float, float, float]


def _as_bbox(value: Sequence[float] | None) -> BBox | None:
    """Convert a four-value sequence to an ordered normalized bbox."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= item <= 1.0 for item in (x0, y0, x1, y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


@dataclass(frozen=True)
class OCRPage:
    """One page prepared by the existing PDFium/Page Elements stages."""

    page_number: int | None
    image_b64: str
    page_elements_v3: Mapping[str, Any] | None = None
    table_structure_v1: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    native_text: str = ""
    source_id: str | None = None
    # Temporary PDFium character geometry stays out of metadata/result rows.
    # Option 2 uses it to populate native table cells without invoking OCR.
    native_spans: list[dict[str, Any]] | None = field(default=None, repr=False)

    @classmethod
    def from_row(cls, row: Any) -> OCRPage:
        """Adapt a dataframe-like row without importing pandas.

        The current PDF path stores the page raster in ``page_image`` and the
        upstream detections in ``page_elements_v3``/``table_structure_v1``.
        Mapping inputs and attribute-based rows are both accepted for future
        adapters and for small unit tests.
        """
        page_image = _mapping_value(row, "page_image", {}) or {}
        image_b64 = _mapping_value(row, "image_b64", "")
        if not image_b64 and isinstance(page_image, Mapping):
            image_b64 = page_image.get("image_b64", "")
        metadata = _mapping_value(row, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        page_number = _mapping_value(row, "page_number")
        try:
            page_number = int(page_number) if page_number is not None else None
        except (TypeError, ValueError):
            page_number = None
        page_elements = _mapping_value(row, "page_elements_v3")
        table_structure = _mapping_value(row, "table_structure_v1")
        native_text = _mapping_value(
            row, "native_text", _mapping_value(row, "text", "")
        )
        native_spans = _mapping_value(row, "_native_text_spans")
        return cls(
            page_number=page_number,
            image_b64=str(image_b64 or ""),
            page_elements_v3=page_elements
            if isinstance(page_elements, Mapping)
            else None,
            table_structure_v1=table_structure
            if isinstance(table_structure, Mapping)
            else None,
            metadata=dict(metadata),
            native_text=str(native_text or ""),
            source_id=str(_mapping_value(row, "source_id"))
            if _mapping_value(row, "source_id")
            else None,
            native_spans=[dict(item) for item in native_spans if isinstance(item, Mapping)]
            if isinstance(native_spans, list)
            else None,
        )


@dataclass
class OCRUnit:
    """A shared crop sent to one or more OCR backends."""

    unit_id: str
    kind: str
    source: str
    bbox_xyxy_norm: BBox
    crop_bbox_xyxy_norm: BBox
    crop_b64: str = field(repr=False)
    crop_shape_hw: tuple[int, int] = (0, 0)
    reading_order: int = 0
    detector_score: float | None = None
    label: str | None = None
    table_id: str | None = None
    cell_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRCandidate:
    """One canonicalizable recognition line."""

    text: str
    bbox_xyxy_norm: BBox
    score: float | None
    source: str
    model: str
    language: str | None = None
    content_type: str = "text"
    reading_order: int = 0
    unit_id: str | None = None
    table_id: str | None = None
    cell_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        score = None if self.score is None else float(max(0.0, min(1.0, self.score)))
        value: dict[str, Any] = {
            "text": self.text,
            "bbox_xyxy_norm": list(self.bbox_xyxy_norm),
            "score": score,
            "confidence": score,
            "source": self.source,
            "model": self.model,
            "language": self.language,
            "content_type": self.content_type,
            "reading_order": int(self.reading_order),
            "unit_id": self.unit_id,
            "provenance": dict(self.provenance),
        }
        if self.candidates:
            value["candidates"] = list(self.candidates)
        if self.table_id is not None:
            value["table_id"] = self.table_id
        if self.cell_id is not None:
            value["cell_id"] = self.cell_id
        return value


@dataclass
class OCRPageOutput:
    """Serializable output contract shared by the isolated OCR pipelines."""

    pipeline: str
    text: str = ""
    ocr_text_blocks: list[dict[str, Any]] = field(default_factory=list)
    bbox_xyxy_norm: BBox | None = None
    score: float | None = None
    confidence: float | None = None
    source: str = ""
    model: str = ""
    language: str | None = None
    tables: list[dict[str, Any]] = field(default_factory=list)
    visuals: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "text": self.text,
            "ocr_text_blocks": list(self.ocr_text_blocks),
            "bbox_xyxy_norm": list(self.bbox_xyxy_norm)
            if self.bbox_xyxy_norm
            else None,
            "score": self.score,
            "confidence": self.confidence,
            "source": self.source,
            "model": self.model,
            "language": self.language,
            "tables": list(self.tables),
            "visuals": list(self.visuals),
            "candidates": list(self.candidates),
            "errors": list(self.errors),
            "timing": dict(self.timing),
            "status": self.status,
        }


def page_value(value: OCRPage | Mapping[str, Any] | Any) -> OCRPage:
    """Normalize a page-like input for both isolated pipeline entry points."""
    return value if isinstance(value, OCRPage) else OCRPage.from_row(value)
