# SPDX-License-Identifier: Apache-2.0

"""Pipeline 7: Page Elements + Ministral FP8 OCR.

This is the original Pipeline 7 shape used before the NVFP4 experiment:

* PDFium remains the native source for text-bearing native PDF pages;
* Page Elements supplies semantic text/title/table geometry and visual evidence;
* Ministral 3B FP8 reads semantic text/title/table crops and full-page fallbacks
  for OCR only;
* scan pages use one full-page Ministral crop as the primary page read, while
  detected table regions remain separate whole-table OCR crops;

The pipeline deliberately does not use Qwen/NVFP4, language probing, VietOCR,
or the hybrid native-visual router.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.adapters import (
    make_ministral_vlm_backend,
)
from nemo_retriever.common.modality.ocr.isolated.option6 import (
    OPTION6_CROP_BATCH_SIZE,
    OPTION6_CROP_IMAGE_FORMAT,
    OPTION6_CROP_JPEG_QUALITY,
    OPTION6_CROP_MAX_CONCURRENCY,
    OPTION6_DETECTOR_BATCH_SIZE,
    OPTION6_VLM_BATCH_SIZE,
    OPTION6_VLM_MAX_CONCURRENCY,
    Option6Config,
    Option6Pipeline,
)
from nemo_retriever.common.modality.ocr.isolated.contracts import page_value

OPTION7_SELECTOR = "pipeline-option7"
# Keep the established internal name so stored results and existing trace
# consumers remain compatible.  The user-facing strategy/diagnostics below
# identify this as the restored semantic pipeline.
OPTION7_PIPELINE_NAME = "option7_ministral_vlm"
OPTION7_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"

OPTION7_DETECTOR_BATCH_SIZE = 100
OPTION7_OCR_BATCH_SIZE = 100
OPTION7_VLM_BATCH_SIZE = OPTION6_VLM_BATCH_SIZE
OPTION7_MAX_REQUEST_WORKERS = OPTION6_VLM_MAX_CONCURRENCY
# Keep Pipeline 7's established cap independent of Pipeline 6's long-form
# OCR/table output budget.
OPTION7_MAX_OUTPUT_TOKENS = 3072


@dataclass(frozen=True)
class Option7Config(Option6Config):
    """Full semantic-crop configuration for Pipeline 7."""

    semantic_ocr: bool = True
    # Pipeline 7 deliberately uses the Page Elements table bbox directly.
    # Table Structure is not part of this route.
    table_structure: bool = False
    # Page Elements visual detections are retained as bbox evidence only. P7
    # does not make a visual crop or a visual-classification/OCR request.
    include_visual_regions: bool = False
    classify_visual_regions: bool = False
    ocr_visual_regions: bool = False
    visual_ocr_max_tokens: int = OPTION7_MAX_OUTPUT_TOKENS
    text_max_output_tokens: int = OPTION7_MAX_OUTPUT_TOKENS
    table_max_output_tokens: int = OPTION7_MAX_OUTPUT_TOKENS
    native_table_text: bool = False
    scan_full_page: bool = True
    scan_mask_layout: bool = True
    full_page_layout_fallback: bool = True


class Option7Pipeline(Option6Pipeline):
    """Run semantic Page Elements crops through Ministral."""

    pipeline_name = OPTION7_PIPELINE_NAME
    model_name = OPTION7_MODEL

    def __init__(
        self,
        vlm: Any,
        *,
        config: Option7Config | Option6Config | None = None,
    ) -> None:
        if config is None:
            normalized = Option7Config()
        elif isinstance(config, Option7Config):
            normalized = replace(
                config,
                table_structure=False,
                include_visual_regions=False,
                classify_visual_regions=False,
                ocr_visual_regions=False,
                visual_ocr_max_tokens=OPTION7_MAX_OUTPUT_TOKENS,
                text_max_output_tokens=OPTION7_MAX_OUTPUT_TOKENS,
                table_max_output_tokens=OPTION7_MAX_OUTPUT_TOKENS,
            )
        else:
            # Keep callers that still pass Option6Config-compatible settings
            # working while forcing Pipeline 7's semantic invariants. In
            # Keep the Pipeline 7 invariants even when a caller passes the
            # older Option6-compatible config shape.
            values = {
                field.name: getattr(config, field.name)
                for field in Option6Config.__dataclass_fields__.values()
                if hasattr(config, field.name)
            }
            values["table_structure"] = False
            values["include_visual_regions"] = False
            values["classify_visual_regions"] = False
            values["ocr_visual_regions"] = False
            values["visual_ocr_max_tokens"] = OPTION7_MAX_OUTPUT_TOKENS
            values["text_max_output_tokens"] = OPTION7_MAX_OUTPUT_TOKENS
            values["table_max_output_tokens"] = OPTION7_MAX_OUTPUT_TOKENS
            normalized = Option7Config(**values)
        super().__init__(vlm, vlm, config=normalized)

    def process_document(self, pages: Any, *, document_key: str | None = None):
        """Reuse the semantic planner and relabel its provenance as P7."""

        page_list = [page_value(page) for page in pages]
        outputs = super().process_document(page_list, document_key=document_key)
        diagnostics = dict(self.last_document_diagnostics or {})
        text_units = int(diagnostics.get("text_units", 0) or 0)
        table_regions = int(diagnostics.get("table_regions", 0) or 0)
        full_page_count = int(
            diagnostics.get("scan_full_page_pages", 0) or 0
        ) + int(diagnostics.get("layout_full_page_pages", 0) or 0)
        semantic_text_crop_count = max(0, text_units - full_page_count)
        request_count = int(diagnostics.get("vlm_request_count", 0) or 0)
        diagnostics.update(
            {
                "pipeline": OPTION7_SELECTOR,
                "pipeline_name": OPTION7_PIPELINE_NAME,
                "model": OPTION7_MODEL,
                "semantic_ocr": True,
                "semantic_text_crop_ocr": True,
                "semantic_text_crop_count": semantic_text_crop_count,
                "visual_ocr_enabled": False,
                "visual_classification_enabled": False,
                "visual_ocr_requests": 0,
                "visual_vlm_requests": 0,
                "table_structure_enabled": False,
                "table_structure_called": False,
                "table_structure_region_count": 0,
                "table_structure_request_count": 0,
                "table_structure_detection_count": 0,
                "table_structure_counts_by_label": {},
                "table_structure_cell_count": 0,
                "table_structure_row_count": 0,
                "table_structure_column_count": 0,
                "table_region_count": table_regions,
                "table_crop_count": table_regions,
                "page_elements_table_region_count": table_regions,
                "page_elements_enabled": True,
                "page_elements_detection_only": False,
                "page_elements_used_for_ocr": bool(
                    semantic_text_crop_count or table_regions or full_page_count
                ),
                "full_page_ocr": bool(full_page_count),
                "full_page_count": full_page_count,
                "full_page_scan_only": False,
                "line_detector_enabled": False,
                "language_probe": {"strategy": "disabled_single_backend"},
                "probe_pages": [],
                "probe_unit_count": 0,
                "request_count": request_count,
                "vlm_request_count": request_count,
                "ocr_strategy": "page_elements_semantic_text_title_plus_table_ministral_ocr_full_page_fallback",
                "layout_strategy": "page_elements_text_title_table",
                "bbox_granularity": "semantic_text_title_table_and_page",
            }
        )
        self.last_document_diagnostics = diagnostics

        for output in outputs:
            native_passthrough = (
                output.status == "skipped"
                and output.source == "native_passthrough"
            )
            output.pipeline = OPTION7_PIPELINE_NAME
            if not native_passthrough:
                output.source = OPTION7_PIPELINE_NAME
                output.model = OPTION7_MODEL
            output.timing["document"] = dict(diagnostics)
            for block in output.ocr_text_blocks:
                if not isinstance(block, dict):
                    continue
                block["source"] = OPTION7_PIPELINE_NAME
                block["model"] = OPTION7_MODEL
                block["ocr_mode"] = (
                    str(block.get("ocr_mode") or "")
                    or (
                        "full_page"
                        if block.get("bbox_xyxy_norm") == [0.0, 0.0, 1.0, 1.0]
                        else "semantic_crop"
                    )
                )
                provenance = dict(block.get("provenance") or {})
                provenance.update(
                    {
                        "backend": "ministral_vlm",
                        "selected_backend": "ministral_vlm",
                        "page_elements_enabled": True,
                        "table_structure_enabled": False,
                    }
                )
                block["provenance"] = provenance
            for table in output.tables:
                if not isinstance(table, dict):
                    continue
                table["source"] = OPTION7_PIPELINE_NAME
                table["model"] = OPTION7_MODEL
                provenance = dict(table.get("provenance") or {})
                provenance.update(
                    {
                        "backend": "ministral_vlm",
                        "selected_backend": "ministral_vlm",
                        "table_structure_enabled": False,
                        "semantic_ocr": True,
                    }
                )
                table["provenance"] = provenance
        return outputs


def make_option7_pipeline(
    endpoint: str,
    *,
    api_key: str | None = None,
    language: str | None = "auto",
    timeout_s: float = 120.0,
    batch_size: int = OPTION7_OCR_BATCH_SIZE,
    include_table_cells: bool = True,
    scan_page_fallback: bool = True,
) -> Option7Pipeline:
    """Build the server-owned semantic Pipeline 7 runner."""

    del include_table_cells  # P7 OCRs whole tables from Page Elements table bboxes.
    vlm = make_ministral_vlm_backend(
        endpoint,
        model=OPTION7_MODEL,
        api_key=api_key,
        timeout_s=timeout_s,
        batch_size=max(1, int(batch_size or OPTION7_VLM_BATCH_SIZE)),
        max_pool_workers=OPTION7_MAX_REQUEST_WORKERS,
        max_tokens=OPTION7_MAX_OUTPUT_TOKENS,
    )
    return Option7Pipeline(
        vlm,
        config=Option7Config(
            language=language or "auto",
            detector_batch_size=OPTION7_DETECTOR_BATCH_SIZE,
            crop_batch_size=max(1, int(OPTION7_OCR_BATCH_SIZE)),
            crop_max_concurrency=OPTION6_CROP_MAX_CONCURRENCY,
            crop_image_format=OPTION6_CROP_IMAGE_FORMAT,
            crop_jpeg_quality=OPTION6_CROP_JPEG_QUALITY,
            vlm_batch_size=max(1, int(batch_size or OPTION7_VLM_BATCH_SIZE)),
            text_max_output_tokens=OPTION7_MAX_OUTPUT_TOKENS,
            table_max_output_tokens=OPTION7_MAX_OUTPUT_TOKENS,
            request_timeout_s=float(timeout_s),
            scan_page_fallback=bool(scan_page_fallback),
            include_visual_regions=False,
            classify_visual_regions=False,
            ocr_visual_regions=False,
            visual_ocr_max_tokens=OPTION7_MAX_OUTPUT_TOKENS,
            native_table_text=False,
            scan_full_page=True,
            scan_mask_layout=True,
            full_page_layout_fallback=True,
            semantic_ocr=True,
            table_structure=False,
        ),
    )


__all__ = [
    "OPTION7_DETECTOR_BATCH_SIZE",
    "OPTION7_MAX_OUTPUT_TOKENS",
    "OPTION7_MAX_REQUEST_WORKERS",
    "OPTION7_MODEL",
    "OPTION7_OCR_BATCH_SIZE",
    "OPTION7_PIPELINE_NAME",
    "OPTION7_SELECTOR",
    "OPTION7_VLM_BATCH_SIZE",
    "Option7Config",
    "Option7Pipeline",
    "make_option7_pipeline",
]
