# SPDX-License-Identifier: Apache-2.0

"""Graph adapter for the opt-in OCR pipeline implementations.

The Option 3/4/5/6/7 implementations intentionally live outside the existing OCR
operator.  This module is the only bridge from a graph dataframe to those
implementations.  The graph imports it only for an explicit request selector;
the normal Option 1/2 OCR operator is left on its existing code path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.adapters import (
    HTTPDetectorBackend,
    make_nemotron_backend,
    make_tesseract_backend,
    make_vietnamese_recognizer,
)
from nemo_retriever.common.modality.ocr.isolated.contracts import (
    OCRPageOutput,
)
from nemo_retriever.common.modality.ocr.isolated.option3 import (
    Option3Config,
    Option3Pipeline,
)
from nemo_retriever.common.modality.ocr.isolated.option4 import (
    Option4Config,
    Option4Pipeline,
)
from nemo_retriever.common.modality.ocr.isolated.option5 import (
    OPTION5_LINE_DETECTOR_BATCH_SIZE,
    OPTION5_LINE_DETECTOR_ENABLED,
    OPTION5_LINE_DETECTOR_MAX_POOL_WORKERS,
    OPTION5_MAX_REQUEST_WORKERS,
    OPTION5_OCR_BATCH_SIZE,
    Option5Config,
    Option5Pipeline,
    _document_key_for_row,
    option5_line_detector_endpoint,
    option5_vietnamese_endpoint,
)
from nemo_retriever.common.modality.ocr.isolated.option6 import (
    OPTION6_CROP_BATCH_SIZE,
    OPTION6_SELECTOR,
    OPTION6_VLM_BATCH_SIZE,
    Option6Pipeline,
    make_option6_pipeline,
)
from nemo_retriever.common.modality.ocr.isolated.option7 import (
    OPTION7_OCR_BATCH_SIZE,
    OPTION7_SELECTOR,
    Option7Pipeline,
    make_option7_pipeline,
)

OPTION3_SELECTOR = "pipeline-option3"
OPTION4_SELECTOR = "pipeline-option4"
OPTION5_SELECTOR = "pipeline-option5"
ISOLATED_SELECTORS = frozenset(
    {
        OPTION3_SELECTOR,
        OPTION4_SELECTOR,
        OPTION5_SELECTOR,
        OPTION6_SELECTOR,
        OPTION7_SELECTOR,
    }
)


def run_isolated_ocr_batch(
    batch_df: Any,
    *,
    ocr_pipeline: str,
    line_detector_invoke_url: str | None = None,
    ocr_recognizer_invoke_url: str | None = None,
    ocr_invoke_url: str | None = None,
    vietnamese_ocr_invoke_url: str | None = None,
    vintern_ocr_invoke_url: str | None = None,
    ministral_vlm_invoke_url: str | None = None,
    tesseract_ocr_invoke_url: str | None = None,
    api_key: str | None = None,
    ocr_api_key: str | None = None,
    ocr_lang: str | None = None,
    inference_batch_size: int = 8,
    request_timeout_s: float = 120.0,
    scan_ocr_fallback: bool = True,
    scan_ocr_tile_size: int = 1024,
    scan_ocr_tile_overlap: float = 0.15,
    extract_text: bool = True,
    extract_tables: bool = True,
    _runner: Any | None = None,
) -> Any:
    """Run an explicit Option 3/4/5/6/7 pipeline over graph page rows.

    Each page is adapted to the isolated ``OCRPage`` contract and then
    converted back to the columns consumed by the existing clean/chunk
    stages.  Table-cell blocks stay inside ``table[*].cells`` and are not also
    placed in ``_ocr_text_blocks``; this prevents the generic content
    exploder from emitting a second copy of every cell.
    """

    import pandas as pd

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    if ocr_pipeline not in ISOLATED_SELECTORS:
        raise ValueError(f"Unsupported isolated OCR selector: {ocr_pipeline!r}")

    # Pipeline 6's streaming Ray actor passes a persistent runner here so its
    # eight HTTP workers and keep-alive connections survive across ready PDF
    # blocks.  All other callers keep the original one-runner-per-call path.
    runner = _runner or _build_runner(
        ocr_pipeline=ocr_pipeline,
        line_detector_invoke_url=line_detector_invoke_url,
        ocr_recognizer_invoke_url=ocr_recognizer_invoke_url,
        ocr_invoke_url=ocr_invoke_url,
        vietnamese_ocr_invoke_url=vietnamese_ocr_invoke_url,
        vintern_ocr_invoke_url=vintern_ocr_invoke_url,
        ministral_vlm_invoke_url=ministral_vlm_invoke_url,
        tesseract_ocr_invoke_url=tesseract_ocr_invoke_url,
        api_key=ocr_api_key or api_key,
        ocr_lang=ocr_lang,
        inference_batch_size=inference_batch_size,
        request_timeout_s=request_timeout_s,
        scan_ocr_fallback=scan_ocr_fallback,
        scan_ocr_tile_size=scan_ocr_tile_size,
        scan_ocr_tile_overlap=scan_ocr_tile_overlap,
        extract_tables=extract_tables,
    )

    source_rows = [source_row.to_dict() for _, source_row in batch_df.iterrows()]
    if ocr_pipeline in {OPTION5_SELECTOR, OPTION6_SELECTOR, OPTION7_SELECTOR}:
        # Options 5, 6, and 7 are document-scoped: source_id is page-scoped in the PDF
        # splitter, so group by metadata.source_path and let the coordinator
        # perform one global Nemotron/VietOCR pass per document.
        page_outputs: list[OCRPageOutput | None] = [None] * len(source_rows)
        grouped: dict[str, list[int]] = {}
        for index, row in enumerate(source_rows):
            grouped.setdefault(_document_key_for_row(row, index), []).append(index)
        for document_key, indices in grouped.items():
            try:
                document_outputs = runner.process_document(
                    [source_rows[index] for index in indices],
                    document_key=document_key,
                )
            except Exception as exc:  # noqa: BLE001 - preserve page batch shape
                document_outputs = [
                    OCRPageOutput(
                        pipeline=runner.pipeline_name,
                        source="isolated_runtime",
                        model=getattr(runner, "model_name", "Nemotron OCR v2"),
                        errors=[
                            {
                                "stage": "isolated_ocr_runtime.document",
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        ],
                        status="failed",
                    )
                    for _ in indices
                ]
            for index, page_output in zip(indices, document_outputs):
                page_outputs[index] = page_output
        page_outputs = [
            output
            if output is not None
            else OCRPageOutput(
                pipeline=runner.pipeline_name,
                source="isolated_runtime",
                model=getattr(runner, "model_name", "Nemotron OCR v2"),
                errors=[
                    {
                        "stage": "isolated_ocr_runtime",
                        "type": "RuntimeError",
                        "message": "missing document output",
                    }
                ],
                status="failed",
            )
            for output in page_outputs
        ]
    elif ocr_pipeline == OPTION3_SELECTOR:
        # Option 3 owns bounded page-parallel execution and preserves the
        # dataframe order through Option3Pipeline.process_pages().
        if hasattr(runner, "process_pages"):
            page_outputs = runner.process_pages(source_rows)
        else:
            # Keep small injected test doubles and legacy adapters compatible;
            # the production Option 3 pipeline still takes its parallel path.
            page_outputs = [runner.process_page(row) for row in source_rows]
    else:
        # Keep Option 4/5's existing page-local error behavior unchanged.
        page_outputs = []
        for row in source_rows:
            try:
                page_output = runner.process_page(row)
            except Exception as exc:  # noqa: BLE001 - page-local failure; preserve batch
                page_output = OCRPageOutput(
                    pipeline=(
                        getattr(runner, "pipeline_name", "option5_nemotron_language_routed_vietnamese_ocr")
                        if ocr_pipeline in {OPTION5_SELECTOR, OPTION7_SELECTOR}
                        else "option4_parallel_nemotron_tesseract_fusion"
                    ),
                    source="isolated_runtime",
                    errors=[
                        {
                            "stage": "isolated_ocr_runtime",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                    status="failed",
                )
            page_outputs.append(page_output)

    output_rows = [
        _apply_page_output(
            row,
            page_output,
            selector=ocr_pipeline,
            extract_text=extract_text,
            extract_tables=extract_tables,
        )
        for row, page_output in zip(source_rows, page_outputs)
    ]
    return pd.DataFrame(output_rows)


def _build_runner(
    *,
    ocr_pipeline: str,
    line_detector_invoke_url: str | None,
    ocr_recognizer_invoke_url: str | None,
    ocr_invoke_url: str | None,
    tesseract_ocr_invoke_url: str | None,
    vietnamese_ocr_invoke_url: str | None = None,
    vintern_ocr_invoke_url: str | None = None,
    ministral_vlm_invoke_url: str | None = None,
    api_key: str | None,
    ocr_lang: str | None,
    inference_batch_size: int,
    request_timeout_s: float,
    scan_ocr_fallback: bool,
    scan_ocr_tile_size: int,
    scan_ocr_tile_overlap: float,
    extract_tables: bool,
) -> Option3Pipeline | Option4Pipeline | Option5Pipeline | Option6Pipeline | Option7Pipeline:
    # Option 4 is intentionally a single server-owned strategy: Vietnamese
    # Tesseract first, then Nemotron when Tesseract fails or scores below the
    # configured threshold.  Do not let an old/client-supplied language value
    # silently select a different sidecar configuration.
    if ocr_pipeline == OPTION4_SELECTOR:
        language = "auto"
        tesseract_language = "vie"
        language_probe_language = "vie+eng"
    else:
        # Option 3 routes raw Nemotron text per candidate.  Do not expose the
        # old Tesseract-oriented default (``vie``) as if it controlled this
        # branch; the page output is automatic while candidate provenance
        # records vi/en/uncertain decisions.
        if ocr_pipeline in {
            OPTION3_SELECTOR,
            OPTION5_SELECTOR,
            OPTION6_SELECTOR,
            OPTION7_SELECTOR,
        }:
            # Option 3 and Option 5 are both language-routed pipelines.  An
            # omitted ``ocr_lang`` must remain automatic; mapping ``None`` to
            # ``vie`` here would silently force every Option 5 document down
            # the Vietnamese fast path and bypass its five-page probe.
            language = str(ocr_lang or "auto")
        else:
            language = _ocr_language(ocr_lang)
        tesseract_language = _tesseract_language(ocr_lang)
        language_probe_language = None
    batch_size = max(1, int(inference_batch_size or 1))
    option5_batch_size = max(OPTION5_OCR_BATCH_SIZE, batch_size)
    timeout = max(1.0, float(request_timeout_s or 120.0))

    if ocr_pipeline == OPTION6_SELECTOR:
        endpoint = str(vintern_ocr_invoke_url or "").strip()
        if not endpoint:
            raise ValueError(
                "pipeline-option6 requires the server-owned Qwen 3.5 "
                "endpoint (vintern_ocr_invoke_url)"
            )
        return make_option6_pipeline(
            endpoint,
            api_key=api_key,
            language=language,
            timeout_s=timeout,
            # Keep this hard bounded at the vLLM admission target.  The
            # adapter creates one persistent pool and vLLM performs the
            # continuous batching itself.
            batch_size=OPTION6_VLM_BATCH_SIZE,
            crop_batch_size=OPTION6_CROP_BATCH_SIZE,
            scan_page_fallback=bool(scan_ocr_fallback),
        )

    if ocr_pipeline == OPTION7_SELECTOR:
        if not str(ministral_vlm_invoke_url or "").strip():
            raise ValueError(
                "pipeline-option7 requires the server-owned Ministral VLM "
                "endpoint (ministral_vlm_invoke_url)"
            )
        return make_option7_pipeline(
            str(ministral_vlm_invoke_url),
            api_key=api_key,
            language=language,
            timeout_s=timeout,
            batch_size=max(OPTION7_OCR_BATCH_SIZE, batch_size),
            scan_page_fallback=bool(scan_ocr_fallback),
        )

    if ocr_pipeline in {OPTION3_SELECTOR, OPTION5_SELECTOR}:
        if not ocr_invoke_url or not vietnamese_ocr_invoke_url:
            raise ValueError(
                f"{ocr_pipeline} requires Nemotron OCR and Vietnamese recognizer endpoints"
            )
        backend_batch_size = (
            option5_batch_size if ocr_pipeline == OPTION5_SELECTOR else batch_size
        )
        backend_pool_workers = (
            OPTION5_MAX_REQUEST_WORKERS
            if ocr_pipeline == OPTION5_SELECTOR
            else None
        )
        nemotron = make_nemotron_backend(
            ocr_invoke_url,
            api_key=api_key,
            timeout_s=timeout,
            batch_size=backend_batch_size,
            max_pool_workers=backend_pool_workers,
        )
        vietnamese_recognizer = make_vietnamese_recognizer(
            (
                option5_vietnamese_endpoint(vietnamese_ocr_invoke_url)
                if ocr_pipeline == OPTION5_SELECTOR
                else vietnamese_ocr_invoke_url
            ),
            api_key=api_key,
            timeout_s=timeout,
            batch_size=backend_batch_size,
            max_pool_workers=backend_pool_workers,
        )
        pipeline_class = Option5Pipeline if ocr_pipeline == OPTION5_SELECTOR else Option3Pipeline
        config_class = Option5Config if ocr_pipeline == OPTION5_SELECTOR else Option3Config
        config_kwargs: dict[str, Any] = {
            "language": language,
            "scan_page_fallback": bool(scan_ocr_fallback),
            "batch_size": backend_batch_size,
            "include_table_cells": bool(extract_tables),
            "request_timeout_s": timeout,
        }
        if ocr_pipeline == OPTION5_SELECTOR:
            config_kwargs["max_request_workers"] = OPTION5_MAX_REQUEST_WORKERS
        if ocr_pipeline == OPTION5_SELECTOR:
            line_detector = (
                HTTPDetectorBackend(
                    endpoint=option5_line_detector_endpoint(
                        line_detector_invoke_url
                    ),
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
            config_kwargs["line_detection"] = line_detector is not None
            return Option5Pipeline(
                nemotron,
                vietnamese_recognizer,
                config=Option5Config(**config_kwargs),
                line_detector=line_detector,
            )
        return Option3Pipeline(
            nemotron,
            vietnamese_recognizer,
            config=Option3Config(**config_kwargs),
        )

    if not line_detector_invoke_url or not ocr_invoke_url or not tesseract_ocr_invoke_url:
        raise ValueError(
            "pipeline-option4 requires PP-OCRv6 line detector, Nemotron OCR, "
            "and Tesseract endpoints"
        )
    nemotron = make_nemotron_backend(
        ocr_invoke_url,
        api_key=api_key,
        language=language,
        timeout_s=timeout,
        batch_size=batch_size,
    )
    tesseract = make_tesseract_backend(
        tesseract_ocr_invoke_url,
        api_key=api_key,
        language=tesseract_language,
        psm=7,
        timeout_s=timeout,
        batch_size=batch_size,
    )
    language_probe = make_tesseract_backend(
        tesseract_ocr_invoke_url,
        api_key=api_key,
        language=language_probe_language or "vie+eng",
        psm=7,
        timeout_s=timeout,
        batch_size=batch_size,
    )
    line_detector = (
        HTTPDetectorBackend(
            endpoint=line_detector_invoke_url,
            api_key=api_key,
            timeout_s=timeout,
            batch_size=batch_size,
        )
        if line_detector_invoke_url
        else None
    )
    return Option4Pipeline(
        nemotron,
        tesseract,
        line_detector=line_detector,
        language_probe=language_probe,
        config=Option4Config(
            language=language,
            tesseract_language=tesseract_language,
            language_probe_language=language_probe_language or "vie+eng",
            language_routing=ocr_pipeline == OPTION4_SELECTOR,
            tesseract_psm=7,
            tesseract_first=True,
            include_table_cells=bool(extract_tables),
            scan_page_fallback=bool(scan_ocr_fallback),
            request_timeout_s=timeout,
        ),
    )


def _ocr_language(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "english": "eng",
        "vietnamese": "vie",
        "multi": "vie+eng",
    }.get(normalized, "vie")


def _tesseract_language(value: str | None) -> str:
    """Use Vietnamese-only Tesseract by default; preserve explicit modes."""
    normalized = str(value or "").strip().lower()
    return {
        "english": "eng",
        "vietnamese": "vie",
        "multi": "eng+vie",
    }.get(normalized, "vie")


def _apply_page_output(
    row: dict[str, Any],
    output: OCRPageOutput,
    *,
    selector: str,
    extract_text: bool,
    extract_tables: bool,
) -> dict[str, Any]:
    """Adapt the canonical page result to existing consumer columns."""

    metadata = row.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata.update(
        {
            "ocr_pipeline": selector,
            "ocr_source": output.source,
            "ocr_model": output.model,
            "ocr_language": output.language,
            "ocr_status": output.status,
            "ocr_timing": dict(output.timing),
        }
    )
    if selector in {
        OPTION3_SELECTOR,
        OPTION5_SELECTOR,
        OPTION6_SELECTOR,
        OPTION7_SELECTOR,
    }:
        metadata["ocr_pipeline_name"] = output.pipeline
    document_diagnostics = output.timing.get("document")
    if selector in {OPTION5_SELECTOR, OPTION6_SELECTOR, OPTION7_SELECTOR} and isinstance(
        document_diagnostics, Mapping
    ):
        metadata["ocr_document_diagnostics"] = dict(document_diagnostics)
    if output.errors:
        metadata["ocr_errors"] = list(output.errors)
    row["metadata"] = metadata

    blocks = [
        block
        for block in output.ocr_text_blocks
        if isinstance(block, Mapping)
        and str(block.get("content_type") or "text") != "table_cell"
    ]
    cell_blocks = [
        block
        for block in output.ocr_text_blocks
        if isinstance(block, Mapping)
        and str(block.get("content_type") or "text") == "table_cell"
    ]

    # A native PDF page is deliberately a no-op for text.  The selected
    # pipeline is recorded for traceability, but native PDFium text, spans,
    # and any upstream structured columns remain untouched.
    if output.status == "skipped" and output.source == "native_passthrough":
        row["ocr"] = _ocr_metadata(output, selector, block_count=0, cell_count=0)
        return row

    # Pipelines 6/7 may preserve native PDFium text while adding only the
    # Page-Elements blocks that had no native character geometry.  When that
    # happens, hand the downstream cleaner one native block plus the missing
    # VLM blocks. Keep the per-character column until ``clean_content_rows``:
    # it suppresses native characters inside authoritative structured regions
    # before the content exploder runs.
    # A layout fallback is a full-page VLM replacement only for a weak native
    # page. Do not merge the same PDFium fragment with the page response, and
    # do not let the cleaner later suppress that fragment by a detector bbox.
    full_page_primary = selector in {OPTION6_SELECTOR, OPTION7_SELECTOR} and bool(
        output.timing.get("full_page_primary")
    )
    if full_page_primary:
        native_text = str(row.get("text") or "").strip()
        if native_text:
            metadata["native_text_before_full_page"] = len(native_text)
        row.pop("_native_text_spans", None)
        row.pop("_native_text_blocks", None)
        row["raw_text"] = output.text.strip()
    native_merge = (
        selector in {OPTION6_SELECTOR, OPTION7_SELECTOR}
        and bool(output.timing.get("native_page"))
        and not full_page_primary
    )
    if native_merge and blocks:
        native_text = str(row.get("text") or "").strip()
        if native_text:
            row["raw_text"] = native_text
            row["_native_text_blocks"] = [
                {
                    "text": native_text,
                    "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                    "source": "pdfium_native",
                    "model": "PDFium native text",
                    "content_type": "text",
                    "reading_order": -1,
                }
            ] + list(blocks)

    # Pipeline 7 emits semantic text/table OCR alongside native PDFium text.
    # A geometry-free native source still needs a fallback native block;
    # otherwise ``_ocr_text_blocks`` would take precedence in the exploder and
    # hide the native page text.  A page-level layout fallback is authoritative
    # and therefore deliberately bypasses this native merge.
    option7_native_merge = (
        selector == OPTION7_SELECTOR
        and bool(output.timing.get("native_page"))
        and not full_page_primary
    )
    if option7_native_merge:
        native_text = str(row.get("text") or "").strip()
        if native_text:
            row["raw_text"] = native_text
    if option7_native_merge and not isinstance(row.get("_native_text_spans"), list):
        native_text = str(row.get("raw_text") or "").strip()
        if native_text:
            row["_native_text_blocks"] = [
                {
                    "text": native_text,
                    "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                    "source": "pdfium_native",
                    "model": "PDFium native text",
                    "content_type": "text",
                    "reading_order": -1,
                }
            ]

    row["_ocr_text_blocks"] = list(blocks)
    row["ocr_text_blocks"] = list(output.ocr_text_blocks)
    if extract_text:
        # An isolated OCR failure must not leak stale text from an upstream
        # placeholder/disabled OCR path.  Native PDF text is handled by the
        # passthrough branch above and remains untouched.
        row["text"] = output.text.strip()

    if extract_tables:
        row["table"] = _consumer_tables(output.tables)
    if selector in {OPTION6_SELECTOR, OPTION7_SELECTOR}:
        # Option 6 may retain visual crops. Pipeline 7 disables visual crop
        # creation, so this list is empty there while Page Elements' raw bbox
        # evidence remains available to the dashboard sidecar.
        row["images"] = [
            dict(item) for item in output.visuals if isinstance(item, Mapping)
        ]
    row["ocr"] = _ocr_metadata(
        output,
        selector,
        block_count=len(blocks),
        cell_count=len(cell_blocks),
        blocks=blocks,
        cells=cell_blocks,
    )
    row["ocr_v1_num_detections"] = len(blocks) + len(cell_blocks)
    row["ocr_v1_counts_by_label"] = _counts_by_label(blocks, cell_blocks)
    return row


def _ocr_metadata(
    output: OCRPageOutput,
    selector: str,
    *,
    block_count: int,
    cell_count: int,
    blocks: list[Mapping[str, Any]] | None = None,
    cells: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    value = {
        "pipeline": selector,
        "source": output.source,
        "model": output.model,
        "language": output.language,
        "score": output.score,
        "confidence": output.confidence,
        "status": output.status,
        "num_detections": block_count + cell_count,
        "counts_by_label": _counts_by_label(blocks or [], cells or []),
        "candidates": list(output.candidates),
        "errors": list(output.errors),
        "timing": dict(output.timing),
        "output": {
            "bbox_xyxy_norm": output.to_dict().get("bbox_xyxy_norm"),
            "block_count": block_count,
            "table_cell_count": cell_count,
        },
    }
    if selector in {
        OPTION3_SELECTOR,
        OPTION5_SELECTOR,
        OPTION6_SELECTOR,
        OPTION7_SELECTOR,
    }:
        value["pipeline_name"] = output.pipeline
    return value


def _consumer_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the table text field expected by ``explode_content_to_rows``.

    Cell-level candidates remain nested under ``cells`` for geometry and
    provenance.  The single table text is only a consumer summary, so the
    generic content exploder emits one table row rather than one additional
    row per nested cell.
    """
    result: list[dict[str, Any]] = []
    for table in tables:
        if not isinstance(table, Mapping):
            continue
        item = dict(table)
        cells = [cell for cell in item.get("cells") or [] if isinstance(cell, Mapping)]
        if item.get("table_text_format") != "markdown":
            text = "\n".join(str(cell.get("text") or "").strip() for cell in cells).strip()
            if text:
                item["text"] = text
        result.append(item)
    return result


def _counts_by_label(
    blocks: list[Mapping[str, Any]], cells: list[Mapping[str, Any]]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in blocks:
        label = str(block.get("content_type") or "text")
        counts[label] = counts.get(label, 0) + 1
    if cells:
        counts["table_cell"] = len(cells)
    if counts.get("title"):
        counts["text"] = counts.get("text", 0) + counts["title"]
    return counts


__all__ = [
    "ISOLATED_SELECTORS",
    "OPTION3_SELECTOR",
    "OPTION4_SELECTOR",
    "OPTION5_SELECTOR",
    "OPTION6_SELECTOR",
    "OPTION7_SELECTOR",
    "run_isolated_ocr_batch",
]
