# SPDX-License-Identifier: Apache-2.0

"""Backend transport tests for document-level Option 5 diagnostics."""

from __future__ import annotations

import pandas as pd

from nemo_retriever.service.services.job_tracker import JobTracker
from nemo_retriever.service.services.pipeline_executor import (

    _attach_pipeline_selector,
    _extract_pipeline_diagnostics,
)


def _diagnostics() -> dict[str, object]:
    return {
        "scope": "document",
        "language": "vietnamese",
        "page_count": 12,
        "probe_pages": [2, 5, 9, 10, 12],
        "cache_hits": 5,
    }


def test_pipeline_executor_extracts_document_metrics_before_result_retention() -> None:
    diagnostics = _diagnostics()
    frame = pd.DataFrame(
        [
            {"metadata": {"ocr_document_diagnostics": diagnostics}},
            {"metadata": {"ocr_timing": {"document": diagnostics}}},
        ]
    )

    extracted = _extract_pipeline_diagnostics(frame)

    assert extracted == diagnostics


def test_pipeline_executor_persists_request_selector_without_result_rows() -> None:
    diagnostics = _attach_pipeline_selector(
        None,
        {"ocr_pipeline": "pipeline-option5"},
    )

    assert diagnostics == {
        "scope": "document",
        "ocr_pipeline": "pipeline-option5",
    }


def test_job_tracker_keeps_document_metrics_when_result_data_is_not_retained() -> None:
    tracker = JobTracker()
    tracker.register_job("job", expected_documents=1, retain_results=False)
    tracker.register_document("doc", job_id="job")
    diagnostics = _diagnostics()

    tracker.mark_completed(
        "doc",
        result_rows=12,
        result_data=[{"text": "large OCR payload"}],
        pipeline_diagnostics=diagnostics,
    )

    record = tracker.get_document("doc")
    assert record is not None
    assert record.result_data is None
    assert record.pipeline_diagnostics == diagnostics
