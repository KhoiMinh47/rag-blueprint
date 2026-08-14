# SPDX-License-Identifier: Apache-2.0

"""Ray operator for the document-scoped Option 5 OCR coordinator."""

from __future__ import annotations

from typing import Any

from nemo_retriever.common.modality.ocr.isolated.runtime import run_isolated_ocr_batch
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


class Option5DocumentOCRActor(AbstractOperator, CPUOperator):
    """Run Option 5 over one globally batched collection of document pages.

    ``RayDataExecutor`` sees the marker and creates a single input block.  The
    coordinator then groups pages by ``metadata.source_path`` internally, so
    multiple documents in the same graph batch remain isolated while every
    document gets one language probe and one logical OCR pass.
    """

    REQUIRES_GLOBAL_BATCH = True
    # source_id is page-scoped in PDFSplitActor; grouping by it would defeat
    # document coordination.  An empty tuple means one global block, followed
    # by explicit grouping in ``run_isolated_ocr_batch``.
    GLOBAL_BATCH_GROUP_KEYS: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline_kwargs = dict(kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return run_isolated_ocr_batch(data, **self._pipeline_kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


__all__ = ["Option5DocumentOCRActor"]
