# SPDX-License-Identifier: Apache-2.0

"""Ray operator for the document-scoped Pipeline 7 OCR coordinator."""

from __future__ import annotations

from typing import Any

from nemo_retriever.common.modality.ocr.isolated.runtime import run_isolated_ocr_batch
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


class Option7DocumentOCRActor(AbstractOperator, CPUOperator):
    """Run Pipeline 7 over one global collection of document pages."""

    REQUIRES_GLOBAL_BATCH = True
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


__all__ = ["Option7DocumentOCRActor"]
