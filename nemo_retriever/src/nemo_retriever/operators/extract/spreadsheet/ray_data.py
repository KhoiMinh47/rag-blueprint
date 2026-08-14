# SPDX-License-Identifier: Apache-2.0

"""Ray Data adapter for native XLSX/XLS/CSV extraction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.common.modality.spreadsheet import spreadsheet_bytes_to_chunks_df
from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


@designer_component(
    name="Spreadsheet Native Extractor",
    category="Document Processing",
    compute="cpu",
    description="Reads XLSX, XLS and CSV natively into sheet/range-aware Markdown rows",
    category_color="#64b4ff",
)
class SpreadsheetExtractActor(AbstractOperator, CPUOperator):
    """Extract workbook cells and CSV records without OCR."""

    def __init__(self, params: TextChunkParams | None = None) -> None:
        super().__init__()
        self._params = params or TextChunkParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame(columns=["text", "content", "path", "page_number", "metadata"])
        outputs: list[pd.DataFrame] = []
        for _, row in data.iterrows():
            path = str(row.get("path") or "")
            payload = row.get("bytes")
            if payload is None:
                with open(path, "rb") as handle:
                    payload = handle.read()
            if not isinstance(payload, (bytes, bytearray)):
                continue
            outputs.append(spreadsheet_bytes_to_chunks_df(bytes(payload), path, self._params))
        non_empty = [frame for frame in outputs if isinstance(frame, pd.DataFrame) and not frame.empty]
        return pd.concat(non_empty, ignore_index=True, sort=False) if non_empty else pd.DataFrame()

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        return self.run(batch_df)
