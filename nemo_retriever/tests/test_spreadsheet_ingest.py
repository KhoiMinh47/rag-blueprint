# SPDX-License-Identifier: Apache-2.0

"""Tests for native spreadsheet extraction and file routing."""

from __future__ import annotations

from io import BytesIO

import pytest

from nemo_retriever.common.modality.spreadsheet import spreadsheet_bytes_to_chunks_df
from nemo_retriever.common.input_files import INPUT_TYPE_EXTENSIONS, input_type_for_path
from nemo_retriever.service.utils.file_type import FileCategory, FileClassifier, infer_extraction_mode_from_filename


def test_spreadsheet_extensions_route_to_one_family() -> None:
    assert {input_type_for_path(name) for name in ("a.xlsx", "b.xls", "c.csv")} == {"spreadsheet"}
    assert {"xlsx", "xls", "csv"}.issubset(INPUT_TYPE_EXTENSIONS["spreadsheet"])
    assert FileClassifier.SUFFIX_MAP[".xlsx"][0] is FileCategory.SPREADSHEET
    assert infer_extraction_mode_from_filename("table.csv") == "spreadsheet"


def test_csv_native_parser_preserves_semicolon_and_image_uri() -> None:
    payload = b"name;photo;value\nA;data:image/png;base64,AAA;1\nB;;2\n"
    frame = spreadsheet_bytes_to_chunks_df(payload, "items.csv")

    assert len(frame) == 1
    metadata = frame.iloc[0]["metadata"]["content_metadata"]
    assert metadata["source_type"] == "native_csv"
    assert metadata["delimiter"] == ";"
    assert metadata["image_columns"] == ["photo"]
    assert "data:image/png;base64,AAA" in frame.iloc[0]["text"]


def test_xlsx_parser_preserves_sheet_range_and_formula() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["Product", "Total"])
    sheet.append(["A", "=SUM(1,2)"])
    second = workbook.create_sheet("Summary")
    second["B2"] = "Done"

    payload = BytesIO()
    workbook.save(payload)
    frame = spreadsheet_bytes_to_chunks_df(payload.getvalue(), "report.xlsx")

    native = frame[frame["metadata"].map(lambda value: value["content_metadata"]["source_type"] == "native_cell")]
    assert {item["metadata"]["content_metadata"]["sheet_name"] for _, item in native.iterrows()} == {
        "Sales",
        "Summary",
    }
    sales = next(
        item for _, item in native.iterrows() if item["metadata"]["content_metadata"]["sheet_name"] == "Sales"
    )
    assert sales["metadata"]["content_metadata"]["range"] == "A1:B2"
    assert "Total" in sales["text"]
    assert sales["metadata"]["content_metadata"]["formula_cells"] == {"B2": "=SUM(1,2)"}
