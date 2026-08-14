import base64
import io
import json

import pandas as pd
from PIL import Image

from nemo_retriever.common.modality.content_transforms import clean_content_rows, chunk_pdf_content_rows, explode_content_to_rows
from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.common.modality.ocr import ppocr as ppocr_adapter
from nemo_retriever.common.modality.ocr import shared as ocr_shared
from nemo_retriever.common.modality.page_elements import shared as page_elements_shared
from nemo_retriever.common.modality import stamp_detection as stamp_shared
from nemo_retriever.service.services.visual_evidence import (
    build_visual_evidence,
    deduplicate_visual_evidence,
    manifest_without_images,
)


def _png_b64(width=100, height=80):
    image = Image.new("RGB", (width, height), (240, 240, 240))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _native_spans(text, *, y=0.64, x=0.20, step=0.012):
    spans = []
    for index, char in enumerate(text):
        if char in {"\r", "\n"}:
            spans.append({"char": char, "bbox_xyxy_norm": None})
            continue
        left = min(0.98, x + index * step)
        spans.append({"char": char, "bbox_xyxy_norm": [left, y, min(1.0, left + step * 0.8), y + 0.012]})
    return spans


def test_native_text_survives_overlapping_chart_bbox():
    frame = pd.DataFrame([
        {
            "path": "screenshot-like.pdf",
            "text": "Maecenas tincidunt est efficitur ligula euismod",
            "_native_text_spans": _native_spans("Maecenas tincidunt est efficitur ligula euismod"),
            "chart": [{"bbox_xyxy_norm": [0.254, 0.689, 0.785, 0.880]}],
            "metadata": {},
        }
    ])

    cleaned = clean_content_rows(frame).iloc[0]

    assert "Maecenas tincidunt" in cleaned["text"]
    assert cleaned["metadata"]["cleaning"]["suppressed_native_characters"] == 0
    assert cleaned["metadata"]["cleaning"]["native_visual_characters_suppressed"] == 0


def test_native_chart_text_is_removed_without_removing_neighbouring_paragraph():
    chart_text = "Row 1 Row 2 0 2 4 Column 1 Column 2"
    frame = pd.DataFrame([
        {
            "path": "chart.pdf",
            "text": f"Paragraph above\n{chart_text}",
            "_native_text_spans": _native_spans("Paragraph above", y=0.64)
            + [{"char": "\n", "bbox_xyxy_norm": None}]
            + _native_spans(chart_text, y=0.72, x=0.30),
            "chart": [{"bbox_xyxy_norm": [0.254, 0.689, 0.785, 0.880]}],
            "metadata": {},
        }
    ])

    cleaned = clean_content_rows(frame).iloc[0]

    assert "Paragraph above" in cleaned["text"]
    assert "Row 1" not in cleaned["text"]
    assert cleaned["metadata"]["cleaning"]["native_visual_characters_suppressed"] > 0


def test_native_text_is_suppressed_only_when_duplicate_table_text_matches():
    frame = pd.DataFrame([
        {
            "path": "table.pdf",
            "text": "Cell content",
            "_native_text_spans": _native_spans("Cell content"),
            "table": [{
                "bbox_xyxy_norm": [0.10, 0.60, 0.90, 0.90],
                "text": "| Cell content |",
            }],
            "metadata": {},
        }
    ])

    cleaned = clean_content_rows(frame).iloc[0]

    assert cleaned["text"] == ""
    assert cleaned["metadata"]["cleaning"]["suppressed_native_characters"] > 0


def test_native_blocks_stop_at_table_boundary_even_when_table_text_differs():
    frame = pd.DataFrame([
        {
            "path": "mixed.pdf",
            "text": "Above\nInside\nBelow",
            "_native_text_spans": _native_spans("Above", y=0.20)
            + [{"char": "\n", "bbox_xyxy_norm": None}]
            + _native_spans("Inside", y=0.45)
            + [{"char": "\n", "bbox_xyxy_norm": None}]
            + _native_spans("Below", y=0.72),
            "table": [{"bbox_xyxy_norm": [0.10, 0.30, 0.90, 0.65], "text": "| Other |"}],
            "metadata": {},
        }
    ])

    cleaned = clean_content_rows(frame).iloc[0]

    assert [block["text"] for block in cleaned["_native_text_blocks"]] == ["Above", "Inside", "Below"]
    assert [block["bbox_xyxy_norm"][1] for block in cleaned["_native_text_blocks"]] == [0.20, 0.45, 0.72]


def test_visual_evidence_keeps_text_and_records_chart_overlap():
    evidence = build_visual_evidence([
        {
            "page_number": 1,
            "text": "Caption remains text",
            "_bbox_xyxy_norm": [0.10, 0.60, 0.90, 0.66],
            "images": [{
                "label_name": "chart",
                "image_type": "detected_region",
                "bbox_xyxy_norm": [0.20, 0.636, 0.84, 0.93],
            }],
        }
    ])

    blocks = evidence["pages"][0]["blocks"]
    text_block = next(block for block in blocks if block["content_type"] == "text")
    chart_block = next(block for block in blocks if block["content_type"] == "chart")

    assert text_block["text"] == "Caption remains text"
    assert text_block["overlaps_regions"][0]["content_type"] == "chart"
    assert text_block["id"] in chart_block["contains_text_blocks"]


def test_visual_evidence_never_deduplicates_text_into_same_bbox_chart():
    evidence = build_visual_evidence([
        {
            "page_number": 1,
            "text": "Chart caption",
            "_bbox_xyxy_norm": [0.20, 0.30, 0.80, 0.60],
            "images": [{
                "label_name": "chart",
                "image_type": "detected_region",
                "bbox_xyxy_norm": [0.20, 0.30, 0.80, 0.60],
                "image_b64": _png_b64(),
            }],
        }
    ])
    blocks = evidence["pages"][0]["blocks"]
    assert {block["content_type"] for block in blocks} == {"text", "chart"}


def test_option6_visual_sidecar_drops_page_noise_but_keeps_standalone_crop():
    evidence = build_visual_evidence([
        {
            "page_number": 1,
            "text": "Native page text",
            "metadata": {"ocr_pipeline": "pipeline-option6"},
            "page_elements_v3": {
                "detections": [
                    {"label_name": "infographic", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
                    {"label_name": "chart", "bbox_xyxy_norm": [0.20, 0.30, 0.70, 0.60]},
                    {"label_name": "title", "bbox_xyxy_norm": [0.10, 0.08, 0.80, 0.12]},
                ]
            },
        }
    ])

    blocks = evidence["pages"][0]["blocks"]

    assert "infographic" not in {block["content_type"] for block in blocks}
    assert any(block["content_type"] == "chart" for block in blocks)
    assert not any(
        block["content_type"] in {"title", "header_footer"} and not block["text"]
        for block in blocks
    )


def test_page_elements_visual_bbox_keeps_raw_model_geometry():
    raw_bbox = [0.2545, 0.6894, 0.7846, 0.8797]
    processed = page_elements_shared._apply_page_elements_v3_postprocess([
        {"label_name": "chart", "bbox_xyxy_norm": raw_bbox, "score": 0.8611}
    ])

    chart = next(item for item in processed if item["label_name"] == "chart")
    assert chart["model_bbox_xyxy_norm"] == raw_bbox
    assert chart["bbox_xyxy_norm"] == raw_bbox
    assert chart["crop_bbox_xyxy_norm"][1] < raw_bbox[1]
    assert chart["processed_bbox_xyxy_norm"] != raw_bbox


def test_visual_evidence_orders_blocks_by_page_geometry():
    evidence = build_visual_evidence([
        {"page_number": 1, "text": "chart", "_bbox_xyxy_norm": [0.20, 0.70, 0.80, 0.88], "_content_type": "chart"},
        {"page_number": 1, "text": "paragraph", "_bbox_xyxy_norm": [0.10, 0.20, 0.90, 0.65]},
    ])

    assert [block["text"] for block in evidence["pages"][0]["blocks"]] == ["paragraph", "chart"]


def test_visual_evidence_projects_ocr_lines_from_page_row():
    evidence = build_visual_evidence([
        {
            "page_number": 1,
            "path": "scan.jpg",
            "page_image": {"image_b64": _png_b64()},
            "text": "Line one\nLine two",
            "_ocr_text_blocks": [
                {
                    "text": "Line one",
                    "bbox_xyxy_norm": [0.10, 0.10, 0.80, 0.15],
                    "source": "tesseract-5",
                    "line_detector_score": 0.91,
                    "region_label": "text",
                },
                {
                    "text": "Line two",
                    "bbox_xyxy_norm": [0.10, 0.20, 0.80, 0.25],
                    "source": "tesseract-5",
                    "line_detector_score": 0.88,
                    "region_label": "text",
                },
            ],
        }
    ])

    blocks = evidence["pages"][0]["blocks"]
    assert [block["text"] for block in blocks] == ["Line one", "Line two"]
    assert all(block["bbox"] for block in blocks)
    assert all(block["origin"] == "ocr_line" for block in blocks)
    assert all(block["ocr_source"] == "tesseract-5" for block in blocks)
    assert [block["line_detector_score"] for block in blocks] == [0.91, 0.88]


def test_chunk_pdf_rows_preserves_line_ocr_provenance():
    frame = pd.DataFrame([
        {
            "path": "scan.pdf",
            "page_number": 1,
            "text": "Line one",
            "_ocr_text_blocks": [{
                "text": "Line one",
                "bbox_xyxy_norm": [0.10, 0.10, 0.80, 0.15],
                "source": "tesseract-5",
                "line_detector_score": 0.91,
                "region_label": "text",
            }],
            "metadata": {"needs_ocr_for_text": True},
        }
    ])

    chunked = chunk_pdf_content_rows(frame, params=TextChunkParams(max_tokens=128, overlap_tokens=0))

    assert len(chunked) == 1
    assert chunked.iloc[0]["_bbox_xyxy_norm"] == [0.10, 0.10, 0.80, 0.15]
    assert chunked.iloc[0]["source"] == "tesseract-5"
    assert chunked.iloc[0]["line_detector_score"] == 0.91


def test_explode_does_not_copy_structured_regions_to_every_text_row():
    frame = pd.DataFrame([
        {
            "path": "chart.pdf",
            "page_number": 1,
            "text": "Paragraph",
            "_native_text_blocks": [{"text": "Paragraph", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}],
            "chart": [{"bbox_xyxy_norm": [0.2, 0.3, 0.8, 0.8], "text": "Chart output"}],
            "table": [],
            "infographic": [],
            "stamp": [],
            "images": [],
            "metadata": {},
        }
    ])

    exploded = explode_content_to_rows(frame)

    text_row = exploded[exploded["_content_type"] == "text"].iloc[0]
    chart_row = exploded[exploded["_content_type"] == "chart"].iloc[0]
    assert text_row["chart"] == []
    assert chart_row["chart"] == []


def test_ocr_parser_keeps_normalized_bbox():
    blocks = ocr_shared._parse_ocr_result(
        [
            {
                "text_prediction": {"text": "Dấu xác nhận"},
                "bounding_box": {
                    "points": [
                        {"x": 0.10, "y": 0.20},
                        {"x": 0.30, "y": 0.20},
                        {"x": 0.30, "y": 0.40},
                        {"x": 0.10, "y": 0.40},
                    ]
                },
            }
        ]
    )
    assert blocks[0]["bbox_xyxy_norm"] == [0.1, 0.2, 0.3, 0.4]


def test_ocr_crop_rounds_detector_edges_before_encoding():
    page_b64 = _png_b64(width=1000, height=700)
    crops = ocr_shared._crop_all_from_page(
        page_b64,
        [{"label_name": "title", "bbox_xyxy_norm": [0.25407082, 0.017907938, 0.71789104, 0.14189517]}],
        {"title"},
        as_b64=True,
    )
    assert len(crops) == 1
    with Image.open(io.BytesIO(base64.b64decode(crops[0][2]))) as crop:
        assert crop.size == (464, 86)


def test_ocr_merge_collapses_fuzzy_text_variants_in_same_region():
    merged = ocr_shared._merge_ocr_blocks(
        [
            {
                "text": "Độc lập T do aa pphc",
                "bbox_xyxy_norm": [0.300, 0.100, 0.600, 0.150],
                "ocr_source": "scan_tile",
            },
            {
                "text": "Độc lập – Tự do – Hạnh phúc",
                "bbox_xyxy_norm": [0.304, 0.102, 0.604, 0.152],
                "ocr_source": "scan_full_page",
            },
        ]
    )

    assert len(merged) == 1
    assert merged[0]["text"] == "Độc lập – Tự do – Hạnh phúc"
    assert set(merged[0]["ocr_sources"]) == {"scan_tile", "scan_full_page"}


def test_ocr_merge_uses_overlapping_bbox_when_text_errors_are_very_different():
    merged = ocr_shared._merge_ocr_blocks(
        [
            {
                "text": "Độc 1pp T do aa pphc",
                "confidence": 0.64,
                "bbox_xyxy_norm": [0.349, 0.095, 0.627, 0.133],
            },
            {
                "text": "Độc lập – Tự do - Hạnh phúc",
                "confidence": 0.86,
                "bbox_xyxy_norm": [0.3495, 0.0945, 0.6265, 0.1325],
            },
        ]
    )

    assert len(merged) == 1
    assert merged[0]["text"] == "Độc lập – Tự do - Hạnh phúc"


def test_ocr_merge_deduplicates_nested_regions_without_stripping_accents():
    merged = ocr_shared._merge_ocr_blocks(
        [
            {
                "text": "CÔNG TY TNHH DỊCH VỤ CÔNG NGHỆ ANH KIỆT",
                "confidence": 0.95,
                "bbox_xyxy_norm": [0.467, 0.044, 0.874, 0.061],
            },
            {
                "text": "CÔNG TY TNHH DỊCH VỤ CÔNG NGHỆ ANH KIỆT\nĐịa chỉ: 63 Nguyễn Thiện Thuật",
                "confidence": 0.70,
                "bbox_xyxy_norm": [0.462, 0.044, 0.928, 0.116],
            },
        ]
    )

    assert len(merged) == 1
    assert "Địa chỉ" in merged[0]["text"]
    assert "Nguyễn" in merged[0]["text"]


def test_page_elements_box_merge_deduplicates_nested_regions_without_stripping_accents():
    blocks = ppocr_adapter._deduplicate_box_blocks(
        [
            {
                "text": "CÔNG TY TNHH DỊCH VỤ CÔNG NGHỆ ANH KIỆT",
                "score": 0.95,
                "bbox_xyxy_norm": [0.467, 0.044, 0.874, 0.061],
            },
            {
                "text": "CÔNG TY TNHH DỊCH VỤ CÔNG NGHỆ ANH KIỆT\nĐịa chỉ: 63 Nguyễn Thiện Thuật",
                "score": 0.70,
                "bbox_xyxy_norm": [0.462, 0.044, 0.928, 0.116],
            },
        ]
    )

    assert len(blocks) == 1
    assert "Địa chỉ" in blocks[0]["text"]
    assert "Nguyễn" in blocks[0]["text"]


def test_ocr_merge_drops_punctuation_only_separator_blocks():
    merged = ocr_shared._merge_ocr_blocks(
        [{"text": "-", "bbox_xyxy_norm": [0.30, 0.10, 0.60, 0.12]}]
    )

    assert merged == []


def test_ocr_merge_keeps_adjacent_text_regions_separate():
    merged = ocr_shared._merge_ocr_blocks(
        [
            {"text": "Độc lập", "bbox_xyxy_norm": [0.10, 0.20, 0.25, 0.23]},
            {"text": "Độc lập – Tự do", "bbox_xyxy_norm": [0.10, 0.26, 0.35, 0.29]},
        ]
    )

    assert len(merged) == 2


def test_ocr_merge_converges_for_bridge_cluster():
    merged = ocr_shared._merge_ocr_blocks(
        [
            {
                "text": "same repeated heading",
                "confidence": 0.70,
                "bbox_xyxy_norm": [0.00, 0.10, 0.40, 0.20],
                "ocr_source": "page_elements_crop",
            },
            {
                "text": "same repeated heading",
                "confidence": 0.80,
                "bbox_xyxy_norm": [0.20, 0.10, 0.60, 0.20],
                "ocr_source": "scan_tile",
            },
            {
                "text": "same repeated heading",
                "confidence": 0.90,
                "bbox_xyxy_norm": [0.20, 0.10, 0.40, 0.20],
                "ocr_source": "scan_full_page",
            },
        ]
    )

    assert len(merged) == 1
    assert merged[0]["text"] == "same repeated heading"
    assert set(merged[0]["ocr_sources"]) == {"page_elements_crop", "scan_full_page", "scan_tile"}


def test_remote_ocr_keeps_visual_crop_and_maps_text_bbox(monkeypatch):
    page_b64 = _png_b64()
    page_bbox = [0.20, 0.30, 0.80, 0.70]

    def fake_ocr(**kwargs):
        assert len(kwargs["image_b64_list"]) == 1
        return [
            {
                "text_detections": [
                    {
                        "text_prediction": {"text": "Dấu xác nhận"},
                        "bounding_box": {
                            "points": [
                                {"x": 0.10, "y": 0.20},
                                {"x": 0.60, "y": 0.20},
                                {"x": 0.60, "y": 0.50},
                                {"x": 0.10, "y": 0.50},
                            ]
                        },
                    }
                ]
            }
        ]

    monkeypatch.setattr(ocr_shared, "invoke_image_inference_batches", fake_ocr)
    frame = pd.DataFrame(
        [
            {
                "path": "scan.pdf",
                "page_number": 1,
                "page_image": {"image_b64": page_b64},
                "images": [],
                "metadata": {"needs_ocr_for_text": True},
                "page_elements_v3": {
                    "detections": [
                        {
                            "label_name": "infographic",
                            "score": 0.91,
                            "bbox_xyxy_norm": page_bbox,
                        }
                    ]
                },
            }
        ]
    )

    result = ocr_shared.ocr_page_elements(
        frame,
        invoke_url="http://ocr.test/v1/ocr",
        extract_text=True,
        extract_infographics=True,
    )
    row = result.iloc[0]
    assert len(row["images"]) == 1
    assert row["images"][0]["image_type"] == "detected_region"
    assert row["images"][0]["bbox_xyxy_norm"] == page_bbox
    assert row["_ocr_visual_text_blocks"][0]["source_label"] == "infographic"
    assert row["_ocr_visual_text_blocks"][0]["bbox_xyxy_norm"] == [0.26, 0.38, 0.56, 0.5]


def test_integrated_nemotron_ocr_takes_priority_over_split_ppocr(monkeypatch):
    calls = []

    def fake_nemotron(**kwargs):
        calls.append(kwargs["invoke_url"])
        return [{"text_detections": [{"text_prediction": {"text": "Integrated OCR"}}]}]

    def fail_if_split_selected(*args, **kwargs):
        raise AssertionError("split PP-OCRv6 path must not run when ocr_invoke_url is configured")

    monkeypatch.setattr(ocr_shared, "invoke_image_inference_batches", fake_nemotron)
    monkeypatch.setattr(ppocr_adapter, "ppocrv6_page_elements", fail_if_split_selected)
    frame = pd.DataFrame(
        [
            {
                "path": "scan.pdf",
                "page_number": 1,
                "page_image": {"image_b64": _png_b64()},
                "images": [],
                "metadata": {"needs_ocr_for_text": False},
                "page_elements_v3": {
                    "detections": [
                        {"label_name": "infographic", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9], "score": 0.9}
                    ]
                },
            }
        ]
    )

    result = ocr_shared.ocr_page_elements(
        frame,
        invoke_url="http://nim-ocr/v1/ocr",
        line_detector_invoke_url="http://pp-det/v1/detect",
        ocr_recognizer_invoke_url="http://pp-rec/v1/recognize",
        extract_infographics=True,
    )

    assert calls == ["http://nim-ocr/v1/ocr"]
    assert result.iloc[0]["infographic"][0]["text"] == "Integrated OCR"
    assert result.iloc[0]["ocr"]["backend"] == "nemotron_ocr_v2_nim"


def test_integrated_scan_pipeline_keeps_crop_full_page_and_tile_recall(monkeypatch):
    request_sizes = []

    def fake_nemotron(**kwargs):
        request_sizes.append(len(kwargs["image_b64_list"]))
        return [
            {
                "text_detections": [
                    {
                        "text_prediction": {"text": "A complete scan line"},
                        "bounding_box": {
                            "points": [
                                {"x": 0.1, "y": 0.1},
                                {"x": 0.9, "y": 0.1},
                                {"x": 0.9, "y": 0.2},
                                {"x": 0.1, "y": 0.2},
                            ]
                        },
                    }
                ]
            }
            for _ in kwargs["image_b64_list"]
        ]

    monkeypatch.setattr(ocr_shared, "invoke_image_inference_batches", fake_nemotron)
    frame = pd.DataFrame(
        [
            {
                "path": "scan.pdf",
                "page_number": 1,
                "page_image": {"image_b64": _png_b64(width=1200, height=1400)},
                "images": [],
                "metadata": {"needs_ocr_for_text": True},
                "page_elements_v3": {
                    "detections": [
                        {"label_name": "text", "bbox_xyxy_norm": [0.05, 0.05, 0.95, 0.25], "score": 0.9}
                    ]
                },
            }
        ]
    )

    result = ocr_shared.ocr_page_elements(
        frame,
        invoke_url="http://nim-ocr/v1/ocr",
        extract_text=True,
        scan_ocr_fallback=True,
        scan_ocr_preprocess=False,
        scan_ocr_tile_size=512,
        scan_ocr_tile_overlap=0.15,
        scan_ocr_max_retries=0,
    )

    quality = result.iloc[0]["ocr"]["scan_ocr_quality"]
    assert request_sizes[0] == 1  # Page Elements crop
    assert any(size > 1 for size in request_sizes[1:])  # full page + overlapping tiles
    assert quality["fallback"]["full_page"] is True
    assert quality["fallback"]["tiles"] is True
    assert result.iloc[0]["_ocr_text_blocks"]


def test_explode_uses_ocr_blocks_and_retains_visuals_once():
    page_b64 = _png_b64()
    frame = pd.DataFrame(
        [
            {
                "path": "scan.pdf",
                "page_number": 1,
                "text": "Dấu xác nhận",
                "metadata": {"needs_ocr_for_text": True},
                "_ocr_text_blocks": [
                    {
                        "text": "Dấu xác nhận",
                        "confidence": 0.91,
                        "bbox_xyxy_norm": [0.26, 0.38, 0.56, 0.5],
                    }
                ],
                "images": [
                    {
                        "image_type": "detected_region",
                        "image_b64": page_b64,
                        "bbox_xyxy_norm": [0.20, 0.30, 0.80, 0.70],
                    }
                ],
                "table": [],
                "chart": [],
                "infographic": [],
                "stamp": [],
            }
        ]
    )
    result = explode_content_to_rows(frame)
    assert len(result) == 1
    assert result.iloc[0]["_content_type"] == "text"
    assert result.iloc[0]["_bbox_xyxy_norm"] == [0.26, 0.38, 0.56, 0.5]
    assert result.iloc[0]["confidence"] == 0.91
    assert len(result.iloc[0]["images"]) == 1


def test_pdf_chunking_operates_on_canonical_blocks_and_preserves_page(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(token_ids)

    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer",
        lambda model_id, cache_dir=None: FakeTokenizer(),
    )
    frame = pd.DataFrame(
        [
            {
                "path": "long.pdf",
                "page_number": 7,
                "text": "one two three four five six",
                "metadata": {"needs_ocr_for_text": False},
                "_native_text_blocks": [
                    {"text": "one two three four five six", "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.4]}
                ],
                "table": [{"text": "| H |\n| --- |\n| V |", "bbox_xyxy_norm": [0.1, 0.5, 0.8, 0.8]}],
                "chart": [],
                "infographic": [],
                "stamp": [],
                "images": [],
            }
        ]
    )

    result = chunk_pdf_content_rows(
        frame,
        TextChunkParams(max_tokens=3, overlap_tokens=1),
    )

    assert result["page_number"].tolist() == [7, 7, 7]
    assert result["text"].tolist() == ["one two three", "three four five", "five six"]
    assert result["_chunk_index"].tolist() == [0, 1, 2]
    assert result["_chunk_count"].tolist() == [3, 3, 3]
    assert len(result.iloc[0]["table"]) == 1
    assert result.iloc[1]["table"] == []
    assert result.iloc[0]["metadata"]["parse_chunk"]["parent_block_id"].endswith("block-0")


def test_pdf_chunking_fixes_ocr_block_path_without_duplicate_visual_rows(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(token_ids)

    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer",
        lambda model_id, cache_dir=None: FakeTokenizer(),
    )
    frame = pd.DataFrame(
        [
            {
                "path": "scan.pdf",
                "page_number": 3,
                "text": "alpha beta gamma delta",
                "metadata": {"needs_ocr_for_text": True},
                "_ocr_text_blocks": [
                    {"text": "alpha beta gamma delta", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}
                ],
                "table": [],
                "chart": [],
                "infographic": [],
                "stamp": [],
                "images": [],
            }
        ]
    )

    chunked = chunk_pdf_content_rows(frame, TextChunkParams(max_tokens=2, overlap_tokens=0))
    result = explode_content_to_rows(chunked)

    assert result["page_number"].tolist() == [3, 3]
    assert result["text"].tolist() == ["alpha beta", "gamma delta"]
    assert result["_reader_backend"].tolist() == ["ocr", "ocr"]


def test_stamp_detector_crop_is_ocrd_and_retained_as_one_block(monkeypatch):
    page_b64 = _png_b64()
    stamp_bbox = [0.60, 0.55, 0.90, 0.85]

    monkeypatch.setattr(
        stamp_shared,
        "invoke_image_inference_batches",
        lambda **kwargs: [{"detections": [{"label_name": "stamp", "score": 0.91, "bbox_xyxy_norm": stamp_bbox}]}],
    )
    detected = stamp_shared.detect_stamps(
        pd.DataFrame([{
            "page_image": {"image_b64": page_b64},
            "metadata": {"needs_ocr_for_text": True, "has_text": False},
        }]),
        invoke_url="http://stamp.test/v1/stamp-detection",
    )
    assert detected.iloc[0]["stamp_regions"][0]["bbox_xyxy_norm"] == stamp_bbox
    assert detected.iloc[0]["stamp_regions"][0]["image_b64"]

    def fake_ocr(**kwargs):
        return [{"text_detections": [{"text_prediction": {"text": "Đã ký đóng dấu"}}]}]

    monkeypatch.setattr(ocr_shared, "invoke_image_inference_batches", fake_ocr)
    detected["page_elements_v3"] = [{}]
    ocrd = ocr_shared.ocr_page_elements(
        detected,
        invoke_url="http://ocr.test/v1/ocr",
        extract_stamps=True,
    )
    assert ocrd.iloc[0]["stamp"][0]["text"] == "Đã ký đóng dấu"
    assert ocrd.iloc[0]["images"][0]["label_name"] == "stamp"

    rows = explode_content_to_rows(ocrd)
    stamp_rows = rows[rows["_content_type"] == "stamp"]
    assert len(stamp_rows) == 1
    assert stamp_rows.iloc[0]["text"] == "Đã ký đóng dấu"
    assert stamp_rows.iloc[0]["images"][0]["label_name"] == "stamp"


def test_scan_tiles_cover_page_and_map_bbox_back_to_page():
    image_b64 = _png_b64(width=1800, height=2200)
    tiles = page_elements_shared._scan_tiles_from_b64(image_b64)
    assert len(tiles) > 1
    assert all(len(bbox) == 4 and all(0.0 <= value <= 1.0 for value in bbox) for bbox, _ in tiles)

    mapped = page_elements_shared._map_detection_bbox_to_page(
        {"label_name": "image", "bbox_xyxy_norm": [0.25, 0.25, 0.75, 0.75]},
        [0.2, 0.3, 0.8, 0.9],
    )
    assert all(
        abs(actual - expected) <= 1e-9
        for actual, expected in zip(mapped["bbox_xyxy_norm"], [0.35, 0.45, 0.65, 0.75])
    )


def test_visual_evidence_deduplicates_near_identical_boxes_but_keeps_repeated_text():
    evidence = build_visual_evidence([
        {"page_number": 1, "text": "Minh", "_bbox_xyxy_norm": [0.10, 0.20, 0.30, 0.40]},
        {"page_number": 1, "text": "Minh", "_bbox_xyxy_norm": [0.1004, 0.2003, 0.3002, 0.4001]},
        {"page_number": 1, "text": "Minh", "_bbox_xyxy_norm": [0.70, 0.20, 0.80, 0.30]},
    ])

    assert evidence["block_count"] == 2
    assert [block["text"] for block in evidence["pages"][0]["blocks"]] == ["Minh", "Minh"]

    legacy = {
        "pages": [{
            "page_number": 1,
            "blocks": [
                {"id": "a", "text": "vn", "bbox": [0.372, 0.096, 0.397, 0.105]},
                {"id": "b", "text": ".vn", "bbox": [0.3721, 0.0962, 0.3969, 0.1051]},
            ],
        }],
    }
    assert deduplicate_visual_evidence(legacy)["block_count"] == 1


def test_visual_evidence_deduplicates_fuzzy_ocr_variants_with_contained_boxes():
    evidence = build_visual_evidence(
        [
            {
                "page_number": 1,
                "text": "Độc lập T do aa pphc",
                "_bbox_xyxy_norm": [0.100, 0.200, 0.300, 0.400],
            },
            {
                "page_number": 1,
                "text": "Độc lập – Tự do – Hạnh phúc",
                "_bbox_xyxy_norm": [0.108, 0.208, 0.308, 0.408],
            },
        ]
    )

    assert evidence["block_count"] == 1
    assert evidence["pages"][0]["blocks"][0]["text"] == "Độc lập – Tự do – Hạnh phúc"


def test_visual_evidence_deduplicates_nested_text_boxes_even_when_ocr_differs():
    evidence = build_visual_evidence([
        {
            "page_number": 1,
            "text": "CÔNG TY TNHH DỊCH VỤ CÔNG NGHỆ ANH KIỆT",
            "_bbox_xyxy_norm": [0.467, 0.044, 0.874, 0.061],
            "_content_type": "text",
        },
        {
            "page_number": 1,
            "text": "CÔNG TY TNHH DỊCH VỤ CÔNG NGHỆ ANH KIỆT\nĐịa chỉ: 63 Nguyễn Thiện Thuật",
            "_bbox_xyxy_norm": [0.462, 0.044, 0.928, 0.116],
            "_content_type": "text",
        },
    ])

    blocks = evidence["pages"][0]["blocks"]
    assert len(blocks) == 1
    assert "Địa chỉ" in blocks[0]["text"]


def test_visual_evidence_keeps_adjacent_text_boxes_that_only_touch():
    evidence = build_visual_evidence([
        {"page_number": 1, "text": "Dòng một", "_bbox_xyxy_norm": [0.10, 0.20, 0.80, 0.25]},
        {"page_number": 1, "text": "Dòng hai", "_bbox_xyxy_norm": [0.10, 0.251, 0.80, 0.301]},
    ])

    assert len(evidence["pages"][0]["blocks"]) == 2


def test_visual_evidence_keeps_scan_page_raster_as_background_not_image_block():
    page_b64 = _png_b64()
    evidence = build_visual_evidence(
        [
            {
                "page_number": 1,
                "page_image": {"image_b64": page_b64, "encoding": "png"},
                "text": "OCR text",
                "_reader_backend": "ocr",
                "_bbox_xyxy_norm": [0.10, 0.10, 0.90, 0.20],
                "images": [
                    {
                        "label_name": "image",
                        "image_type": "pdfium_page_image",
                        "bbox_xyxy_norm": [0.0, 0.0, 0.998, 1.0],
                    },
                    {
                        "label_name": "image",
                        "image_type": "detected_region",
                        "bbox_xyxy_norm": [0.65, 0.60, 0.85, 0.82],
                    },
                ],
            }
        ]
    )

    assert evidence["pages"][0]["image_available"] is True
    assert [block["content_type"] for block in evidence["pages"][0]["blocks"]] == ["text", "image"]
    assert evidence["pages"][0]["blocks"][1]["image_type"] == "detected_region"

    legacy = {
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"id": "background", "content_type": "image", "text": "", "bbox": [0, 0, 0.998, 1]},
                    {"id": "text", "content_type": "text", "text": "OCR text", "bbox": [0.1, 0.1, 0.9, 0.2]},
                ],
            }
        ]
    }
    assert deduplicate_visual_evidence(legacy)["block_count"] == 1


def test_visual_manifest_sanitizes_nan_model_scores():
    evidence = build_visual_evidence(
        [
            {
                "page_number": 1,
                "text": "table text",
                "confidence": float("nan"),
                "_reading_order": float("nan"),
                "_bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2],
            }
        ]
    )
    manifest = manifest_without_images(evidence)

    # Starlette's JSONResponse uses allow_nan=False.
    json.dumps(manifest, allow_nan=False)
    block = manifest["pages"][0]["blocks"][0]
    assert block["confidence"] is None
    assert block["reading_order"] is None


def test_visual_evidence_flattens_page_elements_detection_payload():
    evidence = build_visual_evidence([
        {
            "page_number": 1,
            "page_image": {"image_b64": _png_b64()},
            "text": "Full page OCR text",
            "page_elements_v3": {
                "detections": [
                    {"label_name": "title", "bbox_xyxy_norm": [0.10, 0.05, 0.90, 0.12], "score": 0.93},
                    {"label_name": "table", "bbox_xyxy_norm": [0.10, 0.30, 0.90, 0.70], "score": 0.88},
                ]
            },
        }
    ])

    blocks = evidence["pages"][0]["blocks"]
    assert {block["content_type"] for block in blocks} >= {"text", "title", "table"}
    assert sum(block["origin"] == "page_elements_v3" for block in blocks) == 2


def test_ppocrv6_recognizer_accepts_singular_rec_fields():
    assert ppocr_adapter._recognized_text({"rec_text": "Dấu xác nhận", "rec_score": 0.91}) == (
        "Dấu xác nhận",
        0.91,
    )


def test_option2_keeps_page_elements_bbox_separate_from_ppocr_line_bbox(monkeypatch):
    page_bbox = [0.10, 0.20, 0.90, 0.60]

    def fake_inference(**kwargs):
        if kwargs["invoke_url"] == "http://pp-det/v1":
            return [{"boxes": [[0.0, 0.0, 1.0, 0.45], [0.0, 0.55, 1.0, 1.0]], "scores": [0.95, 0.94]}]
        return [
            {"text": "first line", "score": 0.91, "model": "PP-OCRv6_medium_rec"},
            {"text": "second line", "score": 0.90, "model": "PP-OCRv6_medium_rec"},
        ]

    monkeypatch.setattr(ppocr_adapter, "invoke_image_inference_batches", fake_inference)
    frame = pd.DataFrame([
        {
            "path": "scan.png",
            "page_number": 1,
            "page_image": {"image_b64": _png_b64()},
            "images": [],
            "metadata": {"needs_ocr_for_text": True},
            "page_elements_v3": {"detections": [{"label_name": "text", "bbox_xyxy_norm": page_bbox, "score": 0.88}]},
        }
    ])

    result = ppocr_adapter.ppocrv6_page_elements(
        frame,
        line_detector_invoke_url="http://pp-det/v1",
        ocr_recognizer_invoke_url="http://pp-rec/v1",
        box_ocr_mode=True,
        extract_text=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        extract_images=False,
        use_table_structure=False,
    )

    blocks = result.iloc[0]["_ocr_text_blocks"]
    assert len(blocks) == 2
    assert all(block["ocr_mode"] == "page_elements_ppocr_line" for block in blocks)
    assert all(block["model_bbox_xyxy_norm"] == page_bbox for block in blocks)
    assert blocks[0]["bbox_xyxy_norm"] != page_bbox
    assert blocks[1]["bbox_xyxy_norm"] != page_bbox


def test_tesseract_option_ocr_uses_page_elements_bbox_without_line_detector(monkeypatch):
    calls = []

    def fake_inference(**kwargs):
        calls.append((kwargs["invoke_url"], len(kwargs["image_b64_list"])))
        return [
            {"text": "Text from Page Elements box", "score": 0.91, "model": "tesseract-5", "backend": "tesseract"}
            for _ in kwargs["image_b64_list"]
        ]

    monkeypatch.setattr(ppocr_adapter, "invoke_image_inference_batches", fake_inference)
    page_bbox = [0.10, 0.20, 0.80, 0.32]
    frame = pd.DataFrame([
        {
            "path": "scan.pdf",
            "page_number": 1,
            "page_image": {"image_b64": _png_b64()},
            "images": [],
            "metadata": {"needs_ocr_for_text": True},
            "page_elements_v3": {"detections": [
                {"label_name": "text", "bbox_xyxy_norm": page_bbox, "score": 0.87},
                {"label_name": "image", "bbox_xyxy_norm": [0.60, 0.60, 0.90, 0.90], "score": 0.82},
            ]},
        }
    ])

    result = ppocr_adapter.ppocrv6_page_elements(
        frame,
        line_detector_invoke_url=None,
        ocr_recognizer_invoke_url="http://tesseract/v1/ocr",
        box_ocr_mode=True,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        use_table_structure=False,
    )

    assert calls == [("http://tesseract/v1/ocr", 1)]
    block = result.iloc[0]["_ocr_text_blocks"][0]
    assert block["bbox_xyxy_norm"] == page_bbox
    assert block["ocr_mode"] == "page_elements_box"
    assert block["source"] == "tesseract-5"
    assert result.iloc[0]["images"][0]["bbox_xyxy_norm"] == [0.60, 0.60, 0.90, 0.90]


def test_tesseract_option_uses_raw_page_elements_geometry_and_scan_recall(monkeypatch):
    calls = []

    def fake_inference(**kwargs):
        calls.append(len(kwargs["image_b64_list"]))
        if len(calls) == 1:
            return [{"text": "", "score": None, "model": "tesseract-5", "backend": "tesseract"}]
        return [{"text": "Recovered scan text", "score": 0.88, "model": "tesseract-5", "backend": "tesseract"}]

    monkeypatch.setattr(ppocr_adapter, "invoke_image_inference_batches", fake_inference)
    raw_bbox = [0.12, 0.20, 0.42, 0.26]
    frame = pd.DataFrame([
        {
            "path": "scan.pdf",
            "page_number": 1,
            "page_image": {"image_b64": _png_b64()},
            "metadata": {"needs_ocr_for_text": True},
            "page_elements_v3": {"detections": [
                {
                    "label_name": "text",
                    "bbox_xyxy_norm": [0.02, 0.10, 0.96, 0.70],
                    "model_bbox_xyxy_norm": raw_bbox,
                    "processed_bbox_xyxy_norm": [0.02, 0.10, 0.96, 0.70],
                    "score": 0.87,
                }
            ]},
        }
    ])

    result = ppocr_adapter.ppocrv6_page_elements(
        frame,
        ocr_recognizer_invoke_url="http://tesseract/v1/ocr",
        box_ocr_mode=True,
        scan_ocr_fallback=True,
        extract_text=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        extract_images=False,
        use_table_structure=False,
    )

    row = result.iloc[0]
    assert calls == [1, 1]
    assert row["_ocr_text_blocks"][0]["bbox_xyxy_norm"] == [0.0, 0.0, 1.0, 1.0]
    assert row["_ocr_text_blocks"][0]["ocr_mode"] == "scan_full_page"
    assert row["ocr"]["scan_recall_used_as_output"] is True


def test_tesseract_option_keeps_raw_bbox_when_postprocess_expands_text_region(monkeypatch):
    monkeypatch.setattr(
        ppocr_adapter,
        "invoke_image_inference_batches",
        lambda **kwargs: [{"text": "Raw geometry", "score": 0.9, "model": "tesseract-5", "backend": "tesseract"}],
    )
    raw_bbox = [0.30, 0.30, 0.52, 0.35]
    processed_bbox = [0.05, 0.20, 0.90, 0.70]
    frame = pd.DataFrame([
        {
            "page_image": {"image_b64": _png_b64()},
            "metadata": {"needs_ocr_for_text": True},
            "page_elements_v3": {"detections": [{
                "label_name": "text",
                "bbox_xyxy_norm": processed_bbox,
                "model_bbox_xyxy_norm": raw_bbox,
                "processed_bbox_xyxy_norm": processed_bbox,
            }]},
        }
    ])
    result = ppocr_adapter.ppocrv6_page_elements(
        frame,
        ocr_recognizer_invoke_url="http://tesseract/v1/ocr",
        box_ocr_mode=True,
        scan_ocr_fallback=False,
        extract_text=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        extract_images=False,
        use_table_structure=False,
    )
    block = result.iloc[0]["_ocr_text_blocks"][0]
    assert block["bbox_xyxy_norm"] == raw_bbox
    assert block["processed_bbox_xyxy_norm"] == processed_bbox


def test_ppocrv6_drops_page_sized_text_dominated_infographics():
    detections = [
        {"label_name": "text", "bbox_xyxy_norm": [0.032, 0.172, 0.517, 0.243]},
        {"label_name": "text", "bbox_xyxy_norm": [0.693, 0.173, 0.930, 0.244]},
        {"label_name": "text", "bbox_xyxy_norm": [0.462, 0.044, 0.928, 0.116]},
        {"label_name": "text", "bbox_xyxy_norm": [0.027, 0.429, 0.966, 0.640]},
        {"label_name": "text", "bbox_xyxy_norm": [0.034, 0.042, 0.399, 0.107]},
        {"label_name": "text", "bbox_xyxy_norm": [0.014, 0.375, 1.0, 0.985]},
        {
            "label_name": "infographic",
            "score": 0.56,
            "bbox_xyxy_norm": [0.006, 0.008, 0.533, 0.40],
        },
        {
            "label_name": "infographic",
            "score": 0.61,
            "bbox_xyxy_norm": [0.454, 0.0, 0.986, 0.40],
        },
    ]
    assert not ppocr_adapter._keep_visual_detection(detections[2], detections)
    assert not ppocr_adapter._keep_visual_detection(detections[3], detections)
    assert ppocr_adapter._keep_visual_detection(
        {"label_name": "image", "bbox_xyxy_norm": [0.70, 0.70, 0.85, 0.85]},
        detections,
    )
