# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.common.modality.page_elements.shared import (
    _apply_final_score_filter,
    _apply_page_elements_v3_postprocess,
    _remote_response_to_detections,
)


def test_rejected_structured_box_cannot_suppress_surviving_title() -> None:
    raw_detections = [
        {
            "bbox_xyxy_norm": [0.081876, 0.077799, 0.738124, 0.111654],
            "label": 2,
            "label_name": "title",
            "score": 0.912088,
        },
        {
            "bbox_xyxy_norm": [0.056368, 0.004097, 0.965700, 0.972466],
            "label": 0,
            "label_name": "table",
            "score": 0.052635,
        },
    ]

    postprocessed = _apply_page_elements_v3_postprocess(raw_detections)
    final_detections = _apply_final_score_filter(postprocessed)

    assert [detection["label_name"] for detection in final_detections] == ["title"]


def test_surviving_structured_box_can_still_absorb_title() -> None:
    raw_detections = [
        {
            "bbox_xyxy_norm": [0.081876, 0.077799, 0.738124, 0.111654],
            "label": 2,
            "label_name": "title",
            "score": 0.912088,
        },
        {
            "bbox_xyxy_norm": [0.056368, 0.004097, 0.965700, 0.972466],
            "label": 0,
            "label_name": "table",
            "score": 0.9,
        },
    ]

    postprocessed = _apply_page_elements_v3_postprocess(raw_detections)
    final_detections = _apply_final_score_filter(postprocessed)

    assert [detection["label_name"] for detection in final_detections] == ["table"]


def test_scan_remote_boxes_can_bypass_generic_title_expansion() -> None:
    response = {
        "data": [
            {
                "index": 0,
                "bounding_boxes": {
                    "table": [
                        {
                            "x_min": 0.16,
                            "y_min": 0.16,
                            "x_max": 0.85,
                            "y_max": 0.83,
                            "confidence": 0.68,
                        }
                    ],
                    "title": [
                        {
                            "x_min": 0.75,
                            "y_min": 0.04,
                            "x_max": 0.87,
                            "y_max": 0.08,
                            "confidence": 0.24,
                        }
                    ],
                },
            }
        ]
    }
    scan_detections = _remote_response_to_detections(
        response_json=response,
        label_names=["table", "chart", "title", "infographic", "text", "header_footer"],
        thresholds_per_class=[0.0] * 6,
        apply_v3_postprocess=False,
    )
    table = next(item for item in scan_detections if item["label_name"] == "table")
    assert table["bbox_xyxy_norm"] == [0.16, 0.16, 0.85, 0.83]
