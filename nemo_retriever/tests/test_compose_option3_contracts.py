# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration contracts for the Option 3 VietOCR sidecar wiring."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "dev/compose/service-mode.compose.yaml"
PRESET = ROOT / "dev/compose/presets/option3-vietocr.env"
HELM_VALUES = ROOT / "helm/values.yaml"
HELM_CONFIGMAP = ROOT / "helm/templates/configmap.yaml"
DASHBOARD = ROOT / "src/nemo_retriever/service/routers/dashboard.py"


def test_compose_defaults_option3_to_the_gpu_vietocr_sidecar() -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    config = compose["configs"]["retriever_service_config"]["content"]
    vietocr = compose["services"]["vietocr-ocr"]
    preset = PRESET.read_text(encoding="utf-8")

    assert (
        'vietnamese_ocr_invoke_url: "${VIETNAMESE_OCR_URL:-http://vietocr-ocr:8000/v1/ocr}"'
        in config
    )
    assert vietocr["profiles"] == ["vietocr"]
    assert "http://localhost:8000/v1/health/ready" in vietocr["healthcheck"]["test"]
    assert "VIETNAMESE_OCR_URL=http://vietocr-ocr:8000/v1/ocr" in preset
    assert "VIETOCR_MODEL=vgg_seq2seq" in preset
    assert "PIPELINE_BATCH_WORKERS=2" in preset


def test_helm_exposes_option3_vietnamese_endpoint_without_reusing_option2_fields() -> None:
    values = HELM_VALUES.read_text(encoding="utf-8")
    configmap = HELM_CONFIGMAP.read_text(encoding="utf-8")

    assert 'vietnameseOcrInvokeUrl: ""' in values
    assert (
        "vietnamese_ocr_invoke_url: "
        "{{ .Values.serviceConfig.nimEndpoints.vietnameseOcrInvokeUrl | quote }}"
    ) in configmap
    assert "vinternOcrInvokeUrl" not in configmap
    assert "tesseractOcrInvokeUrl" in configmap


def test_default_vietocr_endpoint_does_not_relabel_the_default_pipeline() -> None:
    dashboard = DASHBOARD.read_text(encoding="utf-8")

    assert 'if configured_selector == "pipeline-option3":' in dashboard
    assert "integrated_ocr_endpoint and vietnamese_ocr_endpoint" not in dashboard
