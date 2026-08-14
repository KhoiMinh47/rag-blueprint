# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development-only Uvicorn factories used by the Docker Compose stack."""

from __future__ import annotations

import os

from nemo_retriever.common.remote_auth import resolve_remote_api_key
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import load_config
from nemo_retriever.service.vectordb_app import create_vectordb_app as build_vectordb_app


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def create_retriever_app():
    """Create the retriever app from the mounted service YAML."""

    config_path = os.getenv(
        "NEMO_RETRIEVER_SERVICE_CONFIG",
        "/etc/nemo-retriever/retriever-service.yaml",
    )
    return create_app(load_config(config_path=config_path))


def create_vectordb_app():
    """Create the VectorDB app from Compose environment variables."""

    return build_vectordb_app(
        lancedb_uri=os.getenv("NEMO_RETRIEVER_VDB_URI", "/data/vectordb"),
        table_name=os.getenv("NEMO_RETRIEVER_VDB_TABLE", "nemo_retriever"),
        embed_endpoint=os.getenv("NEMO_RETRIEVER_VDB_EMBED_ENDPOINT", ""),
        embed_model=os.getenv(
            "NEMO_RETRIEVER_VDB_EMBED_MODEL",
            "nvidia/llama-nemotron-embed-vl-1b-v2",
        ),
        embed_model_provider_prefix=os.getenv("NEMO_RETRIEVER_VDB_EMBED_PROVIDER_PREFIX") or None,
        embed_api_key=resolve_remote_api_key(os.getenv("NEMO_RETRIEVER_VDB_EMBED_API_KEY", "")) or "",
        local_embed=_env_bool("NEMO_RETRIEVER_VDB_LOCAL_EMBED"),
        local_embed_backend=os.getenv("NEMO_RETRIEVER_VDB_LOCAL_EMBED_BACKEND", "hf"),
        hf_cache_dir=os.getenv("NEMO_RETRIEVER_VDB_HF_CACHE_DIR") or None,
        device=os.getenv("NEMO_RETRIEVER_VDB_DEVICE") or None,
        gpu_memory_utilization=_env_float("NEMO_RETRIEVER_VDB_GPU_MEMORY_UTILIZATION", 0.45),
    )
