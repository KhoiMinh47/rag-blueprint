# SPDX-License-Identifier: Apache-2.0

"""Hard-VRAM phase scheduler used exclusively by OCR Pipeline 2."""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

_GPU_SERVICES = {
    "nim-page-elements",
    "nim-table-structure",
    "nim-ocr",
    "vietocr-ocr",
}
_EXTRACTION_SERVICES = set(_GPU_SERVICES)
_PHASE_SERVICES = {
    # The 20-GiB budget covers only file -> text extraction. Keep this complete
    # group resident across detect and OCR so there is no model cold switch.
    "detect": _EXTRACTION_SERVICES,
    "ocr": _EXTRACTION_SERVICES,
    # Embedding/vector DB are outside this budget and are never controlled or
    # counted here. Keep extraction resident across embedding and future jobs;
    # this removes the largest repeated cold-start cost.
    "embed": _EXTRACTION_SERVICES,
}
_LEASES: dict[str, Any] = {}
_LEASES_LOCK = threading.Lock()


def scheduler_enabled() -> bool:
    return str(os.getenv("OPTION2_GPU_SCHEDULER_ENABLED", "false")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def transition_option2_gpu_phase(data: Any, *, phase: str, owner: str) -> Any:
    """Acquire the global job lease and switch resident models atomically."""

    if not scheduler_enabled():
        return data
    if phase not in _PHASE_SERVICES:
        raise ValueError(f"Unknown Option 2 GPU phase: {phase}")
    _acquire_owner(owner)
    # Detect, OCR and embed boundaries intentionally keep the exact same
    # extraction group resident in 20-GiB mode. The detect boundary already
    # validated health/VRAM while holding this job's lease; repeating four
    # Docker exec probes at the later boundaries only adds latency.
    if phase != "detect" and _PHASE_SERVICES[phase] == _PHASE_SERVICES["detect"]:
        return data
    try:
        _switch_phase(phase)
    except Exception:
        release_option2_gpu_phase(data, owner=owner)
        raise
    return data


def release_option2_gpu_phase(data: Any, *, owner: str) -> Any:
    """Release the cross-process lease after embedding or on failure."""

    if not scheduler_enabled():
        return data
    with _LEASES_LOCK:
        lease = _LEASES.pop(owner, None)
    if lease is not None:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()
    return data


def _acquire_owner(owner: str) -> None:
    with _LEASES_LOCK:
        if owner in _LEASES:
            return
        # Workers are deliberately configured one-at-a-time in hard-VRAM
        # mode. Recover a lease left by a failed prior graph in this process.
        stale_leases = list(_LEASES.values())
        _LEASES.clear()
    for stale in stale_leases:
        fcntl.flock(stale.fileno(), fcntl.LOCK_UN)
        stale.close()
    lock_path = Path(os.getenv("OPTION2_GPU_LOCK_PATH", "/var/lib/nemo-retriever/option2-gpu.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lease = lock_path.open("a+")
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
    with _LEASES_LOCK:
        existing = _LEASES.setdefault(owner, lease)
    if existing is not lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()


def _docker_client() -> httpx.Client:
    socket_path = os.getenv("OPTION2_DOCKER_SOCKET", "/var/run/docker.sock")
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=socket_path),
        base_url="http://docker",
        timeout=float(os.getenv("OPTION2_DOCKER_TIMEOUT_S", "300")),
    )


def _compose_containers(client: httpx.Client) -> dict[str, dict[str, Any]]:
    project = os.getenv("OPTION2_COMPOSE_PROJECT", "nemo-retriever-service-dev")
    filters = quote(json.dumps({"label": [f"com.docker.compose.project={project}"]}))
    response = client.get(f"/containers/json?all=true&filters={filters}")
    response.raise_for_status()
    result: dict[str, dict[str, Any]] = {}
    for item in response.json():
        labels = item.get("Labels") or {}
        service = labels.get("com.docker.compose.service")
        if service in _GPU_SERVICES:
            result[str(service)] = item
    missing = _GPU_SERVICES.difference(result)
    if missing:
        raise RuntimeError(f"Option 2 GPU containers are not created: {sorted(missing)}")
    return result


def _switch_phase(phase: str) -> None:
    desired = _PHASE_SERVICES[phase]
    with _docker_client() as client:
        containers = _compose_containers(client)
        for service in sorted(_GPU_SERVICES - desired):
            item = containers[service]
            if str(item.get("State")) == "running":
                response = client.post(f"/containers/{item['Id']}/stop?t=30")
                if response.status_code not in {204, 304}:
                    response.raise_for_status()
        for service in sorted(desired):
            item = containers[service]
            if str(item.get("State")) != "running":
                response = client.post(f"/containers/{item['Id']}/start")
                if response.status_code not in {204, 304}:
                    response.raise_for_status()
        deadline = time.monotonic() + float(os.getenv("OPTION2_MODEL_READY_TIMEOUT_S", "300"))
        while True:
            statuses = []
            for service in sorted(desired):
                inspect = client.get(f"/containers/{containers[service]['Id']}/json")
                inspect.raise_for_status()
                state = inspect.json().get("State") or {}
                health = (state.get("Health") or {}).get("Status")
                statuses.append(health or state.get("Status"))
            if all(status == "healthy" for status in statuses):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Option 2 phase {phase} did not become healthy: {statuses}")
            time.sleep(1.0)
        used_mib = _services_gpu_used_mib(client, containers, desired)
        limit_mib = int(os.getenv("OPTION2_GPU_VRAM_LIMIT_MIB", "20480"))
        if used_mib > limit_mib:
            _stop_services(client, containers, desired)
            raise RuntimeError(
                f"Option 2 VRAM guard tripped in phase {phase}: {used_mib} MiB > {limit_mib} MiB"
            )


def _stop_services(
    client: httpx.Client,
    containers: dict[str, dict[str, Any]],
    services: set[str],
) -> None:
    for service in sorted(services):
        client.post(f"/containers/{containers[service]['Id']}/stop?t=10")


def _services_gpu_used_mib(
    client: httpx.Client,
    containers: dict[str, dict[str, Any]],
    services: set[str],
) -> int:
    """Sum CUDA process memory belonging only to extraction containers."""

    # Each container has its own PID namespace. Executing nvidia-smi inside it
    # exposes only that container's CUDA processes, while a GPU-level memory
    # query would incorrectly include the separately-budgeted embed service.
    def _container_used_mib(container_id: str) -> int:
        created = client.post(
            f"/containers/{container_id}/exec",
            json={
                "AttachStdout": True,
                "AttachStderr": True,
                "Cmd": [
                    "nvidia-smi",
                    "--query-compute-apps=used_memory",
                    "--format=csv,noheader,nounits",
                ],
            },
        )
        created.raise_for_status()
        response = client.post(
            f"/exec/{created.json()['Id']}/start",
            json={"Detach": False, "Tty": True},
        )
        response.raise_for_status()
        subtotal = 0
        for line in response.text.strip().splitlines():
            value = line.strip()
            if value.isdigit():
                subtotal += int(value)
        return subtotal

    container_ids = [containers[service]["Id"] for service in services]
    with ThreadPoolExecutor(max_workers=len(container_ids)) as executor:
        return sum(executor.map(_container_used_mib, container_ids))
