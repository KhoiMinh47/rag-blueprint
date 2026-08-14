from __future__ import annotations

from typing import Any

from nemo_retriever.common.modality.stamp_detection import detect_stamps
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.operator_archetype import ArchetypeOperator


@designer_component(
    name="Stamp / Seal Detection",
    category="Detection & OCR",
    compute="cpu",
    description="Detects stamps and seals with a remote zero-shot detector",
)
class StampDetectionActor(ArchetypeOperator):
    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        return True

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.operators.extract.stamp.cpu_actor import StampDetectionCPUActor

        return StampDetectionCPUActor

    @classmethod
    def gpu_variant_class(cls):
        return None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
