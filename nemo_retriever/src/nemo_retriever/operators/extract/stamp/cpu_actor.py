from __future__ import annotations

from typing import Any

from nemo_retriever.common.modality.stamp_detection import detect_stamps
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


class StampDetectionCPUActor(AbstractOperator, CPUOperator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.detect_kwargs = dict(kwargs)
        self._nim_client = NIMClient(max_pool_workers=int(kwargs.get("remote_max_pool_workers", 8)))

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return detect_stamps(data, nim_client=self._nim_client, **self.detect_kwargs, **kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
