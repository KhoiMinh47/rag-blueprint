import pandas as pd
import requests

from nemo_retriever.common.modality import stamp_detection as stamp_shared


def test_missing_local_stamp_detector_skips_without_inference(monkeypatch):
    stamp_shared._stamp_probe_cache.clear()

    def unavailable(*_args, **_kwargs):
        raise requests.ConnectionError("stamp-detector is not running")

    def should_not_infer(**_kwargs):
        raise AssertionError("stamp inference should be skipped when the sidecar is unavailable")

    monkeypatch.setattr(stamp_shared.requests, "get", unavailable)
    monkeypatch.setattr(stamp_shared, "invoke_image_inference_batches", should_not_infer)

    result = stamp_shared.detect_stamps(
        pd.DataFrame(
            [
                {
                    "page_image": {"image_b64": "aGVsbG8="},
                    "metadata": {"needs_ocr_for_text": True, "has_text": False},
                }
            ]
        ),
        invoke_url="http://stamp-detector:8000/v1/stamp-detection",
    )

    payload = result.iloc[0]["stamp_detection"]
    assert payload["detections"] == []
    assert payload["regions"] == []
    assert payload["error"]["stage"] == "stamp_detection"
    assert payload["error"]["type"] == "StampDetectorUnavailable"
