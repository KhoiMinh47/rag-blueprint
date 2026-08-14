# Isolated Vietnamese OCR benchmark

This directory is intentionally separate from the ingest pipeline.  It creates
a native-text PDF and a raster-only OCR PDF with the same ground truth, then
sends fixed line crops to one OCR server at a time.

The server contract is:

- `GET /v1/health/ready`
- `POST /v1/recognize` with `{ "images": [{"image_b64": "..."}] }`

`run_benchmark.py` reports exact-line accuracy, CER, CER without diacritics,
numeric-token accuracy, latency, and error examples for both corpora.
