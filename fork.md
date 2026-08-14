# Fork debug ingest

GET `/v1/dashboard/api/jobs/{job_id}/documents/{document_id}/pipeline` — trace file → sheet/trang → block, reader backend, range/bbox, function/model và output.

POST `http://nim-ocr:8000/v1/ocr` — Nemotron OCR v2 tích hợp detect + recognize cho crop Page Elements và scan toàn trang/tile.

PDF — PDFium native → Page Elements v3 → Table Structure v1 (table) → Nemotron OCR v2 (crop + scan full-page/tile) → clean → embedding.

Option 2 — PDFium raster từng trang → Page Elements v3 → Table Structure v1 → line detect PP-OCRv6 → Vintern probe một lần/page → route toàn bộ line sang Vintern hoặc Nemotron OCR v2 → clean → embedding; vLLM ổn định ở 2 request đồng thời với GPU cap 0.30.

`vintern-ocr` — vLLM server riêng dùng image `nemo-retriever-service-gpu:dev` có sẵn; model cache ở `./cache/vintern/model`, runtime cache ở `./cache/vintern/runtime`.

`_merge_ocr_blocks` — gộp crop/toàn trang/tile đến hội tụ, giữ bbox tốt nhất và provenance OCR.

GET `/v1/dashboard/api/jobs/{job_id}/documents/{document_id}/visual` — giữ raster scan làm nền PDF, không phát sinh block ảnh toàn trang giả.

POST `/v1/ingest/job` — tạo job ingest; backend tự route PDF/media hoặc XLSX/XLS/CSV theo suffix.

`SpreadsheetExtractActor` / `spreadsheet_bytes_to_chunks_df` — parser native CPU, sinh Markdown chunks theo sheet/range hoặc nhóm dòng CSV.

XLSX — đọc cell, công thức, merged/grid, chart/image anchor và provenance bằng `openpyxl`.

XLS — dùng LibreOffice chuyển sang XLSX rồi đọc native bằng `openpyxl`.

CSV — tự nhận encoding/delimiter, xử lý quoted/ragged rows và giữ cột image/path/data URI.

Spreadsheet pipeline — native cell/CSV không chạy Page Elements, Table Structure hoặc OCR; chỉ normalize → Markdown → embedding.

Dashboard Job Detail — chọn từng sheet/trang để xem source type, sheet/range, Markdown canonical và output JSON từng block.
