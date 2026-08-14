# NeMo-Retriever — Docker start commands

> Chạy các lệnh từ thư mục gốc repository. Chưa chạy tự động.

## 1. Hosted API — ingest/search cơ bản

Container: `retriever`, `vectordb`.

```bash
export NVIDIA_API_KEY=nvapi-...

docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d retriever vectordb
```

## 2. Frontend debug ingest

Không cần container frontend riêng; dashboard được serve bởi `retriever`.

```bash
RETRIEVER_HTTP_PORT=7780 \
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d retriever vectordb
```

Mở: `http://localhost:7780/v1/dashboard/?v=1#ingest`

Frontend này gọi trực tiếp API ingest của Retriever: tạo job, upload file,
poll status, nhận SSE events và hiển thị text, metadata, raw result cùng
embedding preview.

## 3. Self-host NIM core

Container: `retriever`, `vectordb`, `nim-page-elements`, `nim-table-structure`,
`nim-ocr`, `nim-embedding`.

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

NIM_PAGE_ELEMENTS_GPU_ID=0 \
NIM_TABLE_STRUCTURE_GPU_ID=0 \
NIM_OCR_GPU_ID=0 \
NIM_EMBED_GPU_ID=0 \
INSTALL_FFMPEG=true \
RETRIEVER_HTTP_PORT=7780 \
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up -d
```

Pipeline mặc định: Page Elements v3 → Table Structure v1 → Nemotron OCR v2.

### Chọn Option 2: Page Detect + định tuyến ngôn ngữ

Option 2 giữ flow PDF/native/CSV của pipeline chính: Page Elements → Table
Structure → Vintern probe một lần trên mỗi trang → route toàn bộ bbox của
trang sang Vintern hoặc Nemotron OCR. Vintern qua vLLM ổn định ở 2
request đồng thời với GPU cap 0.30. Model Vintern dùng cache tại
`./cache/vintern/model`; image vLLM dùng lại `nemo-retriever-service-gpu:dev`.

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile vintern \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --no-build -d vintern-ocr

docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core --profile vintern \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --no-build -d retriever vectordb nim-page-elements nim-table-structure nim-ocr nim-embedding vintern-ocr
```

`vintern-ocr` chạy bằng vLLM, mount model tại `./cache/vintern/model` và runtime
cache tại `./cache/vintern/runtime`; không tải image/model mới.

`nim-embedding` chỉ tạo vector sau ingest; không tham gia detect/OCR.

### Tesseract 5 — chỉ dành cho pipeline fusion khác

Bật thêm `tesseract` khi dùng pipeline fusion khác; Option 2 không dùng service này.

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core --profile tesseract \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up -d
```

Tesseract API nội bộ: `http://tesseract:8000/v1/ocr`.

## 4. Local Hugging Face models

Container: `retriever`, `vectordb`.

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/local-models.env \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -f nemo_retriever/dev/compose/service-mode.local-models.compose.yaml \
  up --build -d retriever vectordb
```

## 5. Harness Portal — frontend test

Portal chỉ là server nhận file, tạo job và hiển thị trạng thái. Lệnh này chưa
chạy Harness Runner.

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  build retriever

docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps -p 8101:8101 \
  --entrypoint uvicorn retriever \
  nemo_retriever.harness.portal.app:app \
  --host 0.0.0.0 --port 8101 \
  --reload --reload-dir /workspace/nemo_retriever/src
```

Mở frontend tại `http://localhost:8101`.

## 6. Harness Runner — không có trong HEAD hiện tại

Đây là nguyên nhân file upload từ Portal bị `pending`, không phải do thiếu
tham số Docker:

- Runner từng tồn tại trong source cũ, gồm `harness/runner.py` và wiring
  `runner start` trong CLI.
- Commit `2179cf6b` (`Revamp Retriever harness`) đã xóa `runner.py` cùng
  `harness/run.py` và thay Harness bằng CLI benchmark artifact-first.
- HEAD hiện tại không có `harness/runner.py`, `runner_app` hoặc service
  `runner` trong Compose.
- Các API `/api/runners/*` vẫn còn trong Portal để tương thích với runner cũ.
- Dòng hướng dẫn `retriever harness runner start` trong giao diện là stale UI,
  không phải lệnh chạy được với image hiện tại.

Vì vậy lệnh dưới đây **sẽ không chạy được trong source/image hiện tại**:

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps \
  --entrypoint retriever retriever \
  harness runner start \
  --manager-url http://host.docker.internal:8101
```

Không thể sửa đúng bằng cách thêm một service Compose đơn giản: cần khôi phục
đồng bộ cả implementation Runner và các module Harness cũ mà nó gọi. Dùng
`runner.py` từ commit cũ chép riêng vào HEAD hiện tại cũng không đủ.

Kiểm tra trạng thái Runner của Portal:

```bash
curl http://localhost:8101/api/runners
```

Kết quả `[]` nghĩa là chưa có agent nào đăng ký. Hai pool `Realtime/Batch`
trong Retriever không đăng ký vào API này.

## 7. Ingest trực tiếp qua Docker — đường chạy được với source hiện tại

Đường này bỏ qua Portal queue, gửi file trực tiếp vào Retriever service đang
chạy. Với file nằm trong repository, đường dẫn trong container bắt đầu bằng
`/workspace`.

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps \
  --entrypoint retriever retriever \
  ingest service /workspace/data/<file>.pdf \
  --service-url http://host.docker.internal:7780 \
  --no-quiet
```

Retriever service sẽ phân phối request vào các pool Realtime/Batch hiện có,
sau đó gọi các NIM đã bật. Đây là pipeline ingest thật nhưng không tạo job
trong Portal.

## 8. Harness CLI — chạy benchmark một lần, không dùng Portal queue

Đây là đường chạy trực tiếp của Harness CLI. Nó không xử lý job đã upload trên
Portal và cần benchmark/runfile hợp lệ.

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps \
  --entrypoint retriever retriever \
  harness --help
```

## 9. Logs, status, stop

```bash
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml ps

docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  logs -f retriever vectordb

docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml down
```

## 10. Optional NIM profiles

Không cần cho ingest PDF cơ bản.

```bash
# Reranker
docker compose --profile nim-reranker \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d nim-reranker

# Nemotron Parse
docker compose --profile nim-parse \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d nim-nemotron-parse

# Caption 30B
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nim-caption.env \
  --profile nim-caption \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d nim-caption

# Answer LLM
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nim-answer.env \
  --profile nim-answer \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d nim-answer

# Parakeet ASR
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nim-audio.env \
  --profile nim-audio \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d nim-audio
```




http://localhost:7780/v1/dashboard/?v=9#jobs
