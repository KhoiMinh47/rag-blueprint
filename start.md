# NeMo-Retriever — Docker start commands

> Chạy các lệnh từ thư mục gốc repository. Chưa chạy tự động.
>
> File này chỉ build khi Docker daemon trên server chưa có image mà Compose
> cần. Nếu image đã tồn tại, Compose chạy với --no-build để dùng lại image
> và container hiện có. Dấu -- phân cách option của Compose với danh sách
> service cần start.

## 0. Kiểm tra trước khi chạy

Chạy từ thư mục gốc repository:

```bash
docker version
docker compose version
```

Docker Compose của source này cần version 2.23.1 trở lên. Các profile
self-host NIM/model cần NVIDIA Container Toolkit và GPU NVIDIA:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Build policy dùng trong file này

Chạy block dưới đây một lần trong terminal hiện tại. Hàm sẽ resolve image từ
Compose config, kiểm tra image nào còn thiếu, sau đó chọn --build hoặc
--no-build.

```bash
compose_up_if_missing() {
  local -a compose_args=()
  local -a services=()
  local -a images=()
  local after_separator=0
  local image
  local missing=0

  while (($#)); do
    if [[ "$1" == "--" ]]; then
      after_separator=1
    elif ((after_separator)); then
      services+=("$1")
    else
      compose_args+=("$1")
    fi
    shift
  done

  mapfile -t images < <(
    docker compose "${compose_args[@]}" config --images | sort -u
  )

  if ((${#images[@]} == 0)); then
    echo "Không resolve được image từ Compose config; dừng để tránh chạy sai." >&2
    return 1
  fi

  for image in "${images[@]}"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      missing=1
      echo "Thiếu image: $image"
    fi
  done

  if ((missing)); then
    echo "Có image chưa tồn tại trên server -> chạy up --build -d"
    docker compose "${compose_args[@]}" up --build -d "${services[@]}"
  else
    echo "Đã có đủ image trên server -> chạy up --no-build -d"
    docker compose "${compose_args[@]}" up --no-build -d "${services[@]}"
  fi
}
```

Policy này ưu tiên image đang có trên server. Nếu source hoặc Dockerfile vừa
thay đổi nhưng image vẫn tồn tại, chạy build thủ công:

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  build retriever
```

## 1. Hosted API — ingest/search cơ bản

Container chạy: retriever, vectordb.

Pipeline dùng endpoint hosted của NVIDIA cho Page Elements, Table Structure,
OCR và embedding. Dữ liệu tài liệu sẽ được gửi ra endpoint hosted.

```bash
export NVIDIA_API_KEY=nvapi-...

compose_up_if_missing \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- retriever vectordb
```

Kiểm tra health:

```bash
curl -fsSL http://localhost:7670/v1/health
```

## 2. Frontend dashboard — debug ingest

Không có container frontend riêng; dashboard được serve bởi retriever.

```bash
(
  export NVIDIA_API_KEY=nvapi-...
  export RETRIEVER_HTTP_PORT=7780

  compose_up_if_missing \
    -f nemo_retriever/dev/compose/service-mode.compose.yaml \
    -- retriever vectordb
)
```

Mở dashboard:

```text
http://localhost:7780/v1/dashboard/?v=1#ingest
http://localhost:7780/v1/dashboard/?v=1#jobs
```

Dashboard có thể upload file, tạo job, poll trạng thái, nhận SSE events và
hiển thị text, metadata, raw result cùng embedding preview.

## 3. Self-host NIM core

Container chạy: retriever, vectordb, nim-page-elements,
nim-table-structure, nim-ocr, nim-embedding.

NIM image được pull từ NGC. Model engine/cache được tải vào thư mục cache
local, không nằm trong repository.

Đăng nhập NGC trước khi start:

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
```

Start core pipeline:

```bash
(
  export NIM_PAGE_ELEMENTS_GPU_ID=0
  export NIM_TABLE_STRUCTURE_GPU_ID=0
  export NIM_OCR_GPU_ID=0
  export NIM_EMBED_GPU_ID=0
  export INSTALL_FFMPEG=true
  export RETRIEVER_HTTP_PORT=7780

  compose_up_if_missing \
    --env-file nemo_retriever/dev/compose/presets/nims-core.env \
    --profile nims-core \
    --profile table-structure \
    -f nemo_retriever/dev/compose/service-mode.compose.yaml \
    -- retriever vectordb \
       nim-page-elements nim-table-structure nim-ocr nim-embedding
)
```

Các service trong core:

- nim-page-elements: detect layout/page regions.
- nim-table-structure: detect cell, row và column của bảng.
- nim-ocr: Nemotron OCR v2 multilingual.
- nim-embedding: tạo vector embedding sau ingest.

Kiểm tra trạng thái:

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  ps

curl -fsSL http://localhost:7780/v1/health
```

## 4. Option 2 — Page Detect + định tuyến ngôn ngữ qua Vintern

Flow: Page Elements -> Table Structure -> Vintern probe một lần trên mỗi
trang -> route bbox của trang sang Vintern hoặc Nemotron OCR.

Vintern model được tải riêng từ Hugging Face và phải có sẵn ở
cache/vintern/model:

```bash
mkdir -p cache/vintern/model
hf download 5CD-AI/Vintern-1B-v3_5 \
  --local-dir cache/vintern/model
```

vintern-ocr dùng image GPU local của project. Build image một lần nếu image
chưa tồn tại:

```bash
if ! docker image inspect nemo-retriever-service-gpu:dev >/dev/null 2>&1; then
  docker build --target service-gpu \
    -t nemo-retriever-service-gpu:dev .
fi
```

Start toàn bộ core và Vintern:

```bash
(
  export RETRIEVER_HTTP_PORT=7780
  export VINTERN_GPU_ID=0

  compose_up_if_missing \
    --env-file nemo_retriever/dev/compose/presets/nims-core.env \
    --profile nims-core \
    --profile table-structure \
    --profile vintern \
    -f nemo_retriever/dev/compose/service-mode.compose.yaml \
    -- retriever vectordb \
       nim-page-elements nim-table-structure nim-ocr nim-embedding \
       vintern-ocr
)
```

vintern-ocr dùng port host 8013, model mount read-only từ
cache/vintern/model và runtime cache ở cache/vintern/runtime. Preset core đã
trỏ VINTERN_URL vào http://vintern-ocr:8000/v1/chat/completions.

## 5. Option 2 — PP-OCRv6 official

Nhánh này dùng service ppocrv6-official với các model PP-OCRv6, orientation
và unwarping. Start cùng core:

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  --profile ppocrv6 \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- retriever vectordb \
     nim-page-elements nim-table-structure nim-ocr nim-embedding \
     ppocrv6-official
```

ppocrv6-official publish host port 8012. Các service ppocrv6-det và
ppocrv6-rec chỉ cần start thêm khi chạy pipeline experimental dùng detector
và recognizer tách riêng:

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  --profile ppocrv6 \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- ppocrv6-det ppocrv6-rec
```

Model và cache PP-OCRv6 nằm dưới cache/ppocrv6.

## 6. Option 3/5 — VietOCR tiếng Việt

Container chạy: retriever, vectordb, bốn core NIM và vietocr-ocr.

VietOCR sidecar dùng model vgg_seq2seq mặc định và tải weight vào
cache/vietocr ở lần chạy đầu:

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/option3-vietocr.env \
  --profile nims-core \
  --profile table-structure \
  --profile vietocr \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- retriever vectordb \
     nim-page-elements nim-table-structure nim-ocr nim-embedding \
     vietocr-ocr
```

vietocr-ocr publish host port 8014 và expose API nội bộ tại
http://vietocr-ocr:8000/v1/ocr.

## 7. Tesseract 5 — pipeline fusion tùy chọn

Tesseract là CPU sidecar, không cần cho core hoặc Option 2 Vintern:

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  --profile tesseract \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- retriever vectordb \
     nim-page-elements nim-table-structure nim-ocr nim-embedding \
     tesseract
```

Tesseract API nội bộ: http://tesseract:8000/v1/ocr. Host port mặc định là
8011.

## 8. Option 6 — Qwen3.5 qua vLLM

Model preset NVFP4:

```bash
hf download AxionML/Qwen3.5-2B-NVFP4 \
  --cache-dir cache/huggingface
```

Qwen sidecar dùng image GPU local:

```bash
if ! docker image inspect nemo-retriever-service-gpu:dev >/dev/null 2>&1; then
  docker build --target service-gpu \
    -t nemo-retriever-service-gpu:dev .
fi
```

Start Qwen cùng core:

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/option6-qwen-nvfp4.env \
  --profile nims-core \
  --profile table-structure \
  --profile qwen35-nvfp4 \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -f nemo_retriever/dev/compose/qwen35-vllm.compose.yaml \
  -- retriever vectordb \
     nim-page-elements nim-table-structure nim-ocr nim-embedding \
     qwen35-nvfp4
```

qwen35-nvfp4 publish host port 8015 và alias nội bộ vintern-ocr để tái sử
dụng wiring Option 2. Không chạy đồng thời Qwen, Vintern và Ministral trên
GPU nhỏ.

Có thể thay preset bằng option6-qwen-fp8.env hoặc option6-qwen-bf16.env;
kiểm tra OPTION6_VLLM_MODEL_PATH nếu snapshot local không trùng preset.

## 9. Option 7 — Ministral 3B FP8

Tải model vào Hugging Face cache:

```bash
hf download mistralai/Ministral-3-3B-Instruct-2512 \
  --cache-dir cache/huggingface
```

Build image GPU nếu cần và start service:

```bash
if ! docker image inspect nemo-retriever-service-gpu:dev >/dev/null 2>&1; then
  docker build --target service-gpu \
    -t nemo-retriever-service-gpu:dev .
fi

compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  --profile ministral-fp8 \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- retriever vectordb \
     nim-page-elements nim-table-structure nim-ocr nim-embedding \
     ministral-fp8
```

ministral-fp8 publish host port 8016. Nếu snapshot model khác giá trị mặc
định, đặt MINISTRAL_VLM_MODEL_PATH trỏ tới thư mục snapshot thực tế.

## 10. Local Hugging Face models

Mode này chạy extraction, embedding và ASR bằng model local trong image
service-gpu; không dùng profile nims-core. Hai mode là mutually exclusive.

Build image GPU:

```bash
if ! docker image inspect nemo-retriever-service-gpu:dev >/dev/null 2>&1; then
  docker build --target service-gpu \
    -t nemo-retriever-service-gpu:dev .
fi
```

Start container retriever và vectordb:

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/local-models.env \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -f nemo_retriever/dev/compose/service-mode.local-models.compose.yaml \
  -- retriever vectordb
```

Model Hugging Face được lưu trong cache/huggingface. Không thêm thư mục cache
hoặc weight vào Git.

## 11. Harness Portal — frontend test

Portal chỉ là server nhận file, tạo job và hiển thị trạng thái. Lệnh này chưa
chạy Harness Runner.

Đảm bảo retriever image tồn tại:

```bash
if ! docker image inspect "${NEMO_RETRIEVER_IMAGE:-nemo-retriever-service:dev}" \
  >/dev/null 2>&1; then
  docker compose \
    -f nemo_retriever/dev/compose/service-mode.compose.yaml \
    build retriever
fi
```

Start Portal:

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps -p 8101:8101 \
  --entrypoint uvicorn retriever \
  nemo_retriever.harness.portal.app:app \
  --host 0.0.0.0 --port 8101 \
  --reload --reload-dir /workspace/nemo_retriever/src
```

Mở frontend tại http://localhost:8101.

## 12. Harness Runner — không có trong HEAD hiện tại

Runner cũ đã bị xóa khỏi source hiện tại. Portal vẫn còn API
/api/runners/* để tương thích với runner cũ, nhưng HEAD hiện tại không có
harness/runner.py, runner_app hoặc service runner trong Compose.

Vì vậy lệnh dưới đây sẽ không chạy được:

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps \
  --entrypoint retriever retriever \
  harness runner start \
  --manager-url http://host.docker.internal:8101
```

Kiểm tra trạng thái runner của Portal:

```bash
curl http://localhost:8101/api/runners
```

Kết quả [] nghĩa là chưa có agent nào đăng ký. Hai pool Realtime/Batch của
Retriever không tự đăng ký vào API này.

## 13. Ingest trực tiếp qua Docker — đường chạy được

Đường này bỏ qua Portal queue và gửi file trực tiếp vào Retriever service.
File nằm trong repository được mount vào container tại /workspace.

Nếu đang dùng hosted stack, service URL là port 7670. Nếu đang dùng preset
nims-core.env, thay bằng port 7780:

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps \
  --entrypoint retriever retriever \
  ingest service /workspace/data/<file>.pdf \
  --service-url http://host.docker.internal:7780 \
  --no-quiet
```

Retriever sẽ phân phối request vào các pool Realtime/Batch và gọi các NIM đã
bật. Lệnh này không tạo job trong Portal.

## 14. Harness CLI — benchmark một lần

Harness CLI chạy trực tiếp, không xử lý job đã upload trên Portal. Kiểm tra
CLI và các command hiện có:

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps \
  --entrypoint retriever retriever \
  harness --help
```

Benchmark cần runfile/dataset hợp lệ theo cấu hình của project.

## 15. Optional NIM profiles

Các NIM này không cần cho ingest PDF core. Chúng là service lifecycle/API;
muốn Retriever tự gọi chúng phải layer preset và endpoint tương ứng.

```bash
# Reranker — host port 8005
compose_up_if_missing \
  --profile nim-reranker \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-reranker

# Nemotron Parse — host port 8006
compose_up_if_missing \
  --profile nim-parse \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-nemotron-parse

# Caption 30B — host port 8007
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nim-caption.env \
  --profile nim-caption \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-caption

# Answer LLM — host port 8008
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nim-answer.env \
  --profile nim-answer \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-answer

# Parakeet ASR — HTTP 9000, gRPC 50051
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nim-audio.env \
  --profile nim-audio \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-audio
```

## 16. Observability, Neo4j và local judge

OpenTelemetry Collector và Zipkin:

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/observability.env \
  --profile observability \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- otel-collector zipkin
```

Zipkin UI: http://localhost:9411. Metrics: http://localhost:8889/metrics.

Neo4j:

```bash
export NEO4J_PASSWORD=change-me
docker compose \
  -f nemo_retriever/dev/compose/neo4j.compose.yaml \
  up -d neo4j
```

Local judge:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
docker compose \
  -f nemo_retriever/dev/compose/judge.compose.yaml \
  up -d judge
```

## 17. Port map

Port bên trái là port host mặc định; port trong ngoặc là port bên trong
container. URL dạng http://service:8000/... chỉ dùng giữa các container.

| Host port | Service | Ý nghĩa |
| ---: | --- | --- |
| 7670 | retriever | FastAPI ingest/query/dashboard |
| 7671 | vectordb | LanceDB/vector service |
| 7780 | retriever với nims-core.env | FastAPI self-host stack |
| 8001 | nim-page-elements:8000 | Page Elements v3 |
| 8002 | nim-table-structure:8000 | Table Structure v1 |
| 8003 | nim-ocr:8000 | Nemotron OCR v2 |
| 8004 | nim-embedding:8000 | Multimodal embedding |
| 8005 | nim-reranker:8000 | Optional reranking |
| 8006 | nim-nemotron-parse:8000 | Optional parser |
| 8007 | nim-caption:8000 | Optional captioning |
| 8008 | nim-answer:8000 | Optional answer LLM |
| 8009 | ppocrv6-rec:8000 | PP-OCRv6 recognizer |
| 8010 | ppocrv6-det:8000 | PP-OCRv6 detector |
| 8011 | tesseract:8000 | CPU Tesseract sidecar |
| 8012 | ppocrv6-official:8000 | Official PaddleOCR pipeline |
| 8013 | vintern-ocr:8000 | Vintern vLLM sidecar |
| 8014 | vietocr-ocr:8000 | VietOCR sidecar |
| 8015 | qwen35-nvfp4:8000 | Qwen vLLM sidecar |
| 8016 | ministral-fp8:8000 | Ministral vLLM sidecar |
| 8118 | paddleocr-vl-api:8080 | PaddleOCR-VL API |
| 9000 | nim-audio:9000 | Parakeet HTTP |
| 50051 | nim-audio:50051 | Parakeet gRPC |
| 4317 | otel-collector:4317 | OTLP gRPC |
| 4318 | otel-collector:4318 | OTLP HTTP |
| 8889 | otel-collector:8889 | Prometheus metrics |
| 9411 | zipkin:9411 | Trace UI/API |

Optional service chỉ bind port khi profile tương ứng được bật. Có thể đổi
host port bằng biến *_HOST_PORT, ví dụ RETRIEVER_HTTP_PORT=7780 hoặc
NIM_OCR_HOST_PORT=18003.

## 18. Logs, status và stop

Default hosted stack:

```bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  ps

docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  logs -f retriever vectordb

docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  down
```

Core self-host stack:

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  ps

docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  logs -f retriever vectordb nim-page-elements nim-table-structure nim-ocr nim-embedding

docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  down
```

Các thư mục cache/model/database bind-mounted sẽ còn lại sau down. Chỉ xóa
cache khi thực sự muốn tải model hoặc tạo database lại từ đầu.
