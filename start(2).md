# NeMo-Retriever — Docker start commands

> Chạy các lệnh từ thư mục gốc repository. Chưa chạy tự động.
>
> File này chỉ build khi Docker daemon trên server chưa có một trong các
> image mà Compose cần. Nếu image đã tồn tại, Compose chạy với `--no-build` để
> dùng lại image/container hiện có. Vì hai người đang dùng chung server và
> Docker daemon, image đã được build bởi một người sẽ được người còn lại dùng
> lại.

## Build policy dùng trong file này

Chạy block dưới đây một lần trong terminal hiện tại. Dấu `--` phân cách các
option của Compose với danh sách service cần start.

```bash
compose_up_if_missing() {
  local -a compose_args=()
  local -a services=()
  local -a images=()
  local after_separator=0
  local image missing=0

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

Lưu ý: policy này cố ý ưu tiên image đang có trên server. Nếu source hoặc
`Dockerfile` đã thay đổi nhưng image vẫn tồn tại, lệnh sẽ không tự build lại;
muốn cập nhật image thì chạy `docker compose ... build` hoặc dùng file gốc
`start.md`.

## 1. Hosted API — ingest/search cơ bản

Container: `retriever`, `vectordb`.

```bash
export NVIDIA_API_KEY=nvapi-...

compose_up_if_missing \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- retriever vectordb
```

## 2. Frontend debug ingest

Không cần container frontend riêng; dashboard được serve bởi `retriever`.

```bash
(
  export RETRIEVER_HTTP_PORT=7780
  compose_up_if_missing \
    -f nemo_retriever/dev/compose/service-mode.compose.yaml \
    -- retriever vectordb
)
```

Mở: `http://localhost:7780/v1/dashboard/?v=1#ingest`

Frontend này gọi trực tiếp API ingest của Retriever: tạo job, upload file,
poll status, nhận SSE events và hiển thị text, metadata, raw result cùng
embedding preview.

## 3. Self-host NIM core

Container: `retriever`, `vectordb`, `nim-page-elements`,
`nim-table-structure`, `nim-ocr`, `nim-embedding`.

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

(
  export NIM_PAGE_ELEMENTS_GPU_ID=0
  export NIM_TABLE_STRUCTURE_GPU_ID=0
  export NIM_OCR_GPU_ID=0
  export NIM_EMBED_GPU_ID=0
  export RETRIEVER_HTTP_PORT=7780
  compose_up_if_missing \
    --env-file nemo_retriever/dev/compose/presets/nims-core.env \
    --profile nims-core \
    -f nemo_retriever/dev/compose/service-mode.compose.yaml \
    --
)
```

## 4. Local Hugging Face models

Container: `retriever`, `vectordb`.

```bash
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/local-models.env \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -f nemo_retriever/dev/compose/service-mode.local-models.compose.yaml \
  -- retriever vectordb
```

## 5. Harness Portal — frontend test

Portal chỉ là server nhận file, tạo job và hiển thị trạng thái. Lệnh này chưa
chạy Harness Runner.

```bash
if docker image inspect "${NEMO_RETRIEVER_IMAGE:-nemo-retriever-service:dev}" \
  >/dev/null 2>&1; then
  echo "Đã có Retriever image -> bỏ qua build"
else
  docker compose \
    -f nemo_retriever/dev/compose/service-mode.compose.yaml \
    build retriever
fi

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
compose_up_if_missing \
  --profile nim-reranker \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-reranker

# Nemotron Parse
compose_up_if_missing \
  --profile nim-parse \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-nemotron-parse

# Caption 30B
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nim-caption.env \
  --profile nim-caption \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-caption

# Answer LLM
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nim-answer.env \
  --profile nim-answer \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-answer

# Parakeet ASR
compose_up_if_missing \
  --env-file nemo_retriever/dev/compose/presets/nim-audio.env \
  --profile nim-audio \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -- nim-audio
```




http://localhost:7780/v1/dashboard/?v=9#jobs
