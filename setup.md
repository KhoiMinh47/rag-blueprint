# NeMo Retriever custom fork — setup guide

Tài liệu này mô tả cách dựng project từ một checkout sạch. Repository **không
chứa model weights, NIM cache, Docker image, virtualenv hoặc dữ liệu runtime**;
các thành phần đó được tải và cấu hình ở máy chạy theo hướng dẫn bên dưới.

Source chính là thư viện và service NeMo Retriever đã được mở rộng cho các
pipeline OCR thử nghiệm, gồm OCR tiếng Việt, spreadsheet ingest, dashboard
debug, các profile Docker Compose và pipeline Qwen/Mistral. Các pipeline mới
không thay thế pipeline NVIDIA mặc định; hãy chọn selector/profile tương ứng.

## 1. Tài liệu và liên kết quan trọng

- [NeMo Retriever Documentation](https://docs.nvidia.com/nemo/retriever/latest/)
- [Prerequisites và support matrix](https://docs.nvidia.com/nemo/retriever/latest/extraction/prerequisites/)
- [Deployment options](https://docs.nvidia.com/nemo/retriever/latest/extraction/deployment-options/)
- [Authentication và API keys](https://docs.nvidia.com/nemo/retriever/latest/extraction/api-keys/)
- [NVIDIA NIM catalog](https://catalog.ngc.nvidia.com/)
- [NVIDIA NIM Operator](https://docs.nvidia.com/nim-operator/latest/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [NVIDIA build API](https://build.nvidia.com/)
- [Hugging Face](https://huggingface.co/)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

Chi tiết Compose đầy đủ hơn nằm tại
[`nemo_retriever/dev/compose/README.md`](nemo_retriever/dev/compose/README.md),
còn deployment Kubernetes nằm tại
[`nemo_retriever/helm/README.md`](nemo_retriever/helm/README.md).

## 2. Yêu cầu máy chạy

### Bắt buộc cho Docker + GPU/NIM

- Linux, khuyến nghị Ubuntu 22.04 trở lên.
- Docker Engine, Docker Compose v2 (source này yêu cầu Compose >= 2.23.1) và
  Docker Buildx.
- NVIDIA Driver tương thích CUDA; tài liệu NVIDIA hiện khuyến nghị driver
  >= 535 và CUDA >= 12.2.
- NVIDIA Container Toolkit, kiểm tra bằng:

  ~~~bash
  docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
  ~~~

- Python 3.12 nếu chạy trực tiếp bằng `uv`/CLI. Version được khai báo trong
  [`nemo_retriever/pyproject.toml`](nemo_retriever/pyproject.toml).
- `git`, `curl` và quyền chạy Docker.

Kiểm tra nhanh:

~~~bash
docker version
docker compose version
docker buildx version
python3 --version
~~~

### Tài nguyên khuyến nghị

Pipeline core cần GPU NVIDIA. Một GPU A10G/L40S/A100 hoặc tương đương là
điểm bắt đầu thực tế; model/profile tự host có thể cần nhiều VRAM hơn. Core
NIM và cache thường cần khoảng 150 GB dung lượng trống theo support matrix;
Qwen, Mistral, Vintern, PaddleOCR và cache compile có thể cần thêm hàng chục
GB. CPU/RAM càng nhiều thì số worker ingest có thể càng cao.

Không nên bật đồng thời toàn bộ `nims-core`, Qwen/Mistral, VietOCR và
PaddleOCR trên một GPU nhỏ. Các preset hiện tại mặc định dùng GPU `0`; đổi
`*_GPU_ID` hoặc chỉ bật một profile tại một thời điểm nếu máy chỉ có một GPU.

## 3. Clone source

~~~bash
git clone https://github.com/KhoiMinh47/rag-blueprint.git
cd rag-blueprint
~~~

Các thư mục cache sau đây sẽ tự xuất hiện khi chạy; không cần tạo hoặc commit
chúng:

~~~text
cache/retriever/       # log và state service
cache/vectordb/        # LanceDB
cache/nim/             # NIM model cache
cache/huggingface/     # Hugging Face cache
cache/qwen35-vllm/     # vLLM/FlashInfer/TorchInductor cache
cache/vintern/         # Vintern model và runtime cache
cache/vietocr/         # VietOCR cache
~~~

## 4. Credential — hai loại key khác nhau

### Hosted NVIDIA NIM

Dùng `NVIDIA_API_KEY` để gọi endpoint hosted của NVIDIA (`ai.api.nvidia.com`
và `integrate.api.nvidia.com`). Tạo key tại [build.nvidia.com](https://build.nvidia.com/)
rồi đặt trong terminal:

~~~bash
export NVIDIA_API_KEY='nvapi-...'
~~~

### Self-host NIM trên NGC

Dùng `NGC_API_KEY` để đăng nhập `nvcr.io`, pull NIM container và tải model
artifact. Tạo NGC personal key tại
<https://org.ngc.nvidia.com/setup/api-keys>, sau đó:

~~~bash
export NGC_API_KEY='nvapi-...'
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
~~~

`NVIDIA_API_KEY` và `NGC_API_KEY` có mục đích khác nhau; không ghi key thật
vào `.env`, Compose file, Git history hoặc issue. Nếu dùng model gated trên
Hugging Face, đăng nhập thêm:

~~~bash
python3 -m pip install --upgrade 'huggingface_hub[cli]'
hf auth login
~~~

## 5. Cách chạy nhanh — hosted NIM, không tự host model

Đây là cách nhẹ nhất: chỉ build hai service `retriever` và `vectordb`; OCR,
Page Elements và embedding gọi endpoint hosted. Không cần tự tải NIM image hay
model weight, nhưng dữ liệu tài liệu sẽ đi ra endpoint NVIDIA — chỉ dùng khi
chính sách dữ liệu cho phép.

~~~bash
export NVIDIA_API_KEY='nvapi-...'

docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d retriever vectordb

curl -fsSL http://localhost:7670/v1/health
~~~

Mở dashboard ingest tại
<http://localhost:7670/v1/dashboard/?v=1#ingest>.

Muốn dùng port host `7780` giống preset self-host:

~~~bash
RETRIEVER_HTTP_PORT=7780 \
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d retriever vectordb
~~~

Trong Compose, `retriever` chạy FastAPI ở port container `7670`, còn
`vectordb` chạy ở `7671` trên network nội bộ. Chỉ port retriever được publish
ra host trong stack mặc định.

## 6. Self-host bốn NIM core bằng Docker Compose

### Thành phần core

| Thành phần | NIM image mặc định | Model/role |
| --- | --- | --- |
| Page Elements | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.0` | `nvidia/nemotron-page-elements-v3`, layout/region detection |
| Table Structure | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.0` | `nvidia/nemotron-table-structure-v1`, cell/row/column geometry |
| OCR | `nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0.0` | `nvidia/nemotron-ocr-v2`, multilingual OCR |
| Embedding | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` | `nvidia/llama-nemotron-embed-vl-1b-v2`, vector embedding |

Các image trên được pull từ NGC; model engine/cache được NIM tải vào
`cache/nim/`. Chúng không nằm trong Git repository.

### Khởi động core

~~~bash
export NGC_API_KEY='nvapi-...'
echo "$NGC_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin

docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d \
  retriever vectordb \
  nim-page-elements nim-table-structure nim-ocr nim-embedding
~~~

Preset `nims-core.env` đã nối các endpoint nội bộ:

~~~text
Page Elements    http://nim-page-elements:8000/v1/page-elements
Table Structure  http://nim-table-structure:8000/v1/table-structure
OCR              http://nim-ocr:8000/v1/ocr
Embedding        http://nim-embedding:8000/v1/embeddings
~~~

Chờ đến khi các container healthy:

~~~bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml ps

curl -fsSL http://localhost:7780/v1/health
~~~

Nếu máy có nhiều GPU, chỉnh `NIM_PAGE_ELEMENTS_GPU_ID`,
`NIM_TABLE_STRUCTURE_GPU_ID`, `NIM_OCR_GPU_ID` và `NIM_EMBED_GPU_ID` trong
shell hoặc tạo env file riêng. Không commit env file có secret.

## 7. Các pipeline/model tùy chọn

Các pipeline dưới đây là experimental/development. Hãy tải đúng model trước,
build image phù hợp và tránh chạy đồng thời các model nặng trên cùng GPU.

### Option 3/5 — Nemotron + VietOCR tiếng Việt

Model recognizer mặc định là `vgg_seq2seq` từ [VietOCR](https://github.com/pbcquoc/vietocr).
Image sidecar sẽ cài package và tải weight vào `cache/vietocr/` ở lần chạy
đầu. Khởi động cùng core NIM:

~~~bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/option3-vietocr.env \
  --profile nims-core --profile table-structure --profile vietocr \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d \
  retriever vectordb \
  nim-page-elements nim-table-structure nim-ocr nim-embedding vietocr-ocr
~~~

VietOCR endpoint nội bộ là `http://vietocr-ocr:8000/v1/ocr`; host port mặc
định là `8014`.

### Option 2 — PP-OCRv6/Vintern hoặc PaddleOCR-VL

PP-OCRv6 dùng các model PaddleOCR và cache dưới `cache/ppocrv6/`:

~~~bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core --profile table-structure --profile ppocrv6 \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d \
  retriever vectordb \
  nim-page-elements nim-table-structure nim-ocr nim-embedding \
  ppocrv6-official ppocrv6-det ppocrv6-rec
~~~

Các model chính là `PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`; official
pipeline còn dùng orientation/unwarping models được khai báo trong Compose.

Vintern-1B-v3.5 là model Hugging Face của [5CD-AI](https://huggingface.co/5CD-AI/Vintern-1B-v3_5).
Compose yêu cầu model ở dạng thư mục tại `cache/vintern/model`:

~~~bash
mkdir -p cache/vintern/model
hf download 5CD-AI/Vintern-1B-v3_5 \
  --local-dir cache/vintern/model
~~~

Sau đó bật sidecar vLLM:

~~~bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core --profile table-structure --profile vintern \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d vintern-ocr
~~~

Vintern dùng host port `8013`, tối đa hai sequence theo preset và
`gpu-memory-utilization=0.30`. Model weights không được commit.

PaddleOCR-VL là nhánh riêng, dùng image PaddleOCR-VL và model
`PaddleOCR-VL-1.6-0.9B`; cần cache PaddleX/Hugging Face tại
`cache/paddleocr-vl/` và GPU tương thích. Cấu hình nằm tại
[`nemo_retriever/dev/compose/paddleocr-vl/`](nemo_retriever/dev/compose/paddleocr-vl/):

~~~bash
docker compose \
  --profile paddleocr-vl \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d paddleocr-vl-vlm-server paddleocr-vl-api
~~~

PaddleOCR-VL API publish host port `8118` và dùng port nội bộ `8080` cho
inference.

### Option 6 — Qwen3.5 qua vLLM

Các preset có sẵn:

| Preset | Model |
| --- | --- |
| `option6-qwen-nvfp4.env` | `AxionML/Qwen3.5-2B-NVFP4` |
| `option6-qwen-fp8.env` | `surogate/Qwen3.5-2B-FP8` |
| `option6-qwen-bf16.env` | `Qwen/Qwen3.5-2B` |

Model pages: [Qwen base](https://huggingface.co/Qwen/Qwen3.5-2B),
[NVFP4](https://huggingface.co/AxionML/Qwen3.5-2B-NVFP4),
[FP8](https://huggingface.co/surogate/Qwen3.5-2B-FP8).

Tải cache theo cấu trúc Hugging Face mặc định:

~~~bash
mkdir -p cache/huggingface
hf download AxionML/Qwen3.5-2B-NVFP4 --cache-dir cache/huggingface
~~~

Qwen sidecar dùng image `nemo-retriever-service-gpu:dev`, nên build target GPU
trước:

~~~bash
docker build --target service-gpu \
  -t nemo-retriever-service-gpu:dev .
~~~

Khởi động NVFP4:

~~~bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/option6-qwen-nvfp4.env \
  --profile nims-core --profile table-structure --profile qwen35-nvfp4 \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -f nemo_retriever/dev/compose/qwen35-vllm.compose.yaml \
  up --build -d \
  retriever vectordb \
  nim-page-elements nim-table-structure nim-ocr nim-embedding qwen35-nvfp4
~~~

Qwen endpoint host mặc định là `8015`; bên trong network nó được alias thành
`vintern-ocr` để pipeline Option 6 dùng lại wiring hiện tại. Dừng sidecar
trước khi chuyển từ NVFP4 sang FP8/BF16 để tránh giữ nhiều model trong VRAM.

### Option 7 — Ministral 3B FP8

Model: [`mistralai/Ministral-3-3B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).
Tải vào Hugging Face cache rồi đặt `MINISTRAL_VLM_MODEL_PATH` trỏ tới thư mục
snapshot thực tế nếu hash snapshot khác preset:

~~~bash
hf download mistralai/Ministral-3-3B-Instruct-2512 \
  --cache-dir cache/huggingface
~~~

Sidecar dùng image `nemo-retriever-service-gpu:dev`, chạy profile
`ministral-fp8`, và publish port `8016`. Tham khảo lệnh đầy đủ trong
`nemo_retriever/dev/compose/README.md`.

## 8. NIM tùy chọn cho chức năng nâng cao

Các profile sau không cần cho ingest PDF core và mặc định không tự bật:

| Profile | Service/model | Mục đích | Host port |
| --- | --- | --- | --- |
| `nim-reranker` | `llama-nemotron-rerank-vl-1b-v2:1.11.0` | rerank kết quả | `8005` |
| `nim-parse` | `nemotron-parse-v1.2:1.7.0-variant` | alternate PDF parser | `8006` |
| `nim-caption` | `nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant` | image/video caption | `8007` |
| `nim-answer` | `llama-3.3-nemotron-super-49b-v1.5:2.0.5` | answer/query LLM | `8008` |
| `nim-audio` | `parakeet-1-1b-ctc-en-us:1.5.0` | audio/video ASR | HTTP `9000`, gRPC `50051` |

Ví dụ bật answer NIM cùng core:

~~~bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/nim-answer.env \
  --profile nims-core --profile table-structure --profile nim-answer \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  up --build -d nim-answer
~~~

Các NIM này có cache riêng trong `cache/nim/<service>`. Chỉ bật khi có đủ GPU,
VRAM, dung lượng và key/license phù hợp.

## 9. Chạy local Hugging Face mode

Mode này không dùng các endpoint NIM core cho extraction/embedding mà chạy
model local trong `service-gpu`. Nó **mutually exclusive** với `nims-core`.

~~~bash
docker build --target service-gpu \
  -t nemo-retriever-service-gpu:dev .

docker compose \
  --env-file nemo_retriever/dev/compose/presets/local-models.env \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -f nemo_retriever/dev/compose/service-mode.local-models.compose.yaml \
  up --build -d retriever vectordb
~~~

Model Hugging Face được lưu trong `cache/huggingface/`; lần khởi động đầu có
thể lâu và cần nhiều disk/VRAM. Chỉnh `LOCAL_MODELS_*` và
`LOCAL_EMBED_*` nếu GPU layout khác preset.

## 10. Port map

Port bên trái là port host mặc định; port trong ngoặc là port bên trong
container. Các URL `http://service:8000/...` chỉ dùng giữa các container trong
Docker network, không phải URL trên host.

| Host port | Container/service | Ý nghĩa |
| ---: | --- | --- |
| `7670` | `retriever` | FastAPI ingest/query/dashboard; preset `nims-core` đổi thành `7780` |
| `7671` | `vectordb` | LanceDB/vector service; mặc định chỉ expose nội bộ |
| `8001` | `nim-page-elements` (`8000`) | Page Elements v3 |
| `8002` | `nim-table-structure` (`8000`) | Table Structure v1 |
| `8003` | `nim-ocr` (`8000`) | Nemotron OCR v2 |
| `8004` | `nim-embedding` (`8000`) | Multimodal embedding |
| `8005` | `nim-reranker` (`8000`) | Optional reranking |
| `8006` | `nim-nemotron-parse` (`8000`) | Optional parser |
| `8007` | `nim-caption` (`8000`) | Optional captioning |
| `8008` | `nim-answer` (`8000`) | Optional answer LLM |
| `8009` | `ppocrv6-rec` (`8000`) | PP-OCRv6 recognizer |
| `8010` | `ppocrv6-det` (`8000`) | PP-OCRv6 detector |
| `8011` | `tesseract` (`8000`) | CPU Tesseract sidecar |
| `8012` | `ppocrv6-official` (`8000`) | Official PaddleOCR pipeline |
| `8013` | `vintern-ocr` (`8000`) | Vintern vLLM sidecar |
| `8014` | `vietocr-ocr` (`8000`) | VietOCR sidecar |
| `8015` | `qwen35-nvfp4` (`8000`) | Qwen vLLM sidecar |
| `8016` | `ministral-fp8` (`8000`) | Ministral vLLM sidecar |
| `8118` | `paddleocr-vl-api` (`8080`) | PaddleOCR-VL API |
| `9000` | `nim-audio` (`9000`) | Parakeet HTTP |
| `50051` | `nim-audio` (`50051`) | Parakeet gRPC |
| `4317` | OpenTelemetry Collector (`4317`) | OTLP gRPC |
| `4318` | OpenTelemetry Collector (`4318`) | OTLP HTTP |
| `8889` | OpenTelemetry Collector (`8889`) | Prometheus-format metrics |
| `9411` | Zipkin (`9411`) | Trace UI/API |

Các service optional chỉ bind port khi profile tương ứng được bật. Có thể đổi
host port bằng các biến `*_HOST_PORT`, ví dụ `RETRIEVER_HTTP_PORT=7780` hoặc
`NIM_OCR_HOST_PORT=18003`.

## 11. Ingest/query thử nghiệm

Với stack đang chạy và file mẫu đã mount ở `/workspace/data`:

~~~bash
docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  run --rm --no-deps \
  --entrypoint retriever retriever \
  ingest service /workspace/data/multimodal_test.pdf \
  --service-url http://host.docker.internal:7670 \
  --no-quiet
~~~

Query qua HTTP:

~~~bash
curl -fsSL -X POST http://localhost:7670/v1/query \
  -H 'Content-Type: application/json' \
  --data '{"query":"What is in this document?","top_k":5}'
~~~

Nếu dùng preset `nims-core.env`, thay `7670` bằng `7780` ở URL host.

## 12. Observability, Neo4j và judge (tùy chọn)

- Observability: dùng `observability.env` + profile `observability`; Zipkin ở
  `http://localhost:9411`, metrics ở `http://localhost:8889/metrics`.
- Neo4j: chạy file `nemo_retriever/dev/compose/neo4j.compose.yaml`, đặt
  `NEO4J_PASSWORD` trước khi start; Browser ở `http://localhost:7474`, Bolt ở
  `localhost:7687`.
- Local judge: chạy `judge.compose.yaml`, cần `NGC_API_KEY`; port mặc định
  `8000` và có thể đổi bằng `JUDGE_HTTP_PORT`.

Các state/database/cache của những service này cũng nằm dưới `cache/` và
không được đưa lên GitHub.

## 13. Chạy trực tiếp bằng Python/uv (không dùng Docker)

Chỉ nên dùng cách này khi đã có endpoint model phù hợp hoặc muốn chạy unit
test/library mode. Từ repository root:

~~~bash
cd nemo_retriever
uv python install 3.12
uv sync --extra service --extra multimedia
uv run retriever --help
~~~

Các extra chính:

- `service`: FastAPI, remote-NIM clients và runtime service.
- `local`: PyTorch/Transformers/vLLM và local GPU models.
- `multimedia`: audio/video/SVG support.
- `tabular`: Neo4j/SQL helpers.
- `llm`: answer/judge clients.
- `dev`: pytest/build tooling.

Full dependency graph rất nặng; dùng `uv sync --all-extras` chỉ khi thật sự
cần tất cả feature.

## 14. Kubernetes/Helm cho deployment dài hạn

Docker Compose ở trên dành cho development/single-host. Với Kubernetes:

1. Cài NVIDIA GPU Operator và [NIM Operator](https://docs.nvidia.com/nim-operator/latest/install.html).
2. Chuẩn bị NGC image-pull secret, persistent storage cho model cache và GPU
   node phù hợp.
3. Đọc [`nemo_retriever/helm/README.md`](nemo_retriever/helm/README.md) để
   chọn external NIM endpoints hoặc bật bốn core NIM trong chart.
4. Không paste key thật vào command history nếu môi trường yêu cầu bảo mật;
   dùng Kubernetes Secret/values được quản lý an toàn.

Ví dụ khung lệnh (cần thay registry/tag và secret theo cluster):

~~~bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password="$NGC_API_KEY"
~~~

## 15. Kiểm tra trước khi báo lỗi

~~~bash
# Không start container; chỉ kiểm tra Compose interpolation/YAML.
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml config >/tmp/nemo-compose.yaml

docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml ps

docker compose \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml logs -f retriever vectordb
~~~

Các lỗi thường gặp:

- `403` khi pull `nvcr.io`: chưa `docker login nvcr.io` bằng `NGC_API_KEY`.
- NIM `unhealthy`: model đang tải/compile lần đầu hoặc GPU/VRAM không đủ.
- `connection refused` từ retriever: kiểm tra đúng profile và endpoint nội bộ
  trong preset.
- Upload audio/video lỗi thiếu `ffmpeg`/`ffprobe`: build service image với
  `INSTALL_FFMPEG=true` hoặc cài binary vào image/host theo
  [troubleshooting guide](https://docs.nvidia.com/nemo/retriever/latest/extraction/troubleshoot/).
- Qwen/Vintern/Mistral không start: kiểm tra model path, snapshot hash, quyền
  đọc `cache/` và không để sidecar model khác chiếm VRAM.

## 16. Những gì cố ý không nằm trong repository

Các mục sau được ignore và phải tạo lại ở máy chạy:

- `cache/`, `.cache/`, `.config/`, virtualenv và Python bytecode;
- NIM/Hugging Face/PaddleOCR/VietOCR model weights;
- vLLM, FlashInfer, TorchInductor và Docker BuildKit cache;
- Docker container/image/volume runtime state;
- log runtime và PDF report/generated artifact cục bộ.

Dockerfile, Docker Compose, Helm chart, source Python/JS, test, config không
chứa secret và tài liệu setup vẫn được publish vì đó là phần cần thiết để
người khác dựng lại project.
