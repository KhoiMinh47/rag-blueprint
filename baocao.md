# BÁO CÁO TỔNG HỢP NEMO-RETRIEVER CUSTOM FORK

**Ngày tổng hợp:** 14.08.2026  
**Repository:** https://github.com/KhoiMinh47/rag-blueprint  
**Phạm vi:** kiến trúc source, ingest, OCR, embedding, VectorDB, service,
dashboard, harness, deployment và các kết quả benchmark đang có trong
repository.

## 1. Mục lục

| Phần | Nội dung |
|---:|---|
| 2 | Tóm tắt điều hành |
| 3 | Phạm vi và trạng thái repository |
| 4 | Kiến trúc tổng quan |
| 5 | Định dạng đầu vào và routing |
| 6 | Luồng ingest chi tiết |
| 7 | Hệ thống pipeline và selector |
| 8 | Mô tả từng pipeline OCR |
| 9 | Hậu xử lý, embedding, VectorDB và query |
| 10 | Service API, job system và dashboard |
| 11 | Harness và benchmark framework |
| 12 | Deployment, NIM, model và cache |
| 13 | Port map |
| 14 | Cấu trúc thư mục source |
| 15 | Kết quả benchmark và bằng chứng hiện có |
| 16 | Giới hạn, rủi ro và các điểm cần chú ý |
| 17 | Quy trình setup/run khuyến nghị |
| 18 | Kết luận và lựa chọn pipeline |
| 19 | Tài liệu tham khảo trong repository |

## 2. Tóm tắt điều hành

NeMo Retriever là một framework ingestion cho hệ thống RAG. Project nhận
nhiều loại file, phân tích nội dung, phát hiện vùng semantic như text, title,
table, chart, image và infographic, chạy OCR khi cần, làm sạch/chunk nội dung,
tạo embedding và lưu vào LanceDB để truy vấn.

Custom fork này giữ pipeline NVIDIA/Nemotron làm đường mặc định, đồng thời bổ
sung các nhánh OCR và VLM thử nghiệm:

1. Nemotron OCR v2 với Page Elements v3 và Table Structure v1.
2. Nhánh Option 2 language-routed OCR dùng Nemotron và VietOCR.
3. VietOCR routing detector-free và các biến thể có PP-OCRv6 line detector.
4. Tesseract-first fusion với Nemotron fallback.
5. Qwen3.5 NVFP4/FP8/BF16 qua vLLM.
6. Ministral 3B FP8 qua vLLM.
7. Official PP-OCRv6 và PaddleOCR-VL dưới dạng các adapter/sidecar tùy chọn.

Các pipeline là các route lựa chọn theo job hoặc service configuration; một
job thông thường không gọi tất cả model cùng lúc. Core runtime có thể chạy:

- endpoint NVIDIA hosted;
- NIM tự host bằng Docker Compose;
- model Hugging Face local trong image GPU;
- deployment Kubernetes/Helm cho môi trường dài hạn.

Repository public chỉ chứa source, Dockerfile, Compose/Helm/config, test và
tài liệu. Model weights, NIM cache, Docker image, container/volume state,
virtualenv và dữ liệu runtime không được publish.

## 3. Phạm vi và trạng thái repository

### 3.1. Những gì project giải quyết

Project có ba lớp chức năng:

```text
File/document
  → ingestion và extraction
  → OCR/layout/table/visual processing
  → clean, dedup, chunk
  → embedding
  → LanceDB/vector retrieval
  → query hoặc answer generation
```

Ngoài product workflow, source còn có:

- FastAPI Retriever service và VectorDB service.
- Dashboard để upload, theo dõi job, xem pipeline trace và visual evidence.
- Ray-based batch/inprocess execution.
- Service mode với realtime/batch pool và job tracking.
- Harness cho benchmark/evaluation theo runfile và artifact.
- Helm chart cho deployment Kubernetes.
- Graph Pipeline Registry để đăng ký, build, diff và serialize graph.

### 3.2. Trạng thái tài liệu

Các tài liệu trước đây có hai cách trình bày:

- thongtinbaocao.md tập trung sâu vào ingest/OCR, đặc biệt Option 1 và
  Option 4.
- fullbaocao.md tóm tắt bốn nhóm Qwen, Mistral, NVIDIA và experimental,
  đồng thời ghi benchmark.

baocao.md này hợp nhất hai cách gọi và đối chiếu với selector/source hiện
tại. Một số selector có tên legacy để không phá API/dashboard cũ; vì vậy tên
selector không phải lúc nào cũng trùng hoàn toàn với tên model hoặc tên
pipeline nội bộ.

### 3.3. Những gì không nằm trong repository

Các thành phần sau phải được tạo/tải ở máy chạy:

```text
cache/
.cache/
.config/
NIM model cache
Hugging Face/PaddleOCR/VietOCR weights
vLLM/FlashInfer/TorchInductor cache
Docker image/container/volume runtime state
Generated reports và log runtime
```

Không đưa NVIDIA API key, NGC API key, Hugging Face token, webhook hoặc
machine-local path vào Git.

## 4. Kiến trúc tổng quan

### 4.1. Các lớp chính

| Lớp | Thành phần | Vai trò |
|---|---|---|
| Input | PDF, Office, text, HTML, image, spreadsheet, audio, video | Nhận và phân loại dữ liệu |
| Conversion | LibreOffice, PDFium, image/media loaders | Chuẩn hóa file và render page |
| Layout | Page Elements v3, PP-DocLayoutV3, PP-OCRv6 detector | Tìm vùng semantic/line |
| Structure | Table Structure v1, spreadsheet parser | Lấy geometry bảng hoặc đọc native workbook |
| OCR/VLM | Nemotron, VietOCR, Tesseract, Qwen, Ministral, PaddleOCR | Chuyển ảnh/crop thành text/Markdown |
| Transform | dedup, clean, caption, chunk, metadata | Chuẩn hóa dữ liệu cho retrieval |
| Embedding | NVIDIA hosted/NIM hoặc local model | Tạo vector cho chunk |
| Storage | LanceDB, sidecar metadata, image evidence | Lưu vector, text, metadata và evidence |
| Access | Python API, CLI, FastAPI, dashboard | Ingest, query, answer và theo dõi job |
| Evaluation | Retriever Harness, BEIR/runfiles, artifacts | Đo recall/latency và kiểm tra regression |

### 4.2. Sơ đồ luồng chung

```text
Input
  → detect file type
  → choose extraction mode
  → convert Office/media when required
  → split document into pages/chunks
  → PDFium native extraction or image rendering
  → Page Elements / layout detection
  → Table Structure or native spreadsheet parser
  → select OCR/VLM pipeline
  → merge text, tables, visual rows and provenance
  → clean/deduplicate
  → chunk
  → embed
  → store in LanceDB
  → query / answer / evaluation
```

### 4.3. Graph execution

Graph runtime dùng các operator có thể chạy trong process hoặc qua Ray:

```text
Graph
  → conversion/splitting
  → extraction operator
  → OCR/layout operator
  → post-extract transform
  → embedding
  → VDB upload
```

GraphPipelineRegistry cung cấp một nơi đăng ký các graph blueprint, build
graph mới, override kwargs, in tree/summary, diff hai graph và serialize graph
thành JSON. Đây là cơ chế hỗ trợ audit/reproducibility, không phải một model
inference riêng.

## 5. Định dạng đầu vào và routing

### 5.1. Định dạng được hỗ trợ

| Nhóm | Định dạng | Đường xử lý |
|---|---|---|
| PDF | .pdf | PDFium native/raster, layout, OCR |
| Office | .docx, .pptx | LibreOffice chuyển về PDF rồi xử lý như PDF |
| Text | .txt, .md, .json, .sh | Text splitter, không cần OCR |
| HTML | .html | MarkItDown/HTML conversion rồi chunk |
| Image | .png, .jpg, .jpeg, .bmp, .tif, .tiff, .svg | Image loader và OCR/layout |
| Spreadsheet | .xlsx, .xls, .csv | Native spreadsheet parser |
| Audio | .mp3, .wav, .m4a | Chunk audio và Parakeet/ASR |
| Video | .mp4, .mov, .mkv, .avi | Tách frame/audio rồi xử lý media |

### 5.2. Routing theo extraction mode

extraction_mode=auto chọn nhánh dựa trên phần mở rộng. Các mode chính là:

```text
pdf, image, text, html, spreadsheet, audio, video, auto
```

Nếu extension không được hỗ trợ và mode là auto, input bị bỏ qua hoặc báo lỗi
tùy API/CLI path. Có thể chỉ định mode rõ ràng khi input không có suffix
chuẩn.

### 5.3. Office và spreadsheet

DOCX/PPTX được convert sang PDF để dùng chung pipeline document. XLSX/XLS/CSV
không rasterize mặc định:

```text
XLS/XLSX/CSV
  → workbook/CSV parser
  → normalized rows/cells
  → table/text representation
  → chunk/embedding
```

File .xls cần LibreOffice để convert sang .xlsx. Embedded image trong
spreadsheet hiện chủ yếu được giữ ở metadata; chưa tự động trở thành một
OCR job độc lập trong đường spreadsheet chính.

## 6. Luồng ingest chi tiết

### 6.1. PDF native

```text
PDF
  → PDFSplit
  → PDFium đọc text và character bbox
  → giữ native text nếu geometry đủ tốt
  → Page Elements cho layout/semantic region
  → Table Structure nếu cần table geometry
  → OCR chỉ cho vùng/trang cần bổ sung
  → merge native + OCR
```

Giữ native text giúp giảm số request, giữ reading order và tránh OCR lại các
trang đã có text tốt.

### 6.2. PDF scan

```text
PDF scan
  → render page image
  → Page Elements/layout detection
  → semantic crop hoặc full-page fallback
  → OCR/VLM theo pipeline được chọn
  → map local bbox về normalized page bbox
  → dedup và quality gate
```

Pipeline mặc định có lớp scan recall bằng full-page và overlapping tiles. Một
số pipeline thử nghiệm dùng line detector hoặc horizontal projection fallback.

### 6.3. Page Elements

Page Elements v3 có thể tạo detection cho các vùng:

```text
text, title, table, chart, image, infographic,
header, footer và các visual region khác
```

Các detection được lưu cùng normalized bbox và metadata model/count. Những
vùng này là đơn vị đầu vào cho Table Structure, OCR crop, visual evidence và
cleaning.

### 6.4. Table Structure và table output

Table Structure v1 lấy cell/row/column geometry trong vùng bảng. Kết quả có
thể được:

- OCR thành text row;
- dựng pseudo-Markdown;
- dựng Markdown table;
- lưu cells có bbox/provenance;
- đưa vào chunk/embedding riêng.

Option có table-cell path sẽ tránh đưa cùng một nội dung table vào cả text row
và cell row nếu hai vùng bị overlap.

### 6.5. Clean, dedup và chunk

```text
merge candidate
  → normalize text
  → remove duplicate/overlap
  → preserve model/source/score/bbox
  → clean content rows
  → split/chunk theo cấu hình
  → tạo embedding
```

Chunking được cấu hình theo loại dữ liệu. Với PDF, geometry-bearing content
rows được chunk trước khi explode thành các row cuối để không làm mất
provenance của block gốc.

### 6.6. Output và provenance

Một output row thường chứa các nhóm thông tin:

| Nhóm | Ví dụ |
|---|---|
| Nội dung | text, Markdown table, caption, transcript |
| Nguồn | source path/id, document/page number |
| Geometry | normalized bbox, local bbox, page coordinates |
| Model | OCR/VLM/layout model và backend được chọn |
| Quality | score, detector score, quality gate, error |
| Pipeline | selector, internal pipeline name, stage metadata |
| Evidence | page image, crop/image reference nếu được bật |
| Retrieval | embedding, distance/score, VDB metadata |

## 7. Hệ thống pipeline và selector

### 7.1. Bảng quy đổi selector

| Selector/API | Pipeline nội bộ hoặc backend | Vai trò |
|---|---|---|
| pipeline-nemotron-ocr | Nemotron OCR v2 baseline | Route mặc định |
| pipeline-ppocrv6 | Option 2 language-routed OCR | Nemotron semantic batch + VietOCR |
| pipeline-tesseract | Compatibility selector | Có thể map vào Option 2/legacy adapter tùy service config |
| pipeline-option3 | Option 3 | Nemotron + raw-text language routing + VietOCR |
| pipeline-option4 | Option 4 | Tesseract-first fusion + Nemotron fallback |
| pipeline-option5 | Option 5 | Line detector + language routing + VietOCR |
| pipeline-option6 | Option 6 | Qwen3.5 VLM/vLLM |
| pipeline-option7 | Option 7 | Ministral VLM/vLLM |

Tên pipeline-ppocrv6 và pipeline-tesseract được giữ một phần vì
compatibility với dashboard/API cũ. Không nên suy luận chỉ từ tên selector
rằng mọi request sẽ gọi đúng một model cùng tên. Hãy kiểm tra
/v1/ingest/pipeline-config, dashboard pipeline trace và endpoint wiring đang
được bật.

### 7.2. Các nhóm pipeline theo hai báo cáo cũ

Các báo cáo trước đây gom selector thành bốn nhóm:

1. **Qwen:** pipeline VLM cho scan/native.
2. **Mistral:** semantic crop và full-page fallback.
3. **NVIDIA:** Page Elements → Table Structure → Nemotron OCR → embedding.
4. **Experimental:** routing tiếng Việt, VietOCR, Tesseract và PP-OCRv6.

Cách gom này phù hợp để trình bày cấp cao. Ở cấp source, các nhánh
experimental được tách thành Option 2/3/4/5/6/7 để tuning và test độc lập.

## 8. Mô tả từng pipeline OCR

### 8.1. Pipeline mặc định — NVIDIA/Nemotron

```text
PDFium native/raster
  → Nemotron Page Elements v3
  → Nemotron Table Structure v1 nếu bật table
  → Nemotron OCR v2
  → full-page/tile scan fallback
  → merge/dedup/clean
  → chunk/embedding
```

Các service core:

| Service | Model/role |
|---|---|
| nim-page-elements | nvidia/nemotron-page-elements-v3 |
| nim-table-structure | nvidia/nemotron-table-structure-v1 |
| nim-ocr | nvidia/nemotron-ocr-v2, multilingual |
| nim-embedding | nvidia/llama-nemotron-embed-vl-1b-v2 |

Đây là route có độ khớp cao nhất với kiến trúc NVIDIA hiện tại, hỗ trợ text,
table, chart, image và infographic. Có thể dùng remote hosted endpoints hoặc
NIM self-host.

### 8.2. Option 2 — Nemotron language-routed Vietnamese OCR

Internal pipeline name hiện tại là:

```text
option2_nemotron_language_routed_vietnamese_ocr
```

Flow:

```text
Page Elements + Table Structure
  → semantic OCR units
  → một Nemotron batch lấy text/local bbox và language observation
  → document/page language decision
  → tiếng Việt → VietOCR
  → English/ambiguous → giữ Nemotron
  → quality gate, merge candidate và provenance
```

Option 2 có một số đặc điểm:

- dùng pipeline-ppocrv6 để giữ API compatibility;
- sample language theo document/page trước khi route;
- bỏ qua native text ở những trường hợp cần OCR theo cấu hình;
- có table-cell units;
- dùng horizontal projection khi response không có local line bbox;
- giữ fallback về Nemotron nếu VietOCR lỗi/điểm thấp;
- không còn phụ thuộc bắt buộc vào remote line detector cho mọi crop.

Đây là nhánh quan trọng cho tiếng Việt nhưng tên selector không phản ánh đầy
đủ internal implementation. PP-OCRv6 official là một deployment alternative
riêng trong Compose, không nên đánh đồng với internal Option 2 này.

### 8.3. Option 3 — Nemotron baseline + VietOCR

```text
Page Elements + Table Structure
  → semantic units
  → Nemotron OCR authoritative pass
  → raw-text language routing
  → Vietnamese quality gate
  → VietOCR thay thế candidate nếu đạt gate
  → Nemotron fallback nếu không đạt
```

Option 3 không có detector riêng. Nó phù hợp khi muốn giữ semantic units và
geometry của baseline, chỉ bổ sung Vietnamese recognizer cho candidate được
nhận diện là tiếng Việt.

### 8.4. Option 4 — Tesseract-first fusion

```text
Page Elements + Table Structure
  → PP-OCRv6 line detector
  → line crop
  → Tesseract vie+eng language probe
  → Vietnamese: Tesseract vie
  → English/mixed/uncertain: Nemotron OCR v2
  → quality check và fusion
```

Cấu hình mặc định trong source:

| Tham số | Giá trị/ý nghĩa |
|---|---|
| language_probe_language | vie+eng |
| tesseract_language | vie |
| language_probe_min_score | khoảng 0.70 |
| tesseract_min_score | khoảng 0.80 |
| tesseract_psm | 7 |
| fallback | Nemotron khi Tesseract rỗng/yếu |

Option 4 có thể giữ dấu tiếng Việt tốt hơn trên crop phù hợp nhưng tốn CPU
và thêm network/request. Nó không phải pipeline xử lý visual đầy đủ; visual
region chủ yếu được loại khỏi OCR unit thông thường.

### 8.5. Option 5 — detector + Vietnamese routing

```text
Page Elements + Table Structure
  → document-level language sample
  → PP-OCRv6 detector cho box nhiều dòng/tall text
  → CPU projection fallback nếu detector không khả dụng
  → VietOCR batch route cho tiếng Việt
  → Nemotron selective fallback
  → full-page recall bounded trên scan thưa
```

Option 5 tập trung vào Vietnamese document có nhiều line/crop. Nó dùng các
batch/concurrency riêng để giảm số request nhỏ, nhưng vẫn cần nhiều service
và cache hơn baseline.

### 8.6. Option 6 — Qwen3.5 qua vLLM

```text
PDFium/native text hoặc scan image
  → Page Elements semantic detection
  → crop/streaming queue
  → Qwen3.5-2B VLM
  → page order/Markdown/text output
  → clean/chunk/embedding
```

Các preset model:

| Preset | Model |
|---|---|
| option6-qwen-nvfp4.env | AxionML/Qwen3.5-2B-NVFP4 |
| option6-qwen-fp8.env | surogate/Qwen3.5-2B-FP8 |
| option6-qwen-bf16.env | Qwen/Qwen3.5-2B |

Qwen sidecar dùng qwen35-vllm.compose.yaml, image GPU local
nemo-retriever-service-gpu:dev, model cache Hugging Face và vLLM runtime
cache. Preset NVFP4 hiện giới hạn model length, sequence, batched tokens và
GPU memory utilization để phù hợp môi trường development.

### 8.7. Option 7 — Ministral 3B FP8

```text
Page Elements semantic crop
  → Ministral-3-3B-Instruct-2512
  → OCR/VLM output
  → scan/layout yếu: full-page fallback
  → merge/chunk/embedding
```

Model chạy trong sidecar ministral-fp8, publish host port 8016, dùng image
GPU và Hugging Face cache. Route này phù hợp để so sánh VLM crop với Qwen,
nhưng cần VRAM lớn hơn và không nên chạy đồng thời nhiều VLM nặng trên một GPU
nhỏ.

### 8.8. Official PP-OCRv6

ppocrv6-official là service hoàn chỉnh gồm:

```text
document orientation
  → unwarping
  → text-line orientation
  → PP-OCRv6 medium detection
  → PP-OCRv6 medium recognition
```

Host port mặc định là 8012. Ngoài official service, Compose còn có
ppocrv6-det và ppocrv6-rec cho pipeline tách detector/recognizer.

### 8.9. PaddleOCR-VL

PaddleOCR-VL 1.6 có hai lớp:

```text
PaddleOCR-VL API
  → PP-DocLayoutV3/layout parsing
  → internal vLLM VLM server
  → text/table/visual result assembly
```

Retriever gọi API layer; API layer gọi VLM server nội bộ. Đây là nhánh riêng,
không gọi Page Elements, Table Structure hay Nemotron OCR trong adapter
PaddleOCR-VL.

## 9. Hậu xử lý, embedding, VectorDB và query

### 9.1. Post-extraction stages

Sau extraction, pipeline có thể xếp các stage:

```text
extract → dedup → caption → embed → store → filter → webhook
```

stage_order chỉ điều khiển các stage sau extraction; extraction luôn chạy
trước. Các tham số được truyền qua typed params và có policy validation ở
service để hạn chế client override các trường nhạy cảm.

### 9.2. Embedding

Embedding có thể chạy qua:

- NVIDIA hosted embedding endpoint;
- nim-embedding self-host;
- local Hugging Face/GPU model;
- endpoint tùy chỉnh nếu service policy cho phép.

Phải dùng cùng embedding model hoặc metadata tương thích khi ingest và query.
Đổi embedding model giữa hai bước có thể làm vector space không tương thích.

### 9.3. LanceDB

LanceDB là VectorDB mặc định trong source:

```text
chunk text + metadata + embedding
  → LanceDB table
  → dense / hybrid / sparse retrieval
  → optional rerank
  → answer LLM hoặc agentic retrieval
```

Các mode retrieval hiện có gồm auto, dense, hybrid và sparse. Dense dùng
vector, hybrid kết hợp vector và BM25/FTS, sparse dùng FTS-only theo cấu hình.

### 9.4. Query và answer

Product workflow tách rõ:

- retriever query: trả retrieved hits;
- retriever answer: vector search + configured LLM generation;
- agentic retrieval: ReAct/LLM-driven loop trên index đã tồn tại;
- dashboard VDB query: kiểm tra trực tiếp index/service.

Service có các endpoint chính /v1/query và /v1/answer. Answer LLM có thể là
hosted endpoint, nim-answer, hoặc LiteLLM/local backend phù hợp cấu hình.

## 10. Service API, job system và dashboard

### 10.1. Kiến trúc service

```text
Client/dashboard
  → Retriever FastAPI gateway
  → job aggregate + page/document routing
  → Realtime/Batch work queue
  → worker pipeline
  → VectorDB/sidecar result store
  → status/events/result callback
```

Retriever tự route theo page count/chunk policy. Document/page submission có
route riêng; worker gửi callback và gateway tổng hợp job state.

### 10.2. API chính

| Endpoint/nhóm | Chức năng |
|---|---|
| GET /v1/health | Health của Retriever |
| POST /v1/ingest/job | Tạo job aggregate |
| POST /v1/ingest/job/{job_id}/document | Upload document, auto route |
| POST /v1/ingest/job/{job_id}/page | Gửi một page |
| POST /v1/ingest/job/{job_id}/whole | Gửi whole document |
| GET /v1/ingest/job/{job_id} | Trạng thái job |
| GET /v1/ingest/job/{job_id}/documents | Danh sách document |
| GET /v1/ingest/job/{job_id}/events | SSE events theo job |
| GET /v1/ingest/pipeline-config | Introspect pipeline live |
| POST /v1/query | Vector query qua VectorDB |
| POST /v1/answer | Query + LLM answer |
| /v1/ingest/status/* | Status item/page/document/batch |
| /v1/ingest/sidecar | Upload/retain visual sidecar |
| /v1/internal/* | Worker callback/result, không phải public API |

Các route legacy /v1/ingest cũ đã được đánh dấu removed/compatibility stub;
client mới nên dùng job/document/page API.

### 10.3. Dashboard

Dashboard không có frontend container riêng; static UI được serve bởi
Retriever. Dashboard có thể:

- upload file và tạo job;
- xem danh sách/detail job;
- xem status document/page;
- xem pipeline trace;
- xem visual evidence/page image/block crop;
- chạy VDB query;
- hiển thị model/backend/endpoint trace;
- chọn OCR selector được service cho phép.

Visual evidence và raw result retention là các cờ riêng; không nên bật payload
lớn trong production nếu không cần.

## 11. Harness và benchmark framework

### 11.1. Ranh giới

Product workflow dùng retriever ingest và retriever query. Harness dùng
benchmark registry/runfile, evaluation và artifact; nó không phải scheduler,
secret distributor hay public product API.

### 11.2. Các command chính

```text
retriever harness list
retriever harness show
retriever harness run
retriever harness run-set
retriever harness run-files
retriever harness run-helm
retriever harness check-vidore-access
retriever harness post-slack
retriever harness diff
```

run-files là entrypoint portable chính. Runfile chọn benchmark đã đăng ký,
không tự định nghĩa benchmark mới. Dataset path và credential nên nằm ngoài
source control.

Ví dụ:

```bash
uv run --project nemo_retriever retriever harness list --runsets
uv run --project nemo_retriever retriever harness show jp20_beir --json
```

### 11.3. Artifact contract

| Artifact | Ý nghĩa |
|---|---|
| status.json | run đang chạy và phase hiện tại |
| results.json | một run đã terminal |
| session_summary.json | summary của run-files/run-set session |
| run.log | log chi tiết |
| resolved_benchmark.json | benchmark sau resolve |
| ingest_plan.json | kế hoạch ingest |
| query_plan.json | kế hoạch query |
| environment.json | revision, GPU và runtime context |
| beir_metrics.json | metric BEIR |
| query_results.jsonl | kết quả query chi tiết |

Kết quả nên được đánh giá bằng exit code và JSON artifact, không parse progress
bar/stdout.

### 11.4. Legacy Portal Runner

Portal vẫn có một số UI/API tên runner để tương thích lịch sử, nhưng lệnh
harness runner start cũ không phải entrypoint hiện tại. Current harness dùng
run-files, run-set hoặc run-helm; cần đọc nemo_retriever/harness/README.md và
docs service/library trước khi chạy benchmark.

## 12. Deployment, NIM, model và cache

### 12.1. Các mode deployment

| Mode | Thành phần | Khi dùng |
|---|---|---|
| Hosted | Retriever + VectorDB, gọi NVIDIA endpoint | Setup nhẹ, không tự host model |
| Self-host Compose | Retriever + VectorDB + NIM/sidecar | Development/single host |
| Local GPU | service-gpu + Hugging Face models | Không dùng NIM core, cần GPU local |
| Kubernetes/Helm | Retriever service + external/in-cluster NIM | Production, scale và lifecycle dài hạn |

Docker Compose development không thay thế Helm deployment production.

### 12.2. Yêu cầu máy

- Linux, khuyến nghị Ubuntu 22.04 trở lên.
- Docker Engine, Docker Compose tối thiểu 2.23.1 và Buildx.
- Python 3.12 nếu chạy bằng uv/CLI.
- NVIDIA driver/Container Toolkit phù hợp nếu dùng GPU/NIM.
- GPU/VRAM tùy model; không bật đồng thời tất cả VLM/NIM trên GPU nhỏ.
- Dung lượng trống lớn cho NIM engine/cache, Hugging Face và compile cache.

### 12.3. Credential

| Key | Dùng cho |
|---|---|
| NVIDIA_API_KEY | NVIDIA hosted inference/embedding |
| NGC_API_KEY | login nvcr.io, pull NIM và model artifact |
| Hugging Face token | gated/private model hoặc model cần xác thực |

Không ghi credential vào Compose, .env public, Git history hoặc report.

### 12.4. Core NIM profile

Preset nims-core.env nối các endpoint nội bộ:

```text
Page Elements    → http://nim-page-elements:8000/v1/page-elements
Table Structure  → http://nim-table-structure:8000/v1/table-structure
OCR              → http://nim-ocr:8000/v1/ocr
Embedding        → http://nim-embedding:8000/v1/embeddings
```

nim-page-elements, nim-ocr và nim-embedding dùng profile nims-core;
nim-table-structure dùng profile table-structure, nên core command phải
bật cả hai profile.

### 12.5. Optional profiles

| Profile | Service | Mục đích |
|---|---|---|
| ppocrv6 | ppocrv6-official, ppocrv6-det, ppocrv6-rec | PaddleOCR/PP-OCRv6 |
| vintern | vintern-ocr | Vintern-1B-v3.5 vLLM |
| vietocr | vietocr-ocr | Vietnamese recognizer |
| qwen35-nvfp4 | qwen35-nvfp4 | Qwen3.5 vLLM |
| ministral-fp8 | ministral-fp8 | Ministral FP8 VLM |
| tesseract | tesseract | CPU OCR sidecar |
| paddleocr-vl | API + VLM server | PaddleOCR-VL 1.6 |
| nim-reranker | reranker NIM | rerank |
| nim-parse | Nemotron Parse NIM | alternate parser |
| nim-caption | caption NIM | image/video caption |
| nim-answer | answer NIM | answer generation |
| nim-audio | Parakeet NIM | audio/video ASR |
| observability | OTEL + Zipkin | trace/metrics |

### 12.6. Model/cache ownership

```text
cache/nim/             NIM engine/cache
cache/huggingface/     Hugging Face model/cache
cache/vintern/         Vintern model/runtime
cache/vietocr/         VietOCR model/cache
cache/ppocrv6/         PaddleOCR model/cache
cache/qwen35-vllm/     vLLM/FlashInfer/TorchInductor
cache/ministral-fp8/   Ministral runtime
cache/retriever/       Retriever state/log
cache/vectordb/        LanceDB state
```

Cache bind mount giúp container recreate không phải tải/compile lại toàn bộ,
nhưng cache không phải source và không được commit.

## 13. Port map

Port bên trái là host port mặc định; port trong ngoặc là port trong container.
URL dạng http://service:8000 chỉ dùng giữa các container trong Docker network.

| Host port | Service | Ý nghĩa |
|---:|---|---|
| 7670 | retriever | FastAPI mặc định |
| 7671 | vectordb | LanceDB/vector service |
| 7780 | retriever với core preset | FastAPI self-host stack |
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
| 8011 | tesseract:8000 | CPU Tesseract |
| 8012 | ppocrv6-official:8000 | Official PaddleOCR pipeline |
| 8013 | vintern-ocr:8000 | Vintern vLLM |
| 8014 | vietocr-ocr:8000 | VietOCR |
| 8015 | qwen35-nvfp4:8000 | Qwen vLLM |
| 8016 | ministral-fp8:8000 | Ministral vLLM |
| 8118 | paddleocr-vl-api:8080 | PaddleOCR-VL API |
| 9000 | nim-audio:9000 | Parakeet HTTP |
| 50051 | nim-audio:50051 | Parakeet gRPC |
| 4317 | OTEL collector | OTLP gRPC |
| 4318 | OTEL collector | OTLP HTTP |
| 8889 | OTEL collector | Prometheus metrics |
| 9411 | Zipkin | Trace UI/API |

Port host có thể đổi bằng biến *_HOST_PORT hoặc RETRIEVER_HTTP_PORT.

## 14. Cấu trúc thư mục source

| Thư mục/file | Vai trò |
|---|---|
| nemo_retriever/src/nemo_retriever | package Python chính |
| common/params, common/schemas | typed params và wire contract |
| common/modality | PDF, image, HTML, spreadsheet, audio/video, OCR |
| common/modality/ocr/isolated | Option 2–7 và adapter OCR/VLM |
| operators | graph operators và stage implementation |
| graph | Graph, executor, ingestor runtime, retriever |
| service | FastAPI app, worker/gateway, VectorDB proxy |
| harness | benchmark registry, runfile, artifact, Helm runner |
| nemo_retriever/dev/compose | Docker Compose service profiles |
| nemo_retriever/helm | Kubernetes/Helm deployment |
| nemo_retriever/tests | unit/contract/integration tests |
| tools | sidecar service và development helper |
| data | test/annotation dataset metadata |
| setup.md | setup, model, credential và port guide |
| start.md | command start/stop/run theo từng profile |
| baocao.md | báo cáo tổng hợp này |

## 15. Kết quả benchmark và bằng chứng hiện có

### 15.1. Qwen

Các số dưới đây được ghi trong fullbaocao.md, là kết quả development theo
hardware/preset tại thời điểm đo, chưa phải benchmark production độc lập:

| Workload | Kết quả |
|---|---|
| Scan 40 trang | 32,69 giây; 40/40 đúng thứ tự; 41 requests |
| Request đầu | khoảng 2,38 giây |
| Throughput ghi nhận | khoảng 1.097 token/s aggregate |
| VRAM | khoảng 15,1 GB |
| Native 4 trang | khoảng 0,59 giây |
| Report native 6 trang | khoảng 1,87 giây |
| Similarity | 0,9991 trên phép đo được ghi |
| Exact | 2/4 trên corpus được ghi |

### 15.2. Mistral

| Workload | Kết quả |
|---|---|
| Scan 40 trang | 53,89 giây; 41 rows |
| VRAM | báo cáo ghi khoảng 16,5 GB, gồm model/KV cache |
| Native 4 trang | 1,05 giây; 6 rows; 8 requests |
| Report scan 6 trang | 11,13 giây; 7 rows |
| Quality | chưa persist đầy đủ similarity/ground truth |

### 15.3. Option 1 và Option 4

Hiện chưa có benchmark chính thức với:

- cùng một corpus;
- ground truth thống nhất;
- cùng GPU/model/version;
- cùng concurrency/batch/preset;
- cùng tiêu chí accuracy/latency.

Nhận định đang có chỉ là quan sát development:

| Pipeline | Quan sát |
|---|---|
| NVIDIA/Nemotron | khớp hệ sinh thái NVIDIA, ổn định hơn với tiếng Anh và visual pipeline |
| Option 4 | có thể giữ dấu tiếng Việt tốt hơn trên crop phù hợp, nhưng thêm probe/detector/CPU và latency |
| Option 2/3/5 | có routing tiếng Việt và fallback, nhưng phụ thuộc VietOCR quality/sidecar |

Không nên dùng các số trên để cam kết SLA hoặc accuracy cho dữ liệu production.

### 15.4. Test/validation

Các lớp kiểm tra chính trong repository:

- unit test cho OCR geometry, language routing, candidate merge;
- contract test cho Compose/profile;
- test spreadsheet ingest;
- service/API/pipeline-config test;
- Helm manifest/tracing test;
- harness/evaluation test.

Kiểm tra Compose không khởi động container:

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  --profile table-structure \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  config --quiet
```

## 16. Giới hạn, rủi ro và các điểm cần chú ý

### 16.1. Accuracy

- Scan nhiễu, mờ, lệch, chữ nhỏ và nền phức tạp vẫn có thể gây mất ký tự,
  sai dấu câu hoặc sai dấu tiếng Việt.
- Page Elements có thể bỏ sót text nằm ngoài semantic box.
- OCR/VLM output cần quality gate và fallback; fallback không bảo đảm đúng.
- Tesseract/VietOCR có thể cần tuning theo font, crop width, language và GPU.

### 16.2. Dấu mộc và visual

Source có params/module cho stamp detection, nhưng đường graph/service hiện
không bật stamp stage mặc định một cách đáng tin cậy. Trong runtime OCR path,
extract_stamps có thể bị tắt để tránh xử lý ngoài pipeline chính.

Do đó dấu mộc có thể bị xem như text/image/visual region thông thường. Tiêu
ngữ cũng chưa có detector riêng; nó phụ thuộc layout và OCR route được chọn.

Option 4 tập trung vào text/table line, không phải full visual understanding.
Option 1 và PaddleOCR-VL có đường visual/layout rõ hơn.

### 16.3. Spreadsheet

Spreadsheet được parse native, không chạy OCR mặc định. Embedded image trong
workbook hiện chưa tự động đi qua OCR pipeline độc lập; nếu cần phải thiết kế
flow tách image riêng.

### 16.4. Runtime và tài nguyên

- Core NIM và VLM có thể chiếm nhiều VRAM/disk.
- Không bật Qwen, Vintern, Ministral, VietOCR, PP-OCRv6 và toàn bộ NIM cùng
  lúc trên một GPU nhỏ.
- NIM lần đầu có thể mất nhiều thời gian tải/compile.
- Docker down không xóa bind-mounted cache; cache cũ có thể làm kết quả
  startup/benchmark không giống máy sạch.

### 16.5. Selector compatibility

Tên pipeline-ppocrv6/pipeline-tesseract có lịch sử compatibility và có thể
được dashboard dùng cho nhiều adapter. Cần xem live pipeline config/trace thay
vì chỉ dựa vào label UI.

### 16.6. Service security

- API key phải được giữ ngoài source.
- Nếu expose dashboard/API ra mạng, cần AuthN/AuthZ và reverse proxy.
- Log development có thể chứa input prompt, extracted text, output completion
  hoặc metadata nhạy cảm.
- Endpoint hosted chuyển dữ liệu document ra bên ngoài; phải kiểm tra policy
  dữ liệu trước khi dùng.
- NIM/container/model license của bên thứ ba không tự động giống license của
  source.

### 16.7. Harness

Harness artifact là nguồn kết quả chính. Không parse stdout/progress bar để
đánh giá benchmark. Dataset path, Slack webhook và secret phải nằm ngoài
runfile/source.

## 17. Quy trình setup/run khuyến nghị

### 17.1. Hosted nhanh nhất

1. Cài Docker Compose/Python theo setup.md.
2. Clone repository và đặt NVIDIA_API_KEY.
3. Chạy start.md mục Hosted API.
4. Kiểm tra /v1/health.
5. Upload từ dashboard hoặc dùng CLI ingest.

Ưu điểm: ít image/model local. Nhược điểm: document đi tới hosted endpoint.

### 17.2. Self-host core

1. Cài NVIDIA Container Toolkit và kiểm tra nvidia-smi.
2. Đăng nhập NGC bằng NGC_API_KEY.
3. Bật profile nims-core và table-structure.
4. Chờ bốn core service healthy.
5. Chạy ingest/query test.
6. Chỉ bật optional profile khi thật sự cần.

### 17.3. Vietnamese OCR

| Mục tiêu | Lựa chọn |
|---|---|
| Giữ baseline Nemotron, thêm Vietnamese recognizer | Option 2 hoặc Option 3 |
| Nhiều line/tall Vietnamese text | Option 5 + PP-OCRv6 detector |
| Tesseract-first/fusion | Option 4 |
| VLM scan/native | Qwen Option 6 hoặc Ministral Option 7 |

### 17.4. Production

Docker Compose phù hợp development/single-host. Production nên dùng Helm:

```text
GPU node + NVIDIA GPU Operator
  → NIM Operator/external NIM
  → persistent model/cache storage
  → Retriever Helm chart
  → ingress/AuthN/AuthZ/observability
```

Không dùng setup development làm security boundary production.

## 18. Kết luận và lựa chọn pipeline

| Nhu cầu | Pipeline nên bắt đầu |
|---|---|
| Setup nhanh, ít local model | Hosted baseline |
| Tự chủ dữ liệu/model qua NGC | NVIDIA/Nemotron core |
| Tiếng Việt, muốn giữ geometry/baseline | Option 2 hoặc Option 3 |
| Vietnamese line crop và batching | Option 5 |
| Tesseract-first, fallback Nemotron | Option 4 |
| VLM throughput/scan benchmark | Qwen Option 6 |
| VLM semantic crop khác để so sánh | Ministral Option 7 |
| Full PaddleOCR document pipeline | PP-OCRv6 official/PaddleOCR-VL |

Kết luận tổng thể:

- NVIDIA/Nemotron là baseline vận hành và tích hợp ổn định nhất của project.
- Option 2/3/5 dành cho bài toán tiếng Việt, nhưng cần VietOCR model/cache,
  endpoint và quality gate phù hợp.
- Option 4 cung cấp chiến lược fusion rõ ràng nhưng tăng latency và phụ thuộc
  CPU/sidecar.
- Qwen và Ministral là các VLM route phục vụ thử nghiệm/benchmark, không nên
  mặc định bật cùng toàn bộ NIM.
- Project đã có đủ lớp từ ingest đến retrieval/evaluation, nhưng benchmark
  accuracy giữa các OCR route vẫn cần một corpus/ground truth chuẩn trước khi
  đưa ra kết luận định lượng.

## 19. Tài liệu tham khảo trong repository

- README.md: giới thiệu library và Python quick start.
- setup.md: prerequisite, credential, model, Compose profile và port.
- start.md: command start/stop/run cụ thể theo từng container/profile.
- nemo_retriever/dev/compose/README.md: Compose development deployment.
- nemo_retriever/pyproject.toml: dependency, extras và Python version.
- nemo_retriever/harness/README.md: benchmark command/artifact contract.
- nemo_retriever/helm/README.md: Kubernetes/Helm deployment.
- nemo_retriever/developer_docs/graph_pipeline_registry.md: graph registry.
- thongtinbaocao.md: báo cáo ingest/OCR chi tiết trước đây.
- fullbaocao.md: báo cáo benchmark/tổng hợp trước đây.

Báo cáo này mô tả trạng thái source tại ngày tổng hợp. Model version, image
tag, dependency, GPU behavior và benchmark có thể thay đổi theo preset hoặc
commit mới; khi triển khai thực tế cần đối chiếu lại Compose config, health
endpoint và artifact của lần chạy đó.
