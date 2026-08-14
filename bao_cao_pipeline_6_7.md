# BÁO CÁO TRIỂN KHAI NEMO-RETRIEVER CHO XỬ LÝ DỮ LIỆU

## Phạm vi thử nghiệm

Báo cáo ghi lại kiến trúc, cấu hình, số đo hiện có và tình hình khi thử nghiệm hai model Qwen và Mistral trong hai pipeline OCR:

- **Pipeline Qwen:** Page Elements detect → native PDFium hoặc scan full-page → Qwen3.5-2B NVFP4.
- **Pipeline Mistral:** Page Elements semantic detection/crop → Ministral 3 3B FP8; scan/layout yếu có full-page fallback.

Phần embedding không nằm trong phép đo parse/OCR. Bảng kết quả giữ cùng các hạng mục đã dùng khi đo Qwen; các ô không được log Pipeline 7 lưu lại được ghi rõ là chưa có số, không nội suy.

## 1. Kiến trúc tổng quan

Trong ingest, hai pipeline dùng chung một luồng trước khi tách nhánh:

```text
PDF / image
  → DocToPdf
  → PDFSplit + PDFium render/extract
  → Page Elements detection
  → chọn Pipeline Qwen hoặc Pipeline Mistral
  → OCR output + metadata/provenance
  → clean_content_rows / chunk
  → embedding (ngoài phép đo parse)
```

Page Elements được dùng ở cả hai pipeline để lấy thông tin về vùng trên trang. Khác biệt nằm ở cách đưa kết quả đó vào VLM:

| Điểm đang dùng | Pipeline Qwen | Pipeline Mistral |
|---|---|---|
| Ảnh đưa vào VLM | scan đọc nguyên trang, native chỉ gửi box thiếu text hoặc crop visual hợp lệ | semantic crop text/title/table/visual; scan/layout yếu thêm full-page fallback |
| VLM | Qwen3.5-2B NVFP4 | Ministral 3 3B FP8 |
| Table | crop riêng và dựng Markdown | crop theo bbox Page Elements, Ministral dựng Markdown |
| Visual | crop độc lập được giữ lại, bbox gần toàn trang chứa nhiều text bị loại | crop visual độc lập gửi Ministral |
| BBox | theo block/crop hợp lệ, có thể có fallback full-page khi native yếu | semantic bbox; full-page bbox khi fallback |
| Cách xử lý | PDFium giữ native text, VLM bổ sung phần thiếu, giữ cấu trúc chi tiết | PDFium giữ native text; semantic crop gửi VLM, scan/layout yếu dùng full-page |

## 2. Pipeline Qwen — Page Detect → Qwen 3.5 VLM

### 2.1. Luồng xử lý

```text
PDFium native text + page raster
  → Page Elements v3: text / title / table / visual
  → producer: render + detect + crop theo block 16 trang
  → bounded queue
  → consumer: Qwen VLM CCR 8, continuous batching
  → ghép theo page_number / reading_order
  → text block + Markdown table + visual label/crop
```

Với trang scan, Pipeline Qwen gửi ảnh nguyên trang cho OCR, đồng thời vẫn giữ geometry table và visual từ Page Elements. Với trang native, PDFium native text là nguồn chính, chỉ các vùng thiếu character geometry mới đi tiếp qua VLM. Khi native page có layout yếu, Qwen có thể đọc bổ sung một ảnh toàn trang để tránh dồn quá nhiều bbox. Native table hiện vẫn dùng image crop vì cờ `OPTION6_NATIVE_TABLE_TEXT_ENABLED=false` đang được giữ trong quality gate.

Visual region được kiểm tra riêng. Crop nhỏ và độc lập được gửi prompt phân loại ngắn, còn bbox gần toàn trang có nhiều text bị loại trước khi tạo request visual. Prompt cho phép trả `BỎ QUA`, vì vậy một sơ đồ nằm bên trong nguyên trang không bị gắn nhãn là cả trang đó là sơ đồ. Crop visual hợp lệ vẫn được giữ làm image evidence cho frontend.

### 2.2. Producer–consumer và cấu hình hiện tại

Trong lúc xử lý một file, phần chuẩn bị trang và phần gọi VLM chạy theo producer–consumer. Trang nào render/detect/crop xong được đưa vào queue, không chờ chuẩn bị hết toàn bộ file.

- PDF extract: batch 16, 4 worker.
- Page Elements/detect: batch 128.
- Crop: batch 128, tối đa 4 tác vụ crop đồng thời.
- Handoff: streaming bật, block 16 trang, queue tối đa 2 block.
- Qwen client: batch 8, tối đa 8 request đồng thời, tương ứng CCR 8 ở preset NVFP4 hiện tại.
- vLLM: NVFP4, max model length 32K, max sequence 8, max batched tokens 4096, GPU utilization 0.20.
- Thinking đã tắt, temperature request bằng 0.
- Qwen có nhiều profile prompt để thử nghiệm: `legacy`, `strict`, `char_repair` và `word_repair`. Profile đang dùng là `word_repair`, chép nguyên văn, chỉ khôi phục ký tự/dấu/từ bị nhòe khi có đúng một cách đọc, không tự ý đổi câu, thêm hoặc bớt nội dung. Thinking đã tắt, prompt visual chỉ trả nhãn ngắn hoặc `BỎ QUA`.

### 2.3. Số đo đã có

| Bài đo | Kết quả ghi nhận |
|---|---:|
| Scan 40 trang | 32,69 giây |
| Số trang / thứ tự | 40/40, đúng thứ tự |
| VLM requests | 41 = 40 text + 1 table |
| VLM generation | khoảng 1.097 token/s aggregate |
| Request đầu tiên | sau 2,38 giây |
| Qwen VRAM | khoảng 15,1 GB gồm Qwen + Page Elements |
| Native thật 4 trang | 0,59 giây, 4/4 native pages |
| Native report 6 trang sau visual gate | 1,87 giây, 3 bbox gần toàn trang bị loại, không lỗi |

So sánh prompt trên corpus ground-truth 4 trang cho thấy profile `word_repair` có similarity trung bình 0,9991, các profile `legacy`, `strict` và `char_repair` lần lượt là 0,9531, 0,9158 và 0,9905. Đây là số đo của corpus thử hiện tại, chưa phải accuracy chính thức cho mọi loại tài liệu. Ở một trường hợp khó vẫn còn lỗi lặp từ, nên giới hạn sửa được giữ ở mức chặt.

### 2.4. Output và các vấn đề còn sót

Với scan, Pipeline Qwen đọc nguyên trang; với native, PDFium đọc text trước và chỉ tạo box khi cần. Số box native được hạn chế để giảm request không cần thiết, file native 4 trang đã thử có thời gian 0,59 giây. Pipeline Qwen giữ text block, table region, Markdown table, visual label/crop hợp lệ và provenance theo block/crop. Latency còn phụ thuộc vào số vùng Page Elements phát hiện được, không chỉ phụ thuộc vào số trang.

Các vấn đề còn sót hiện tại:

- Ảnh nhòe vẫn có thể làm model lặp một từ dù prompt đã giới hạn việc sửa.
- Native table của Pipeline Qwen hiện vẫn dùng image crop, đường native text → Markdown chưa bật trong quality gate.
- Page Elements đôi khi trả một bbox gần nguyên trang cho infographic. Gate hình học và prompt `BỎ QUA` đã loại trường hợp này, nhưng các crop visual nhỏ vẫn phụ thuộc vào chất lượng detector.
- Số file đã thử còn ít, chưa tạo thành một tập benchmark chính thức.

## 3. Pipeline Mistral — Semantic crop → Ministral VLM

### 3.1. Luồng xử lý

```text
PDFium render/extract từng trang
  → Page Elements detect text/title/table/visual
  → tạo semantic crop theo bbox và gửi Ministral
  → scan/layout yếu: thêm một full-page raster / trang
  → Ministral 3 3B FP8
  → text block/table Markdown/visual output theo reading order
```

Page Elements cung cấp vùng cần đọc. Native text được PDFium giữ làm nguồn chính; semantic crop OCR được gửi cho Ministral. Trang scan hoặc layout yếu dùng full-page fallback.

### 3.2. Các stage không dùng và cấu hình hiện tại

Không bật Table Structure, line detector hoặc language probe. Semantic OCR crop được bật; Table Structure không chạy, table bbox lấy từ Page Elements và Ministral đọc whole-table crop thành Markdown.

- Model: `mistralai/Ministral-3-3B-Instruct-2512`.
- vLLM: max model length 8192, max sequence 10, max batched tokens 4096, GPU utilization 0.33.
- Client: VLM batch 10, tối đa 10 request workers, max output 1024 token.
- Mỗi trang có thể có semantic OCR unit theo bbox; scan/layout yếu có thêm OCR unit full-page với bbox `[0, 0, 1, 1]`.
- Page Elements cung cấp semantic text/title/table/visual bbox để lập crop và giữ provenance trong trace.
- Table gửi whole-table crop cho Ministral và trả Markdown; visual gửi crop riêng khi cần đọc chữ/caption.
- Reading order semantic block được giữ trước khi ghép output theo page order.

### 3.3. Số đo Pipeline 7 đã lưu trong log

Các số dưới đây được lấy từ run Pipeline 7 đã có, không chạy lại Qwen:

| Bài đo | Kết quả ghi nhận |
|---|---:|
| Scan 40 trang (`NỘI QUY LAO ĐỘNG.pdf`) | 53,89 giây, 41 rows |
| Số trang / thứ tự | File có 40 trang; log chỉ lưu 41 rows, không còn trace để xác nhận lại thứ tự từng trang |
| VLM requests | Không có bộ đếm theo từng job trong log còn giữ |
| VLM generation | Trong cửa sổ run 40 trang: khoảng 795–961 token/s aggregate trên vLLM |
| Request đầu tiên | Không được log theo timestamp, nên không suy ra được |
| Ministral VRAM | Model load 4,44 GiB; vLLM báo 10,17 GiB KV cache khả dụng; Page Elements lúc khởi động 470 MiB |
| Native 4 trang | 1,05 giây, 4 trang, 6 rows; trace ghi 8 VLM requests (1 text + 5 table + 2 visual) |
| Report scan 6 trang (`thongtinbaocao.pdf`) | 11,13 giây, 7 rows; 6 full-page + 1 table request, không lỗi document |
| Chất lượng trên corpus 4 trang | Có run regression 4 rows/4,66 giây nhưng không có similarity hoặc ground truth được lưu lại |

VRAM của Ministral là số khởi động của container, không phải peak VRAM tổng của một job. Vì các log cũ không lưu trace/timestamp per-request cho run 40 trang, không thể gán chính xác số request hay latency request đầu cho riêng job đó.

## 4. Kết quả thử nghiệm

Cùng khuôn đo với Qwen. Mistral dùng đúng các mốc tương ứng đã có trong log Pipeline 7; dấu `chưa lưu` nghĩa là hệ thống lúc đó không persist trường đo, không phải kết quả bằng 0.

| Hạng mục | Số đo / tình hình | Ghi chú |
|---|---|---|
| Scan 40 trang · Pipeline Qwen | 32,69 giây, 40/40 đúng thứ tự, request đầu sau 2,38 giây | 41 requests, khoảng 1.097 token/s aggregate |
| Scan 40 trang · Pipeline Mistral | 53,89 giây, file 40 trang, 41 rows | Thứ tự từng trang chưa trace lại được; request đầu chưa lưu |
| VRAM · Pipeline Qwen | khoảng 15,1 GB gồm Qwen và Page Elements | Qwen khoảng 10 GB, detect khoảng 5 GB |
| VRAM · Pipeline Mistral | 4,44 GiB model load + 10,17 GiB KV cache khả dụng | Page Elements startup 470 MiB; không có peak per-job |
| Native 4 trang · Pipeline Qwen | 0,59 giây, 4/4 native pages | 1 table request, `errors=[]` |
| Native 4 trang · Pipeline Mistral | 1,05 giây, 4 trang, 6 rows | 8 VLM requests: 1 text + 5 table + 2 visual; document hoàn tất |
| Report 6 trang · Pipeline Qwen | 1,87 giây, 3 bbox gần toàn trang bị loại | Không lỗi, sidecar không còn visual giả |
| Report scan 6 trang · Pipeline Mistral | 11,13 giây, 7 rows | 7 requests: 6 full-page + 1 table; document error = null |
| Chất lượng · Pipeline Qwen | `word_repair` similarity 0,9991, exact 2/4 trên corpus 4 trang | Vẫn lặp một từ ở trường hợp khó |
| Chất lượng · Pipeline Mistral | Regression corpus: 4 rows trong 4,66 giây | Similarity/ground truth chưa được persist, không có điểm để so trực tiếp |

### Các mốc đã có

- Pipeline Qwen đã chạy một file scan 40 trang và một file native 4 trang.
- Pipeline Mistral đã có run scan 40 trang, native 4 trang và report scan 6 trang trong log Pipeline 7.
- Đã so sánh các profile prompt trên corpus ground-truth 4 trang của Qwen; `word_repair` là profile đang dùng.

### Các vấn đề còn sót

- Số file thử còn ít, chưa đủ để kết luận latency hoặc độ chính xác trên một tập benchmark chính thức.
- Similarity 0,9991 là điểm tương đồng của corpus thử, không phải con số accuracy áp dụng cho mọi tài liệu.
- Log Pipeline 7 chưa persist số request per-job, request đầu và similarity; ba chỉ số này cần bổ sung telemetry nếu muốn so trực tiếp với Qwen ở lần sau.
- Native table của Pipeline Qwen hiện vẫn đi theo image crop.

Latency, chất lượng nội dung và VRAM trong báo cáo chỉ phản ánh các file đã thử. Embedding nằm ngoài phép đo parse/OCR.
