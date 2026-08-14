# BÁO CÁO CHUNG NEMO-RETRIEVER

## Kiến trúc ingest và bốn pipeline OCR

**Ngày tổng hợp:** 13.08.2026

## 1. Mục lục

| Trang | Nội dung |
|---:|---|
| 1 | Bìa |
| 2 | Mục lục |
| 3 | Kiến trúc chung và điểm tách bốn route |
| 4 | Pipeline Qwen |
| 5 | Pipeline Mistral |
| 6 | Kiến trúc NVIDIA |
| 7 | Kiến trúc thử nghiệm và routing ngôn ngữ |
| 8 | Kết quả benchmark và các case đặc thù |

## Tổng quan bốn pipeline

Báo cáo này ghép phần kiến trúc ingest tổng quát với phần benchmark các pipeline OCR. Hệ thống có một luồng nhận file chung, sau đó chọn route theo loại dữ liệu và selector của job.

Bốn pipeline trong báo cáo là:

1. **Pipeline Qwen:** Page Elements detect → PDFium native hoặc scan full-page → Qwen3.5-2B NVFP4.
2. **Pipeline Mistral:** Page Elements semantic crop → Ministral 3 3B FP8; scan/layout yếu có full-page fallback.
3. **Kiến trúc NVIDIA:** Page Elements v3 → Table Structure v1 → Nemotron OCR v2 → Embedding.
4. **Kiến trúc thử nghiệm:** line detector → language probe → Tesseract `vie` hoặc Nemotron OCR v2 → quality check → output.

Bốn route này là các nhánh lựa chọn; không có nghĩa mọi job đều gọi toàn bộ component cùng lúc.

| Đầu vào | Điều phối | Đầu ra |
|---|---|---|
| PDF native/scan, office, spreadsheet, image, audio và video | Nhận diện → PDFium → Page Elements → chọn route phù hợp | Text, bbox, table và visual → clean/chunk → embedding |

## 2. Kiến trúc chung

Luồng trước khi chọn pipeline được giữ thống nhất:

```text
File đầu vào → nhận diện định dạng → DocToPdf / PDFSplit → PDFium
→ Page Elements → chọn 1 trong 4 pipeline → clean / chunk → embedding
```

Hệ thống nhận PDF, DOCX/PPTX, TXT, HTML, spreadsheet, image, audio và video. XLSX/XLS/CSV đi qua parser spreadsheet trực tiếp, không rasterize hoặc OCR mặc định.

PDFium đọc native text và bbox khi có thể; trang scan hoặc vùng thiếu text được render thành ảnh. Page Elements v3 cung cấp vùng text, title, table, chart, image và infographic. Từ đây selector quyết định route OCR thực tế.

### Nguyên tắc dùng chung

| Có native text | Cần OCR | Sau OCR |
|---|---|---|
| Giữ text và bbox để giảm số crop, giảm request và bảo toàn reading order. | Render full-page hoặc crop theo geometry khi trang scan hay vùng thiếu text. | Ghép text, table và visual theo page order rồi clean/chunk trước embedding. |

## 3. Pipeline Qwen

```text
PDFium text + raster → Page Elements → detect / crop → queue streaming → Qwen VLM → page order
```

Qwen giữ native text làm nguồn chính ở trang native. Vùng thiếu character geometry, table hoặc visual hợp lệ mới đi tiếp qua VLM. Trang scan gửi ảnh nguyên trang; table gửi crop riêng để dựng Markdown.

**Cấu hình:** extract batch 16 với bốn worker; detect/crop batch 128; streaming block 16, queue hai block; Qwen batch 8, tối đa tám request đồng thời; vLLM max model length 32K, max sequence 8, max batched tokens 4096, GPU utilization 0.20. Prompt profile đang dùng là `word_repair`, temperature 0 và thinking tắt.

**Số đo đã ghi:** scan 40 trang 32,69 giây, 40/40 đúng thứ tự, 41 requests, khoảng 1.097 token/s aggregate, request đầu sau 2,38 giây; VRAM khoảng 15,1 GB. Native 4 trang 0,59 giây; report native 6 trang 1,87 giây. `word_repair` có similarity 0,9991 và exact 2/4 trên corpus 4 trang.

## 4. Pipeline Mistral

```text
PDFium → Page Elements semantic bbox → semantic crop → Ministral FP8 → OCR output
```

Mistral giữ native text có geometry, sau đó gửi semantic crop text/title/table/visual khi cần. Table dùng whole-table crop để trả Markdown. Scan hoặc layout yếu có thể thêm full-page OCR unit với bbox `[0, 0, 1, 1]`.

**Cấu hình:** `mistralai/Ministral-3-3B-Instruct-2512`; max model length 8192; max sequence 10; max batched tokens 4096; GPU utilization 0.33; VRAM dùng trong báo cáo là **16.5 GB**; VLM batch/workers 10; max output 1024 token. Table Structure, line detector và language probe không bật trong route này.

**Kết quả thử nghiệm:**

| Hạng mục | Kết quả đo | Ghi chú |
|---|---|---|
| Mistral · scan 40 trang | 53,89s · file 40 trang · 41 rows | Thứ tự/request đầu chưa được log lưu |
| Mistral · VRAM | 4,44GiB model load + 10,17GiB KV cache | Page Elements startup 470MiB · không có peak per-job |
| Mistral · native 4 trang | 1,05s · 4 trang · 6 rows | 8 requests: 1 text + 5 table + 2 visual |
| Mistral · report scan 6 trang | 11,13s · 7 rows | 7 requests: 6 full-page + 1 table · không lỗi |
| Mistral · chất lượng | Regression 4 rows trong 4,66s | Similarity/ground truth chưa được persist |

## 5. Kiến trúc NVIDIA

```text
Page Elements v3 → Table Structure v1 → Nemotron OCR v2 → Embedding
```

Đây là chuỗi component của kiến trúc NVIDIA trong báo cáo tổng quan. Page Elements tìm semantic region; Table Structure lấy cell, row và column; Nemotron OCR v2 đọc crop hoặc full-page; cuối cùng dữ liệu được chuẩn bị cho vector representation.

Các chức năng được mô tả gồm nhận diện file, DocToPdf/PDFSplit, PDFium, Page Elements, Table Structure, Nemotron OCR và clean/chunk/embedding. Kết quả test development được ghi nhận ổn định hơn với file tiếng Anh; tiếng Việt vẫn có trường hợp sai dấu hoặc cần route OCR phù hợp.

## 6. Kiến trúc thử nghiệm

```text
OCR unit → language probe → language gate
                         ├→ tiếng Việt → Tesseract `vie` → quality check → output
                         └→ English / mixed / uncertain → Nemotron OCR v2 → output
```

Kiến trúc này bổ sung PP-OCRv6 medium cho line detector và line crop, Tesseract 5 cho probe/tiếng Việt, cùng Nemotron OCR v2 làm backend tiếng Anh và fallback. Quality check dùng ngưỡng `0.80`.

Các route được thiết kế cho crop text/title/table-cell. Với tiếng Việt, kết quả Tesseract được giữ nếu đạt quality; với English, mixed hoặc uncertain, Nemotron nhận crop trực tiếp. Tài liệu cũng ghi nhận language probe và fallback giúp tăng độ chính xác nhưng làm thời gian xử lý dài hơn.

## 7. Kết quả và các case đặc thù

### 7.1. Benchmark đối chiếu

| Hạng mục | Qwen | Mistral |
|---|---|---|
| Scan 40 trang | 32,69 giây; 40/40 đúng thứ tự; 41 requests; request đầu 2,38 giây | 53,89 giây; 41 rows; request count và first-request chưa lưu trong log cũ |
| Native 4 trang | 0,59 giây; 4/4 native; 1 table request; `errors=[]` | 1,05 giây; 4 trang; 6 rows; 8 requests |
| Report 6 trang | 1,87 giây; 3 bbox gần toàn trang bị loại; không lỗi | 11,13 giây; 7 rows; 6 full-page + 1 table request |
| Chất lượng | similarity 0,9991; exact 2/4 | Chưa có similarity/ground truth |

### 7.2. Scan, file nhiều trang, dấu và hình ảnh

- **File scan:** có thể dùng full-page OCR, overlapping tiles hoặc line detector kết hợp projection fallback tùy route.
- **File nhiều trang:** giữ page number và reading order; latency tăng theo số trang và số crop.
- **Dấu:** stamp detector có thể phát hiện, nhưng cấu hình mô tả `extract_stamps = false`; stamp không mặc định thành node graph.
- **Tiêu ngữ:** xử lý như text/title khi Page Elements nhận diện được.
- **Hình ảnh và visual:** giữ image evidence khi phù hợp; bbox gần toàn trang chứa nhiều text bị gate loại trước khi OCR.

## Kết luận

NeMo-Retriever có một luồng ingest chung và bốn route được mô tả theo mục tiêu khác nhau: Qwen cho scan/native VLM, Mistral cho semantic crop, NVIDIA cho chuỗi component chuẩn và kiến trúc thử nghiệm cho routing ngôn ngữ. Các số benchmark được giữ theo đúng input và nguồn của từng phép đo; trường chưa được log lưu lại không được nội suy.
