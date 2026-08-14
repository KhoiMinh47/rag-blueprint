# Báo cáo kiến trúc ingest và các pipeline OCR

Phạm vi của báo cáo này chỉ là phần ingest: tiếp nhận file, phân loại dữ liệu, trích xuất nội dung, OCR, làm sạch, chunk và chuẩn bị dữ liệu cho embedding. Phần query và answer không nằm trong phạm vi.

## 1. Kiến trúc tổng quan

### 1.1. Kiến trúc ingest chung

Dự án có hai kiểu xử lý chính:

```text
Input file
  → phân loại theo phần mở rộng
  → chọn extraction branch
  → trích xuất nội dung
  → làm sạch và loại dữ liệu trùng
  → chunk nếu được cấu hình
  → chuẩn bị dữ liệu cho embedding
```

Các nhánh dữ liệu hiện có:

- PDF/DOCX/PPTX: chuyển về PDF nếu cần, tách theo trang, sau đó chạy PDFium và các stage detection/OCR.
- Ảnh: load ảnh rồi chạy detection/OCR tương ứng.
- TXT/Markdown/JSON/SH/HTML: đọc bằng text hoặc HTML splitter, không đi qua OCR PDF.
- XLSX/XLS/CSV: chạy native spreadsheet parser, không rasterize file và không đi qua OCR mặc định.

### 1.2. Pipeline chính của Option 1

Option 1 là pipeline NVIDIA/Nemotron mặc định:

```text
Input PDF
  → DocToPdfConversion
  → PDFSplit
  → PDFium extraction/render
  → Nemotron Page Elements v3
  → Nemotron Table Structure v1 nếu bật xử lý bảng
  → Nemotron OCR v2 trên các vùng cần OCR
  → scan fallback: full-page + overlapping tiles nếu là trang scan
  → merge/deduplicate OCR blocks
  → clean_content_rows
  → explode/chunk nội dung PDF
  → embedding
```

Chi tiết xử lý:

1. PDFium đọc text native và lưu tạm bbox của từng ký tự trên trang native.
2. Với trang scan hoặc trang không có text, PDFium đánh dấu `needs_ocr_for_text` và render ảnh trang.
3. Page Elements v3 phát hiện các vùng `text`, `title`, `table`, `chart`, `image`, `infographic` và các vùng visual khác.
4. Table Structure v1 phát hiện cell/row/column bbox trong các vùng table.
5. Nemotron OCR v2 nhận ảnh crop của các vùng Page Elements. Text/title chỉ được thêm vào OCR khi trang cần OCR; table/chart/image được xử lý theo các cờ extract tương ứng.
6. Với scan, OCR còn có lớp recall bằng full-page OCR và tile OCR. Các block được ghép lại dựa trên text, bbox, confidence và chất lượng OCR.
7. `clean_content_rows` loại phần native text bị trùng với nội dung table/chart/visual, nhưng vẫn giữ raw text và metadata provenance.
8. Các block text, table, chart, image được tách thành các row phù hợp trước khi chunk và embedding.

Option 1 có thể dùng endpoint NVIDIA NIM từ xa. Source cũng còn các wrapper actor CPU/GPU cho graph, nhưng đường production hiện tại ưu tiên endpoint `ocr_invoke_url`; nếu thiếu endpoint OCR thì scan text có thể bị bỏ qua.

### 1.3. Pipeline chính của Option 4

Option 4 được kích hoạt bằng selector `pipeline-option4`:

```text
Input PDF
  → DocToPdfConversion
  → PDFSplit
  → PDFium extraction/render
  → Nemotron Page Elements v3
  → Nemotron Table Structure v1 nếu bật table
  → build text/title/table-cell OCR units
  → PP-OCRv6 medium line detector
  → language probe bằng Tesseract vie+eng
  → tiếng Việt: Tesseract vie
  → tiếng Anh/mixed/không chắc chắn: Nemotron OCR v2
  → quality check và fallback
  → merge candidate/provenance
  → dựng text/table Markdown
  → output adapter
  → clean/chunk
  → embedding
```

Option 4 không thay thế Page Elements và Table Structure. Nó dùng hai stage NVIDIA đó để lấy semantic region, sau đó thêm một pipeline OCR isolated riêng.

Các điểm chính:

- Page Elements cung cấp các vùng text/title/table.
- Table Structure cung cấp cell geometry.
- PP-OCRv6 detector tách semantic region thành các line crop.
- Code giữ biên trái/phải của vùng cha để tránh mất ký tự đầu hoặc cuối dòng.
- Khi detector không trả line, source có horizontal projection fallback và fallback về crop cha.
- Table cell được OCR riêng, không pad ngang để tránh lẫn chữ từ cell kế bên.
- Kết quả table cell nằm trong `table[*].cells`, tránh tạo bản sao trong text row.
- Trang PDF native được bypass; Option 4 giữ nguyên text native thay vì OCR lại.

## 2. Kiến trúc NVIDIA

Option 1 đang dùng chuỗi thành phần NVIDIA đồng bộ với nhau:

```text
Nemotron Page Elements v3
  → Nemotron Table Structure v1
  → Nemotron OCR v2
  → NVIDIA embedding endpoint/model
```

Các model này cùng được thiết kế cho cùng một hệ sinh thái dữ liệu document:

- Page Elements v3 quyết định vùng semantic trên trang.
- Table Structure v1 dùng vùng table do Page Elements phát hiện để suy ra cell/row/column.
- Nemotron OCR v2 đọc text trong các crop và được source map ngược về bbox trang.
- Các output đều dùng normalized bbox và metadata tương thích với các stage clean/chunk/embedding.

Vì vậy Option 1 có ưu điểm là các stage kết hợp khá khớp: format response, geometry và semantic region đều được source adapter xử lý theo cùng một quy ước.

Theo quan sát trong quá trình phát triển, Option 1 cho tiếng Anh tốt và ổn định hơn. Với tiếng Việt, một số trường hợp có thể mất dấu hoặc nhận sai dấu câu. Source hiện không có bước hậu xử lý chuyên biệt để khôi phục dấu tiếng Việt; `ocr_lang` chỉ cấu hình language mode cho các đường OCR hỗ trợ nó. Vì vậy nhận định “tiếng Việt dễ mất dấu hơn” là phù hợp với hiện trạng thử nghiệm, nhưng chưa phải kết luận từ benchmark chính thức.

## 3. Kiến trúc thử nghiệm

Option 4 bổ sung các thành phần ngoài chuỗi NVIDIA mặc định:

```text
PP-OCRv6 medium detector
Tesseract 5 language probe
Tesseract 5 Vietnamese recognizer
Nemotron OCR v2 fallback
```

Runtime production của Option 4 được cấu hình như sau:

- language probe: `vie+eng`;
- Tesseract recognizer: `vie`;
- page segmentation mode: `PSM 7`;
- ngưỡng probe language: khoảng `0.70`;
- ngưỡng chấp nhận kết quả Tesseract: `0.80`.

Cơ chế nhận diện:

1. Tesseract `vie+eng` đọc thử crop.
2. Nếu phát hiện ký tự đặc trưng tiếng Việt như `ă`, `â`, `đ`, `ê`, `ô`, `ơ`, `ư`, crop được route sang Tesseract `vie`.
3. Nếu là tiếng Anh, chữ Latin không có dấu đặc trưng, text quá ngắn hoặc probe không chắc chắn, crop được route sang Nemotron.
4. Nếu kết quả Tesseract tiếng Việt yếu hoặc không hợp lệ, Nemotron được dùng làm fallback.
5. Kết quả cuối cùng giữ lại backend được chọn, score, candidate còn lại và lý do routing.

Option 4 có khả năng giữ dấu tiếng Việt tốt hơn trong các crop được nhận diện đúng là tiếng Việt. Tuy nhiên nó chậm hơn Option 1 vì:

- mỗi OCR unit phải có thêm một lần language probe;
- Tesseract hiện chạy CPU, chưa tận dụng GPU;
- line detector tạo thêm request và thêm bước crop/map geometry;
- Tesseract có thể cần fallback sang Nemotron.

Do chưa có benchmark chính thức trên cùng một tập dữ liệu, nhận định “Option 4 chậm hơn Option 1 khá nhiều” nên được ghi nhận là kết quả quan sát trong quá trình development, chưa nên xem là con số hiệu năng chính thức.

Một điểm cần làm rõ: tên pipeline có chữ `parallel_fusion`, nhưng cấu hình production mặc định có language routing nên không gọi song song cả Tesseract và Nemotron cho mọi dòng. Nó thường probe trước rồi chọn một backend. Chế độ parallel fusion chỉ xuất hiện khi tắt language routing hoặc dùng cấu hình isolated khác.

## 4. Khi gặp file scan, file nhiễu và xử lý dấu mộc, tiêu ngữ, hình ảnh

### 4.1. Benchmark hiện tại

Hiện chưa có benchmark chính thức trên một tập dữ liệu cố định, có ground truth và cùng điều kiện chạy cho cả Option 1 và Option 4. Các nhận định dưới đây dựa trên các file test và file phát sinh trong quá trình development, vì vậy chỉ nên xem là đánh giá định tính.

### 4.2. File scan và file nhiễu

Với file scan, cả hai option đều vẫn phụ thuộc đáng kể vào chất lượng detection và OCR:

- Option 1 phụ thuộc chủ yếu vào Page Elements v3 và Nemotron OCR v2.
- Option 4 vẫn dùng Page Elements v3 và Table Structure v1, sau đó thêm PP-OCRv6 line detector, Tesseract và Nemotron fallback.

Option 1 có scan fallback bằng full-page OCR và overlapping tiles. Option 4 có fallback theo semantic unit, line detector và horizontal projection. Tuy vậy, nếu ảnh đầu vào nhiễu, mất nét, lệch, có nền phức tạp hoặc chữ quá nhỏ thì cả hai vẫn có thể sai.

Các lỗi còn có thể gặp:

- mất hoặc sai dấu câu;
- nhận sai ký tự gần giống nhau;
- nhận sai dấu tiếng Việt;
- gộp nhầm nhiều dòng;
- bỏ sót text ngoài vùng detector;
- nhận sai tiêu ngữ hoặc text nằm sát đường viền, bảng và hình ảnh.

Vì vậy đánh giá “cả hai option đều ở mức ổn trên scan nhưng vẫn có sai vặt” phù hợp với kết quả test hiện tại, chưa thể xem là đảm bảo cho mọi loại scan.

### 4.3. Dấu mộc

Source có module và actor cho stamp detection, nhưng trong graph ingest chính hiện tại:

- `stamp_needed` đang bị đặt là `False`;
- OCR graph truyền `extract_stamps=False`;
- stamp detector chưa được nối thành một stage mặc định trong đường chạy chính.

Do đó dấu mộc hiện chưa có pipeline xử lý chuyên biệt đáng tin cậy. Page Elements có thể phân loại vùng mộc thành `stamp`, `chart`, `image`, `table` hoặc `text` tùy kết quả model; khi đó nó có thể bị xử lý như visual/text thông thường. Nhận định “dấu mộc hiện chưa xử lý tốt, có thể bị xem như chart/table/text thường” là đúng với source hiện tại.

### 4.4. Tiêu ngữ

Tiêu ngữ hiện chưa có một detector hoặc rule riêng. Nó được xử lý như text/title thông thường:

- Option 1 phụ thuộc Page Elements và Nemotron OCR.
- Option 4 có thể route crop sang Tesseract hoặc Nemotron tùy kết quả language probe.

Vì tiêu ngữ thường nằm trên scan nhiễu, cỡ chữ nhỏ hoặc có dấu tiếng Việt, nó vẫn có thể gặp lỗi mất dấu, sai dấu câu hoặc bị gộp với dòng lân cận.

### 4.5. Hình ảnh và visual region

Option 1 có đường xử lý visual rõ hơn: Page Elements có thể phát hiện image/chart/infographic, sau đó OCR crop và giữ lại image evidence tùy cờ extract.

Option 4 tập trung vào text và table cell. `Option4Pipeline` xây unit với `include_visual_regions=False`, nên visual region không được đưa vào OCR unit thông thường. Vì vậy Option 4 không nên được hiểu là pipeline xử lý đầy đủ hình ảnh, chart và infographic như Option 1.

### 4.6. Kết luận thực tế

- Option 1 phù hợp khi ưu tiên độ khớp với hệ sinh thái NVIDIA, tiếng Anh và độ đơn giản/vận hành.
- Option 4 phù hợp khi dữ liệu có nhiều tiếng Việt cần giữ dấu tốt hơn và chấp nhận latency cao hơn.
- Cả hai hiện chưa có kết quả benchmark chính thức trên cùng một corpus.
- Dấu mộc là điểm yếu rõ ràng của source hiện tại vì stamp detection chưa được nối vào graph chính.
- Scan nhiễu, tiêu ngữ và chữ nhỏ vẫn phụ thuộc mạnh vào chất lượng Page Elements/OCR và có thể sai dấu, sai punctuation hoặc bỏ sót.
- Excel được xử lý native, không phải OCR Option 1/4; embedded image trong Excel hiện mới được ghi metadata và chưa tự động đi qua pipeline OCR.
