/* IngestDebugView — one-document, end-to-end ingest probe.
 *
 * This view deliberately calls the service ingest contract directly:
 *
 *   POST /v1/ingest/job
 *   POST /v1/ingest/job/{job_id}/whole
 *   GET  /v1/ingest/job/{job_id}/document/{document_id}
 *   GET  /v1/ingest/job/{job_id}/events (SSE)
 *
 * It is a debugging surface, not a second ingestion implementation. The
 * backend remains responsible for classification, extraction, embedding,
 * VectorDB writes, and result retention.
 */

/*
 * The upload/debug screen used to stop at result_data.  That is deliberately
 * compact now, so the page image and normalized OCR boxes live in the visual
 * evidence sidecar instead.  Keep this preview here as well as in the job
 * detail view: after an upload, this is the first screen users actually see.
 */
function isMissingJobResponse(response) {
  return response && (response.status === 404 || response.status === 410);
}

const MISSING_JOB_MESSAGE = 'Backend đã restart nên job không còn trong bộ nhớ. Hãy upload lại tài liệu.';
const MISSING_VISUAL_MESSAGE = 'Backend đã restart nên visual evidence của job không còn. Hãy upload lại tài liệu.';

function VisualEvidencePreview({ jobId, documentId, status, enabled, unavailable, sourceFile, sourceUrl }) {
  const [evidence, setEvidence] = React.useState(null);
  const [selectedPage, setSelectedPage] = React.useState(null);
  const [hoveredBlock, setHoveredBlock] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [pageImageFailed, setPageImageFailed] = React.useState(false);
  const blockRefs = React.useRef({});

  React.useEffect(() => {
    setPageImageFailed(false);
  }, [selectedPage, evidence && evidence.image_endpoint]);

  React.useEffect(() => {
    if (!hoveredBlock) return;
    const node = blockRefs.current[hoveredBlock];
    if (node && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [hoveredBlock]);

  React.useEffect(() => {
    if (!enabled || !jobId || !documentId || unavailable) {
      setEvidence(null);
      setSelectedPage(null);
      if (!unavailable) setError(null);
      return undefined;
    }

    let stopped = false;
    let timer = null;

    async function load() {
      setLoading(true);
      try {
        const response = await fetch(
          `/v1/dashboard/api/jobs/${encodeURIComponent(jobId)}/documents/${encodeURIComponent(documentId)}/visual`,
          { cache: 'no-store' },
        );
        if (isMissingJobResponse(response)) {
          if (!stopped) {
            setEvidence(null);
            setError(MISSING_VISUAL_MESSAGE);
          }
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (stopped) return;

        if (data.available && Array.isArray(data.pages) && data.pages.length > 0) {
          setEvidence(data);
          setSelectedPage(prev => data.pages.some(page => page.page_number === prev)
            ? prev
            : data.pages[0].page_number);
          setError(null);
        } else {
          setEvidence(null);
          setError(status === 'completed'
            ? 'Backend chưa có visual evidence cho tài liệu này. Hãy bật Visual evidence trước khi upload.'
            : null);
        }

        // A sidecar is normally committed together with document completion,
        // but polling also makes the preview appear during a long ingest.
        if (!['completed', 'failed'].includes(status)) timer = setTimeout(load, 2000);
      } catch (err) {
        if (!stopped) {
          setError(String(err && err.message || err));
          if (!['completed', 'failed'].includes(status)) timer = setTimeout(load, 3000);
        }
      } finally {
        if (!stopped) setLoading(false);
      }
    }

    load();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId, documentId, status, enabled, unavailable]);

  if (!enabled || !jobId || !documentId) return null;

  const pages = evidence && Array.isArray(evidence.pages) ? evidence.pages : [];
  const page = pages.find(item => item.page_number === selectedPage) || pages[0] || null;
  const pageImageUrl = evidence && evidence.image_endpoint && page && page.image_available
    ? evidence.image_endpoint.replace('{page_number}', encodeURIComponent(String(page.page_number)))
    : null;
  const visualTypes = ['image', 'chart', 'infographic', 'stamp'];
  const rawBlocks = page
    ? (page.blocks || []).filter(block => block && Array.isArray(block.bbox) && block.bbox.length === 4)
    : [];
  const seenVisualBboxes = new Set();
  const blocks = rawBlocks.map((block, index) => {
    const key = String(block.id || `p${page.page_number}-b${index}`);
    const isVisual = visualTypes.includes(String(block.content_type || ''));
    const imageUrl = isVisual && evidence && evidence.block_image_endpoint
      ? evidence.block_image_endpoint
          .replace('{page_number}', encodeURIComponent(String(page.page_number)))
          .replace('{block_id}', encodeURIComponent(String(block.id || key)))
      : null;
    return {
      key,
      text: block.text || '',
      bbox: block.bbox,
      model_bbox: block.model_bbox || block.bbox,
      ocr_source: block.ocr_source || null,
      ocr_mode: block.ocr_mode || null,
      content_type: block.content_type || 'text',
      label_name: block.label_name || null,
      image_url: imageUrl,
      is_visual: isVisual,
      reading_order: block.reading_order || index + 1,
    };
  }).filter(block => {
    // The sidecar may contain an ``image`` row and its original
    // ``infographic/chart`` row at exactly the same geometry. Keep one crop.
    if (block.is_visual) {
      const bboxKey = block.bbox.slice(0, 4).map(value => Number(value).toFixed(4)).join(',');
      if (seenVisualBboxes.has(bboxKey)) return false;
      seenVisualBboxes.add(bboxKey);
      return true;
    }
    // Empty title/header/footer detections are not parsed output and only
    // produce misleading "Block không có text" cards in the debug view.
    return String(block.text || '').trim().length > 0;
  });
  const visualBlockCount = blocks.filter(block => block.is_visual).length;

  const fallbackImage = !pageImageFailed && pageImageUrl && React.createElement('img', {
    src: pageImageUrl,
    alt: `Ảnh trang ${page && page.page_number || ''} sau parse`,
    onError: () => setPageImageFailed(true),
    style: { display: 'block', width: '100%', maxHeight: 680, objectFit: 'contain', background: '#fff' },
  });
  const sourcePdfFallback = sourceFile && sourceFile.type === 'application/pdf'
    && typeof PdfOverlayView === 'function'
    ? React.createElement(PdfOverlayView, {
        blob: sourceFile,
        pageNumber: page && page.page_number,
        items: blocks,
        hoveredRow: hoveredBlock,
        onHoverItem: setHoveredBlock,
      })
    : sourceFile && sourceUrl && String(sourceFile.type || '').startsWith('image/')
      ? React.createElement('img', {
          src: sourceUrl,
          alt: `Ảnh gốc · trang ${page && page.page_number || ''}`,
          style: { display: 'block', width: '100%', maxHeight: 680, objectFit: 'contain', background: '#fff' },
        })
      : null;
  const imageView = !pageImageFailed && pageImageUrl && typeof PdfOverlayView === 'function'
    ? React.createElement(PdfOverlayView, {
        imageUrl: pageImageUrl,
        pageNumber: page.page_number,
        items: blocks,
        hoveredRow: hoveredBlock,
        onHoverItem: setHoveredBlock,
        onImageError: () => setPageImageFailed(true),
      })
    : sourcePdfFallback || fallbackImage || React.createElement('div', {
        className: 'empty-state',
        style: { minHeight: 360, padding: 24 },
      }, pageImageFailed ? 'Ảnh trang backend lỗi, đang thử ảnh gốc của file.' : 'Chưa có ảnh trang để hiển thị.');

  return React.createElement('div', { className: 'card', style: { marginBottom: 20 } },
    React.createElement('div', { className: 'card-title' }, 'Ảnh sau parse · OCR bbox'),
    React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 12 } },
      React.createElement('div', { style: { flex: 1, color: 'var(--nv-text-muted)', fontSize: 12 } },
        evidence
          ? `${evidence.page_count || pages.length} trang · ${blocks.length} block hiển thị · ${visualBlockCount} crop visual · lia chuột lên bbox để xem text tương ứng`
          : (loading ? 'Đang chờ ảnh và bbox từ worker…' : 'Chưa có ảnh visual evidence'),
      ),
      pages.length > 0 && React.createElement('select', {
        className: 'input',
        style: { padding: '6px 9px', minWidth: 150 },
        value: page ? page.page_number : '',
        onChange: event => { setSelectedPage(Number(event.target.value)); setHoveredBlock(null); },
        'aria-label': 'Chọn trang ảnh sau parse',
      }, pages.map(item => React.createElement('option', { key: item.page_number, value: item.page_number }, `Trang ${item.page_number} · ${item.block_count || (item.blocks || []).length} block`))),
    ),
    error && React.createElement('div', { style: { color: 'var(--nv-yellow)', fontSize: 12, marginBottom: 12 } }, `Visual evidence: ${error}`),
    page
      ? React.createElement('div', null,
          imageView,
          typeof PageContentPanel === 'function' && React.createElement(PageContentPanel, {
            pageNumber: page.page_number,
            items: blocks,
            sourcePdfBlob: sourceFile && sourceFile.type === 'application/pdf' ? sourceFile : null,
            hoveredRow: hoveredBlock,
            onHoverItem: setHoveredBlock,
            onOpenVisual: item => item && item.image_url && window.open(item.image_url, '_blank', 'noopener,noreferrer'),
          }),
          typeof VisualCropGallery === 'function' && React.createElement(VisualCropGallery, {
            page: { ...page, blocks },
            onOpenVisual: item => item && item.image_url && window.open(item.image_url, '_blank', 'noopener,noreferrer'),
          }),
        )
      : !loading && React.createElement('div', { className: 'empty-state', style: { padding: 24 } }, 'Không có ảnh để hiển thị.'),
  );
}

const OPTION4_DEFAULT_LANGUAGE = 'auto';

function IngestDebugView() {
  const [file, setFile] = React.useState(null);
  const [label, setLabel] = React.useState('dashboard-debug');
  // Pipeline 7 is the active server-owned semantic OCR route.
  // Keep the endpoints server-owned; the selector chooses the backend policy.
  const [ocrPipeline, setOcrPipeline] = React.useState('pipeline-option7');
  const [metadataText, setMetadataText] = React.useState('{}');
  const [retainResults, setRetainResults] = React.useState(false);
  const [returnEmbeddings, setReturnEmbeddings] = React.useState(false);
  const [returnImages, setReturnImages] = React.useState(false);
  const [visualEvidence, setVisualEvidence] = React.useState(true);
  const [job, setJob] = React.useState(null);
  const [document, setDocument] = React.useState(null);
  const [events, setEvents] = React.useState([]);
  const [sourceUrl, setSourceUrl] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [sseStatus, setSseStatus] = React.useState('—');
  const [jobUnavailable, setJobUnavailable] = React.useState(false);
  const [pipeline7Status, setPipeline7Status] = React.useState('checking');

  React.useEffect(() => {
    let stopped = false;
    fetch('/v1/ingest/pipeline-config', { cache: 'no-store' })
      .then(response => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
      .then(config => {
        if (stopped) return;
        const pipelines = config && config.pipelines || {};
        const configured = Object.values(pipelines).some(item => {
          const endpoints = item && item.nim_endpoints || {};
          return Boolean(endpoints.ministral_vlm_invoke_url);
        });
        setPipeline7Status(configured ? 'configured' : 'missing');
      })
      .catch(() => {
        if (!stopped) setPipeline7Status('unknown');
      });
    return () => { stopped = true; };
  }, []);

  React.useEffect(() => {
    if (!file) { setSourceUrl(null); return undefined; }
    const url = URL.createObjectURL(file);
    setSourceUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function pretty(value) {
    if (value == null) return '—';
    try { return JSON.stringify(value, null, 2); } catch { return String(value); }
  }

  async function responseError(response) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || body.message || pretty(body);
    } catch {}
    return detail;
  }

  function selectFile(e) {
    setFile((e.target.files && e.target.files[0]) || null);
    setError(null);
  }

  async function startIngest(e) {
    e.preventDefault();
    if (!file) { setError('Chọn một file trước khi tạo job.'); return; }

    let userMetadata;
    try {
      userMetadata = JSON.parse(metadataText || '{}');
      if (!userMetadata || Array.isArray(userMetadata) || typeof userMetadata !== 'object') {
        throw new Error('metadata phải là JSON object');
      }
    } catch (err) {
      setError(`Metadata JSON không hợp lệ: ${err.message}`);
      return;
    }

    setBusy(true);
    setError(null);
    setJob(null);
    setDocument(null);
    setJobUnavailable(false);
    setEvents([]);
    setSseStatus('—');
    try {
      const createResponse = await fetch('/v1/ingest/job', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          expected_documents: 1,
          label: label || null,
          metadata: userMetadata,
          retain_results: retainResults,
        }),
      });
      if (!createResponse.ok) throw new Error(`Tạo job thất bại: ${await responseError(createResponse)}`);
      const created = await createResponse.json();
      setJob(created);
      if (window.NemoDebugStore) {
        window.NemoDebugStore.save(created.job_id, file).catch(() => {
          // Source preview is a convenience; ingest must not fail when a
          // browser blocks IndexedDB (private mode, storage quota, etc.).
        });
      }

      const form = new FormData();
      form.append('file', file, file.name);
      const pipelineConfig = {
        ...(returnEmbeddings ? { return_embeddings: true } : {}),
        ...(returnImages ? { return_images: true } : {}),
        ...(visualEvidence ? { visual_evidence: true } : {}),
        // Option 4 owns language routing on the backend. Do not send the
        // display-only `auto` value into the legacy ExtractParams literal,
        // which intentionally accepts only multi/english/vietnamese.
      };
      const ingestMetadata = {
        filename: file.name,
        content_type: file.type || null,
        metadata: {
          ...userMetadata,
          // Persist the selected named OCR pipeline in upload metadata. The
          // backend resolves its server-owned sidecar URLs per request.
          ocr_pipeline: ocrPipeline,
        },
      };
      if (Object.keys(pipelineConfig).length > 0) {
        ingestMetadata.pipeline = pipelineConfig;
      }
      form.append('metadata', JSON.stringify(ingestMetadata));

      const uploadResponse = await fetch(`/v1/ingest/job/${created.job_id}/whole`, {
        method: 'POST',
        body: form,
      });
      if (!uploadResponse.ok) throw new Error(`Upload thất bại: ${await responseError(uploadResponse)}`);
      const accepted = await uploadResponse.json();
      setDocument({ ...accepted, status: 'pending', job_id: created.job_id });
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  React.useEffect(() => {
    if (!job || !document || !document.document_id) return undefined;
    let stopped = false;
    let timer = null;
    const jobId = job.job_id;
    const docId = document.document_id;

    async function poll() {
      try {
        const response = await fetch(`/v1/ingest/job/${jobId}/document/${docId}`);
        if (isMissingJobResponse(response)) {
          if (!stopped) {
            setJobUnavailable(true);
            setDocument(prev => prev ? {
              ...prev,
              status: 'failed',
              job_lost: true,
              error: MISSING_JOB_MESSAGE,
            } : prev);
            setSseStatus('stopped');
            setError(MISSING_JOB_MESSAGE);
          }
          return;
        }
        if (!response.ok && response.status !== 202) throw new Error(await responseError(response));
        const data = await response.json();
        if (stopped) return;
        setDocument(data);
        if (data.status === 'completed' || data.status === 'failed') return;
        timer = setTimeout(poll, 2000);
      } catch (err) {
        if (!stopped) {
          setError(`Đọc status thất bại: ${err.message || err}`);
          timer = setTimeout(poll, 3000);
        }
      }
    }
    poll();
    return () => { stopped = true; if (timer) clearTimeout(timer); };
  }, [job && job.job_id, document && document.document_id]);

  React.useEffect(() => {
    if (!job || !job.job_id || jobUnavailable) return undefined;
    let es;
    let retry;
    let stopped = false;
    const documentId = document && document.document_id;

    function connect() {
      if (stopped || jobUnavailable) return;
      setSseStatus('connecting');
      es = new EventSource(`/v1/ingest/job/${job.job_id}/events`);
      const onEvent = (kind) => (event) => {
        try {
          const payload = JSON.parse(event.data);
          setEvents(prev => [{ kind, payload, at: new Date().toLocaleTimeString() }, ...prev].slice(0, 80));
          if (payload.status && (!documentId || payload.document_id === documentId)) {
            setDocument(prev => prev ? {
              ...prev,
              status: payload.status,
              error: payload.error,
              pipeline_diagnostics: payload.pipeline_diagnostics || prev.pipeline_diagnostics,
            } : prev);
          }
          if (['completed', 'failed', 'job_finalized', 'job_partial', 'job_failed'].includes(kind)) {
            stopped = true;
            setSseStatus('closed');
            if (es) es.close();
          }
        } catch {}
      };
      ['pending', 'processing', 'completed', 'failed', 'job_created', 'job_started', 'job_progress', 'job_finalized', 'job_partial', 'job_failed']
        .forEach(kind => es.addEventListener(kind, onEvent(kind)));
      es.onopen = () => setSseStatus('connected');
      es.onerror = () => {
        setSseStatus('disconnected');
        es.close();
        if (!stopped && !jobUnavailable) retry = setTimeout(connect, 3000);
      };
    }
    connect();
    return () => {
      stopped = true;
      if (es) es.close();
      if (retry) clearTimeout(retry);
    };
  }, [job && job.job_id, document && document.document_id, jobUnavailable]);

  function statusBadge(status) {
    const cls = {
      completed: 'badge-green', failed: 'badge-red', processing: 'badge-yellow',
      pending: 'badge-blue', accepted: 'badge-blue', partial_success: 'badge-yellow',
    }[status] || 'badge-dim';
    const labels = { completed: 'hoàn tất', failed: 'lỗi', processing: 'đang xử lý', pending: 'đang chờ', accepted: 'đã tiếp nhận', partial_success: 'thành công một phần' };
    return React.createElement('span', { className: `badge ${cls}` }, labels[status] || status || '—');
  }

  function isVector(value) {
    return Array.isArray(value) && value.length > 0 && value.every(v => typeof v === 'number');
  }

  function findEmbedding(value, path = '') {
    if (isVector(value)) return { path, vector: value };
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) {
        const found = findEmbedding(value[i], `${path}[${i}]`);
        if (found) return found;
      }
      return null;
    }
    if (value && typeof value === 'object') {
      const entries = Object.entries(value);
      for (const [key, nested] of entries) {
        const lower = key.toLowerCase();
        if (lower.includes('embedding') || lower === 'vector' || lower === 'vectors') {
          if (isVector(nested)) return { path: path ? `${path}.${key}` : key, vector: nested };
        }
        const found = findEmbedding(nested, path ? `${path}.${key}` : key);
        if (found) return found;
      }
    }
    return null;
  }

  function outputAnalysis() {
    const rows = (document && document.result_data) || [];
    const texts = rows.map((row, i) => {
      if (!row || typeof row !== 'object') return { index: i, text: String(row) };
      return {
        index: i,
        text: row.text || row.content || row.markdown || (row.metadata && row.metadata.content) || '',
        page: row.page_number || (row.metadata && row.metadata.page_number),
      };
    }).filter(item => item.text);
    const embeddings = [];
    rows.forEach((row, i) => {
      const found = findEmbedding(row);
      if (found) embeddings.push({ row: i, ...found });
    });
    const metadata = rows.map(row => row && typeof row === 'object' ? row.metadata : null).filter(Boolean);
    return { rows, texts, embeddings, metadata };
  }

  const analysis = outputAnalysis();
  const terminal = document && ['completed', 'failed'].includes(document.status);
  const inputStyle = { width: '100%', marginBottom: 10 };
  const codeStyle = {
    background: 'var(--nv-bg)', border: '1px solid var(--nv-border)', borderRadius: 6,
    padding: 12, overflow: 'auto', maxHeight: 360, whiteSpace: 'pre-wrap',
    fontSize: 11, lineHeight: 1.45,
  };

  return React.createElement(React.Fragment, null,
    React.createElement('div', { className: 'card', style: { marginBottom: 20 } },
      React.createElement('div', { className: 'card-title' }, 'Kiểm thử ingest trực tiếp'),
      React.createElement('div', { style: { color: 'var(--nv-text-muted)', fontSize: 12, marginBottom: 16 } },
        'Tạo một job thật trên Retriever và kiểm tra output. Với PDF lớn, chỉ bật giữ raw result khi cần debug.'
      ),
      React.createElement('form', { onSubmit: startIngest },
        React.createElement('input', { type: 'file', className: 'input', style: inputStyle, onChange: selectFile }),
        file && React.createElement('div', { className: 'mono', style: { fontSize: 11, marginBottom: 10, color: 'var(--nv-green)' } },
          `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MiB`
        ),
        React.createElement('input', {
          className: 'input', style: inputStyle, value: label,
          onChange: e => setLabel(e.target.value), placeholder: 'Nhãn job',
        }),
        React.createElement('label', { style: { display: 'block', fontSize: 12, marginBottom: 6 } },
          'Pipeline OCR',
          React.createElement('select', {
            className: 'input',
            style: { ...inputStyle, marginTop: 6, marginBottom: 0 },
            value: ocrPipeline,
            onChange: e => setOcrPipeline(e.target.value),
            'aria-label': 'Chọn pipeline OCR',
          },
            React.createElement('option', { value: 'pipeline-nemotron-ocr' }, 'Pipeline Nemotron OCR'),
            React.createElement('option', { value: 'pipeline-ppocrv6' }, 'Pipeline 2 · Tối ưu tốc độ v1'),
            React.createElement('option', { value: 'pipeline-option3' }, 'Option 3 · Nemotron → router → VietOCR (Việt)'),
            React.createElement('option', { value: 'pipeline-option4' }, 'Option 4 · Auto bilingual: Việt → Tesseract · Anh/không chắc → Nemotron'),
            React.createElement('option', { value: 'pipeline-option5' }, 'Pipeline 5 · global document batch (fast)'),
            React.createElement('option', { value: 'pipeline-option6' }, 'Pipeline 6 · Page Detect → Qwen 3.5 VLM · Markdown table'),
            React.createElement('option', {
              value: 'pipeline-option7',
              disabled: pipeline7Status === 'missing',
            }, `Pipeline 7 · Ministral 3 3B FP8 · semantic OCR${pipeline7Status === 'configured' ? ' · đã nối' : pipeline7Status === 'missing' ? ' · chưa cấu hình' : ''}`),
          ),
        ),
        ocrPipeline === 'pipeline-option6' && React.createElement('div', {
          style: { marginTop: -2, marginBottom: 12, color: 'var(--nv-green)', fontSize: 12 },
        }, 'Pipeline 6 đang dùng Qwen3.5-2B-NVFP4 qua vLLM; Page Elements detect ảnh/chart và VLM trả text/Markdown.'),
        ocrPipeline === 'pipeline-option7' && React.createElement('div', {
          style: { marginTop: -2, marginBottom: 12, color: pipeline7Status === 'configured' ? 'var(--nv-green)' : 'var(--nv-yellow)', fontSize: 12 },
        }, pipeline7Status === 'configured'
          ? 'Pipeline 7: Page Elements tạo semantic text/title/table bbox và giữ visual evidence; Ministral FP8 chỉ OCR semantic crop, whole-table crop hoặc full-page fallback. Table Structure tắt. Không gửi visual crop, language probe hay line detector.'
          : pipeline7Status === 'missing'
            ? 'Backend chưa cấp endpoint Ministral FP8. Bật profile ministral-fp8 rồi restart Retriever.'
            : 'Đang kiểm tra kết nối backend với Ministral FP8…'),
        ocrPipeline === 'pipeline-option4' && React.createElement('label', { style: { display: 'block', fontSize: 12, marginBottom: 6 } },
          'Ngôn ngữ Option 4',
          React.createElement('select', {
            className: 'input',
            style: { ...inputStyle, marginTop: 6, marginBottom: 0 },
            value: OPTION4_DEFAULT_LANGUAGE,
            disabled: true,
            'aria-label': 'Ngôn ngữ Option 4',
          },
            React.createElement('option', { value: OPTION4_DEFAULT_LANGUAGE }, 'Auto song ngữ · Việt → Tesseract vie · Anh/không chắc → Nemotron'),
          ),
        ),
        React.createElement('textarea', {
          className: 'input', style: { ...inputStyle, minHeight: 76, fontFamily: 'JetBrains Mono, monospace' },
          value: metadataText, onChange: e => setMetadataText(e.target.value),
          placeholder: '{"source": "manual-test"}',
        }),
        React.createElement('label', { style: { display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, marginBottom: 14 } },
          React.createElement('input', { type: 'checkbox', checked: retainResults, onChange: e => setRetainResults(e.target.checked) }),
          'Giữ result_data raw trong backend (chỉ nên bật với file nhỏ để tránh response rất lớn)'
        ),
        React.createElement('label', { style: { display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, marginBottom: 14 } },
          React.createElement('input', { type: 'checkbox', checked: returnEmbeddings, onChange: e => setReturnEmbeddings(e.target.checked) }),
          'Yêu cầu trả embedding trong result_data (có thể làm response lớn)'
        ),
        React.createElement('label', { style: { display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, marginBottom: 14 } },
          React.createElement('input', { type: 'checkbox', checked: returnImages, onChange: e => setReturnImages(e.target.checked) }),
          'Giữ ảnh trang và crop visual trong result_data để kiểm tra bbox'
        ),
        React.createElement('label', { style: { display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, marginBottom: 14 } },
          React.createElement('input', { type: 'checkbox', checked: visualEvidence, onChange: e => setVisualEvidence(e.target.checked) }),
          'Visual evidence: ảnh trang + bbox/text để hover (không giữ raw result_data)'
        ),
        React.createElement('button', { className: 'btn btn-primary', disabled: busy || !file, type: 'submit' },
          busy ? 'Đang tạo job và tải lên…' : 'Tạo job và tải file lên'
        )
      )
    ),

    error && React.createElement('div', { className: 'card', style: { marginBottom: 20, color: 'var(--nv-red)', whiteSpace: 'pre-wrap' } }, error),

    job && React.createElement('div', { className: 'card-grid' },
      React.createElement('div', { className: 'card' },
        React.createElement('div', { className: 'card-title' }, 'Job'),
        React.createElement('div', { className: 'mono', style: { fontSize: 11, wordBreak: 'break-all' } }, job.job_id),
        React.createElement('div', { style: { marginTop: 8 } }, statusBadge(document && document.status || job.status)),
      ),
      React.createElement('div', { className: 'card' },
        React.createElement('div', { className: 'card-title' }, 'Tài liệu'),
        React.createElement('div', { style: { fontSize: 13 } }, document && (document.filename || file && file.name) || '—'),
        React.createElement('div', { className: 'mono', style: { fontSize: 10, marginTop: 8, wordBreak: 'break-all' } }, document && document.document_id || 'waiting…'),
      ),
      React.createElement('div', { className: 'card' },
        React.createElement('div', { className: 'card-title' }, 'Trạng thái trực tiếp'),
        React.createElement('div', { style: { fontSize: 13 } }, `Luồng SSE: ${sseStatus}`),
        React.createElement('div', { style: { fontSize: 12, color: 'var(--nv-text-muted)', marginTop: 6 } },
          document ? `${document.result_rows || 0} row kết quả · ${document.elapsed_s != null ? document.elapsed_s.toFixed(2) + ' giây' : 'đang chạy'}` : 'đang tải lên…'
        ),
      )
    ),

    job && document && document.document_id && React.createElement(VisualEvidencePreview, {
      jobId: job.job_id,
      documentId: document.document_id,
      status: document.status,
      enabled: visualEvidence,
      unavailable: jobUnavailable,
      sourceFile: file,
      sourceUrl,
    }),

    document && React.createElement(Option5Diagnostics, {
      diagnostics: document.pipeline_diagnostics,
    }),

    job && React.createElement('div', {
      className: 'card',
      style: { marginBottom: 20, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' },
    },
      React.createElement('div', { style: { flex: 1, color: 'var(--nv-text-muted)', fontSize: 12 } },
        'Job đã được tạo. Mở lịch sử để xem timeline, cấu hình pipeline, từng row và đối chiếu PDF.'
      ),
      React.createElement('button', {
        className: 'btn btn-primary',
        onClick: () => { window.location.hash = `job/${job.job_id}`; },
      }, 'Xem lịch sử job →'),
    ),

    file && sourceUrl && file.type === 'application/pdf' && React.createElement('div', {
      className: 'card', style: { marginBottom: 20 },
    },
      React.createElement('div', { className: 'card-title' }, 'Xem trước PDF gốc'),
      React.createElement('object', {
        data: `${sourceUrl}#page=1`, type: 'application/pdf',
        style: { width: '100%', height: 520, border: '1px solid var(--nv-border)', background: '#fff' },
      }, React.createElement('div', { style: { padding: 24, color: 'var(--nv-text-muted)' } },
        'Trình duyệt không render được PDF. ',
        React.createElement('a', { href: sourceUrl, target: '_blank', rel: 'noreferrer', style: { color: 'var(--nv-green)' } }, 'Mở PDF ở tab mới')
      )),
    ),

    terminal && document.status === 'failed' && React.createElement('div', { className: 'card', style: { marginBottom: 20, color: 'var(--nv-red)' } },
      React.createElement('div', { className: 'card-title' }, 'Lỗi pipeline'), document.error || 'Không rõ lỗi'
    ),

    terminal && document.status === 'completed' && React.createElement(React.Fragment, null,
      React.createElement('div', { className: 'card-grid' },
        React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'card-title' }, 'Text đã trích xuất'),
          React.createElement('div', { className: 'stat-value' }, analysis.texts.length),
          React.createElement('div', { className: 'stat-label' }, 'row có text'),
        ),
        React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'card-title' }, 'Row có metadata'),
          React.createElement('div', { className: 'stat-value' }, analysis.metadata.length),
          React.createElement('div', { className: 'stat-label' }, 'row có metadata'),
        ),
        React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'card-title' }, 'Embedding / vector'),
          React.createElement('div', { className: 'stat-value' }, analysis.embeddings.length),
          React.createElement('div', { className: 'stat-label' }, analysis.embeddings.length ? `${analysis.embeddings[0].vector.length} chiều (row đầu tiên)` : 'không có trong result_data'),
        )
      ),
      React.createElement('div', { className: 'card', style: { marginBottom: 20 } },
        React.createElement('div', { className: 'card-title' }, 'Text đã trích xuất'),
        analysis.texts.length === 0
          ? React.createElement('div', { className: 'empty-state', style: { padding: 20 } }, 'Không có trường text trong kết quả.')
          : React.createElement('div', null, analysis.texts.map(item =>
              React.createElement('div', { key: item.index, style: { borderBottom: '1px solid var(--nv-border)', padding: '10px 0', fontSize: 13 } },
                React.createElement('span', { className: 'mono', style: { color: 'var(--nv-green)', marginRight: 10, fontSize: 11 } },
                  `row ${item.index}${item.page ? ` · page ${item.page}` : ''}`
                ), item.text
              )
            )),
      ),
      React.createElement('div', { className: 'card-grid' },
        React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'card-title' }, 'Xem trước embedding'),
          analysis.embeddings.length === 0
            ? React.createElement('div', { style: { color: 'var(--nv-text-muted)', fontSize: 12 } },
                'Không thấy vector. Nếu backend policy chặn per-request return_embeddings, hãy kiểm tra pipeline_overrides.'
              )
            : analysis.embeddings.slice(0, 8).map(item =>
                React.createElement('div', { key: item.row, style: { marginBottom: 12 } },
                  React.createElement('div', { className: 'mono', style: { fontSize: 11, color: 'var(--nv-green)' } },
                    `row ${item.row} · ${item.path} · dim ${item.vector.length}`
                  ),
                  React.createElement('div', { className: 'mono', style: { fontSize: 10, color: 'var(--nv-text-muted)', wordBreak: 'break-all' } },
                    `[${item.vector.slice(0, 16).map(v => Number(v).toFixed(6)).join(', ')}${item.vector.length > 16 ? ', …' : ''}]`
                  )
                )
              )
        ),
        React.createElement('div', { className: 'card' },
          React.createElement('div', { className: 'card-title' }, 'Metadata'),
          React.createElement('pre', { className: 'mono', style: codeStyle }, pretty(analysis.metadata)),
        )
      ),
      React.createElement('details', { className: 'card', style: { marginBottom: 20 } },
        React.createElement('summary', { style: { cursor: 'pointer', fontWeight: 600, fontSize: 12 } }, `result_data thô (${analysis.rows.length} row)`),
        React.createElement('pre', { className: 'mono', style: { ...codeStyle, marginTop: 12, maxHeight: 620 } }, pretty(analysis.rows)),
      )
    ),

    job && React.createElement('details', { className: 'card', style: { marginBottom: 20 } },
      React.createElement('summary', { style: { cursor: 'pointer', fontWeight: 600, fontSize: 12 } }, `Sự kiện pipeline (${events.length})`),
      React.createElement('pre', { className: 'mono', style: { ...codeStyle, marginTop: 12, maxHeight: 300 } }, pretty(events)),
    )
  );
}
