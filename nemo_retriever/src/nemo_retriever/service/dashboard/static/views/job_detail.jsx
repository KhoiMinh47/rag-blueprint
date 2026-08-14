/* JobDetailView — single-job drill-down (J10).
 *
 * Renders:
 *   • aggregate header (status, counts, progress bar, label, timestamps)
 *   • throughput + cumulative-completion mini charts (pure SVG)
 *   • paginated document table backed by /v1/dashboard/api/jobs/{id}/documents
 *   • live SSE feed subscribed to /v1/ingest/job/{id}/events
 *
 * Charts are intentionally library-free (no Chart.js / Recharts) so the
 * dashboard ships as plain JSX over the React UMD bundle.
 */

function jobDetailMissingResponse(response) {
  return response && (response.status === 404 || response.status === 410);
}

const JOB_DETAIL_MISSING_MESSAGE = 'Backend đã restart nên job không còn trong bộ nhớ. Hãy upload lại tài liệu.';

function Option5Diagnostics({ diagnostics }) {
  const value = diagnostics && diagnostics.scope === 'document' ? diagnostics : null;
  if (!value) return null;
  const isOption7 = value.pipeline === 'pipeline-option7'
    || value.pipeline_name === 'option7_ministral_vlm'
    || String(value.model || '').toLowerCase().includes('ministral');
  const isOption6 = value.pipeline === 'pipeline-option6'
    || value.pipeline_name === 'option6_page_detect_qwen35_vlm'
    || String(value.model || '').toLowerCase().includes('qwen3.5-2b');
  const timing = value.timing || {};
  const routes = value.route_counts || {};
  const fmt = item => Number.isFinite(Number(item)) ? Number(item).toFixed(2) : '—';
  const pages = Array.isArray(value.probe_pages) ? value.probe_pages.join(', ') : '—';
  const stat = (label, content) => React.createElement('div', {
    style: { padding: '10px 12px', border: '1px solid var(--nv-border)', borderRadius: 6, background: 'var(--nv-surface)' },
  },
    React.createElement('div', { style: { fontSize: 10, color: 'var(--nv-text-muted)', marginBottom: 4 } }, label),
    React.createElement('div', { style: { fontSize: 15, fontWeight: 600 } }, content),
  );
  return React.createElement('div', { className: 'card', style: { marginBottom: 20 } },
    React.createElement('div', { className: 'card-title' }, isOption6 ? 'Pipeline 6 · diagnostics toàn document' : (isOption7 ? 'Pipeline 7 · diagnostics toàn document' : 'Pipeline 5 · diagnostics toàn document')),
    React.createElement('div', { style: { color: 'var(--nv-text-muted)', fontSize: 12, marginBottom: 12 } },
      'Các số liệu này được lưu riêng từ worker, nên vẫn hiển thị khi không giữ result_data.'
    ),
    React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 8, marginBottom: 12 } },
      stat('Ngôn ngữ file', value.language || 'unknown'),
      stat('Phạm vi OCR', isOption6 ? `${value.page_count || 0} trang` : (isOption7 ? `${value.semantic_text_crop_count || 0} semantic crop · ${value.table_crop_count || 0} table crop · ${value.full_page_count || 0} full-page fallback` : `${(value.probe_pages || []).length}/5 · ${pages}`)),
      stat('Semantic unit', (value.unit_count != null ? value.unit_count : value.text_units) != null ? Number(value.unit_count != null ? value.unit_count : value.text_units).toLocaleString() : '—'),
      stat('Batch logic', isOption6 ? `Detect ${value.detector_batch_size || 128} · crop ${value.crop_batch_size || 128} · VLM ${value.vlm_batch_size || 25}` : (isOption7 ? `VLM ${value.vlm_request_count || value.request_count || 0} request` : `Nemotron ${value.nemotron_logical_batches || 0} · Việt ${value.vietnamese_logical_batches || 0}`)),
      stat('Layout', isOption6 ? `${value.table_regions || 0} table · ${value.visual_regions || 0} visual` : (isOption7 ? `${value.table_region_count || 0} table · Table Structure tắt · visual crop 0` : `${value.cache_hits || 0} crop`)),
      stat('Tổng thời gian', `${fmt(timing.total_seconds)}s`),
    ),
    React.createElement('div', { style: { fontSize: 12, color: 'var(--nv-text-muted)', lineHeight: 1.6 } },
      isOption6
        ? `VLM: ${value.text_units || 0} text unit · ${value.table_regions || 0} table · ${value.native_pages || 0} native page. `
        : isOption7
        ? `Semantic OCR: ${value.semantic_text_crop_count || 0} text/title crop · ${value.table_crop_count || 0} whole-table crop · ${value.full_page_count || 0} full-page fallback; Table Structure tắt, visual crop không gửi. `
        : `Route: Việt ${routes.vietnamese || 0} · Anh ${routes.english || 0} · uncertain ${routes.uncertain || 0} · fallback ${value.fallback_count || 0}. `,
      isOption6
        ? `Thời gian VLM: ${fmt(timing.vlm_seconds || Number(timing.text_vlm_seconds || 0) + Number(timing.table_vlm_seconds || 0))}s.`
        : isOption7
        ? `Thời gian VLM: ${fmt(timing.vlm_seconds)}s · semantic batch · ${value.vlm_request_count || value.request_count || 0} request.`
        : `Thời gian model: Nemotron ${fmt(timing.nemotron_seconds)}s · probe/router ${fmt(Number(timing.language_probe_seconds || 0) + Number(timing.language_router_seconds || 0))}s · VietOCR ${fmt(timing.vietnamese_recognizer_seconds)}s.`
    ),
    React.createElement('details', { style: { marginTop: 10 } },
      React.createElement('summary', { style: { cursor: 'pointer', fontSize: 11, color: 'var(--nv-text-muted)' } }, 'Xem diagnostics JSON'),
      React.createElement('pre', { className: 'mono', style: { marginTop: 8, maxHeight: 280, overflow: 'auto', fontSize: 10, whiteSpace: 'pre-wrap' } }, JSON.stringify(value, null, 2)),
    ),
  );
}

const NEMO_BBOX_COLORS = {
  page: { stroke: '#111111', fill: 'rgba(17,17,17,0.035)', label: 'BBox Page Elements', dash: '1.4 0.8' },
  line: { stroke: '#2563eb', fill: 'rgba(37,99,235,0.10)', label: 'BBox line PP-OCRv6', dash: null },
  option3: { stroke: '#7c3aed', fill: 'rgba(124,58,237,0.10)', label: 'BBox Nemotron → VietOCR', dash: null },
  option5: { stroke: '#ea580c', fill: 'rgba(234,88,12,0.10)', label: 'pipeline 5 (fast)', dash: null },
  option6: { stroke: '#db2777', fill: 'rgba(219,39,119,0.10)', label: 'pipeline 6 · Qwen 3.5 VLM', dash: null },
  option7: { stroke: '#0891b2', fill: 'rgba(8,145,178,0.10)', label: 'pipeline 7 · Ministral OCR', dash: null },
  visual: { stroke: '#16a34a', fill: 'rgba(22,163,74,0.11)', label: 'Ảnh / vùng không OCR', dash: null },
};

function nemoBboxIsVisual(item) {
  return ['image', 'chart', 'infographic', 'stamp'].includes(String(item && item.content_type || ''));
}

function nemoBboxSame(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length >= 4 && right.length >= 4
    && left.slice(0, 4).every((value, index) => Math.abs(Number(value) - Number(right[index])) <= 0.001);
}

function nemoBboxArea(bbox) {
  if (!Array.isArray(bbox) || bbox.length < 4) return 0;
  return Math.max(0, Number(bbox[2]) - Number(bbox[0]))
    * Math.max(0, Number(bbox[3]) - Number(bbox[1]));
}

function nemoNormalizeText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim().toLowerCase();
}

function nemoBboxIntersection(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length < 4 || right.length < 4) return 0;
  const x1 = Math.max(Number(left[0]), Number(right[0]));
  const y1 = Math.max(Number(left[1]), Number(right[1]));
  const x2 = Math.min(Number(left[2]), Number(right[2]));
  const y2 = Math.min(Number(left[3]), Number(right[3]));
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

function nemoTextIsInsideVisual(textItem, visualItems) {
  if (!textItem || !Array.isArray(visualItems) || !visualItems.length) return false;
  const textArea = nemoBboxArea(textItem.bbox);
  if (!textArea) return false;
  return visualItems.some(visual => {
    const overlap = nemoBboxIntersection(textItem.bbox, visual && visual.bbox);
    if (!overlap) return false;
    const visualArea = nemoBboxArea(visual && visual.bbox);
    // A line in an infographic is normally completely inside its crop.  The
    // second condition also covers a line that touches a slightly expanded
    // visual bbox without classifying unrelated text next to the image.
    return overlap / textArea >= 0.65 || (visualArea > 0.25 && overlap / textArea >= 0.45);
  });
}

function nemoBlockIsTable(item) {
  const type = String(item && item.content_type || '').toLowerCase();
  return type === 'table' || type === 'spreadsheet_table' || type.includes('table_');
}

function nemoBlockIsNative(item) {
  const reader = String(item && (item.reader_backend || item.reader || '') || '').toLowerCase();
  const mode = String(item && item.ocr_mode || '').toLowerCase();
  const origin = String(item && (item.origin || item.content_origin || item.source || '') || '').toLowerCase();
  return ['native_pdf', 'native_spreadsheet', 'openpyxl', 'python_csv', 'native'].includes(reader)
    || mode.includes('native')
    || origin.includes('native');
}

function nemoTraceBlockDuplicate(left, right) {
  if (!left || !right) return false;
  if (left.block_id && right.block_id && String(left.block_id) === String(right.block_id)) return true;
  const leftType = String(left.content_type || 'text');
  const rightType = String(right.content_type || 'text');
  const sameType = leftType === rightType
    || (nemoBboxIsVisual(left) && nemoBboxIsVisual(right))
    || (nemoBlockIsTable(left) && nemoBlockIsTable(right));
  if (!sameType) return false;
  const leftText = nemoNormalizeText(left.text);
  const rightText = nemoNormalizeText(right.text);
  if (leftText && rightText && leftText === rightText) return true;
  return nemoBboxSame(left.bbox, right.bbox) && (!leftText || !rightText);
}

function nemoMergeTraceBlocks(baseBlocks, visualBlocks) {
  const merged = Array.isArray(baseBlocks) ? baseBlocks.filter(Boolean).map(item => ({ ...item })) : [];
  (Array.isArray(visualBlocks) ? visualBlocks : []).filter(Boolean).forEach(candidate => {
    const duplicateIndex = merged.findIndex(existing => nemoTraceBlockDuplicate(existing, candidate));
    if (duplicateIndex < 0) {
      merged.push(candidate);
      return;
    }
    // Keep the richer sidecar record when the same bbox came from both
    // endpoints, especially its crop URL and visual label.
    const current = merged[duplicateIndex];
    merged[duplicateIndex] = {
      ...current,
      ...candidate,
      text: String(candidate.text || '').trim() ? candidate.text : current.text,
      image_url: candidate.image_url || current.image_url || null,
      block_id: current.block_id || candidate.block_id,
    };
  });
  return merged.sort((left, right) => {
    const leftOrder = Number(left.reading_order || left.row_index || 0);
    const rightOrder = Number(right.reading_order || right.row_index || 0);
    return leftOrder - rightOrder;
  });
}

function nemoMergeTracePages(backendPages, visualPages) {
  const pages = new Map();
  (Array.isArray(backendPages) ? backendPages : []).forEach(page => {
    if (page && page.page_number != null) pages.set(Number(page.page_number), { ...page, blocks: Array.isArray(page.blocks) ? page.blocks.slice() : [] });
  });
  (Array.isArray(visualPages) ? visualPages : []).forEach(page => {
    if (!page || page.page_number == null) return;
    const number = Number(page.page_number);
    const current = pages.get(number) || {};
    const blocks = nemoMergeTraceBlocks(current.blocks, page.blocks);
    pages.set(number, {
      ...current,
      ...page,
      blocks,
      block_count: blocks.length,
      text_chars: blocks.reduce((total, block) => total + String(block.text || '').length, 0),
      content_types: Array.from(new Set(blocks.map(block => block.content_type).filter(Boolean))),
      reader_backend: (current.reader_backend && current.reader_backend !== 'ocr')
        ? current.reader_backend
        : (page.reader_backend || current.reader_backend || null),
    });
  });
  return Array.from(pages.values()).sort((left, right) => Number(left.page_number) - Number(right.page_number));
}

function nemoBboxTooltip(item, label) {
  const text = String(item && item.text || '').trim();
  const compact = text && nemoBboxArea(item && item.bbox) < 0.55 && text.length <= 240;
  return `${label}${compact ? ` · ${text}` : ''}`;
}

function nemoIsOption3(item) {
  const provenance = item && item.provenance && typeof item.provenance === 'object' ? item.provenance : {};
  const pipelineName = String(item && (item.ocr_pipeline_name || item.pipeline_name) || '');
  const source = String(item && item.ocr_source || '');
  const backend = String(item && (item.selected_backend || provenance.selected_backend) || '').toLowerCase();
  return source.startsWith('option2_')
    || source.startsWith('option3_')
    || source.startsWith('option5_')
    || source.startsWith('option6_')
    || source.startsWith('option7_')
    || pipelineName === 'option2_nemotron_language_routed_vietnamese_ocr'
    || pipelineName === 'option3_nemotron_language_routed_vietnamese_ocr'
    || pipelineName === 'option5_nemotron_language_routed_vietnamese_ocr'
    || pipelineName === 'option6_page_detect_qwen35_vlm'
    || pipelineName === 'option7_ministral_vlm'
    || backend === 'vietocr'
    || backend === 'qwen35_vlm';
}

function nemoBboxEntries(item) {
  if (!item || !Array.isArray(item.bbox) || item.bbox.length < 4) return [];
  if (nemoBboxIsVisual(item)) {
    return [{ item, bbox: item.bbox, kind: 'visual', interactive: true }];
  }

  const mode = String(item.ocr_mode || '');
  const lineMode = mode === 'page_elements_ppocr_line';
  const option3Mode = nemoIsOption3(item);
  const pageMode = mode === 'page_elements_box' || lineMode;
  const entries = [];
  if (pageMode && Array.isArray(item.model_bbox) && item.model_bbox.length >= 4) {
    entries.push({
      item,
      bbox: item.model_bbox,
      kind: 'page',
      // The line overlay is stacked above this box and owns hover/click.
      interactive: !lineMode || nemoBboxSame(item.model_bbox, item.bbox),
      hasLine: lineMode,
    });
  }
  entries.push({
    item,
    bbox: item.bbox,
    kind: option3Mode ? (String(item.ocr_pipeline_name || item.pipeline_name || '').includes('option7') ? 'option7' : (String(item.ocr_pipeline_name || item.pipeline_name || '').includes('option6') ? 'option6' : (String(item.ocr_pipeline_name || item.pipeline_name || '').includes('option5') ? 'option5' : 'option3'))) : (lineMode || item.ocr_source ? 'line' : 'page'),
    interactive: true,
    hasLine: lineMode,
  });
  return entries;
}

function PdfOverlayView({ blob, sourceUrl, imageUrl, pageNumber, items, hoveredRow, onHoverItem, onImageError }) {
  const canvasRef = React.useRef(null);
  const [renderState, setRenderState] = React.useState({ loading: false, error: null, width: 612, height: 792 });
  const [boxHover, setBoxHover] = React.useState(null);
  const [showBoxes, setShowBoxes] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    if (imageUrl) {
      setRenderState(prev => ({ ...prev, loading: false, error: null }));
      return undefined;
    }
    if (!blob || !window.pdfjsLib) {
      setRenderState(prev => ({ ...prev, loading: false, error: blob ? 'PDF.js chưa tải được; không thể vẽ bbox.' : null }));
      return undefined;
    }
    setRenderState(prev => ({ ...prev, loading: true, error: null }));
    let loadingTask = null;
    (async () => {
      try {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        if (cancelled) return;
        loadingTask = window.pdfjsLib.getDocument({ data: bytes });
        const pdf = await loadingTask.promise;
        const page = await pdf.getPage(Math.max(1, Number(pageNumber) || 1));
        const viewport = page.getViewport({ scale: 1.5 });
        const canvas = canvasRef.current;
        if (!canvas || cancelled) return;
        canvas.width = Math.ceil(viewport.width);
        canvas.height = Math.ceil(viewport.height);
        const context = canvas.getContext('2d', { alpha: false });
        await page.render({ canvasContext: context, viewport }).promise;
        if (!cancelled) setRenderState({ loading: false, error: null, width: viewport.width, height: viewport.height });
        await pdf.destroy();
      } catch (error) {
        if (!cancelled) setRenderState(prev => ({ ...prev, loading: false, error: String(error && error.message || error) }));
      }
    })();
    return () => {
      cancelled = true;
      if (loadingTask) loadingTask.destroy();
    };
  }, [blob, imageUrl, pageNumber]);

  const overlayItems = (Array.isArray(items) ? items : []).filter(item => item && item.bbox);
  // Draw the original semantic Page Elements box first, then the PP-OCRv6
  // line box. Visual crops are underneath text overlays so they cannot hide
  // the line that was recognized inside/near a chart or image region.
  const overlayEntries = overlayItems
    .flatMap(nemoBboxEntries)
    .sort((left, right) => ({ visual: 0, page: 1, line: 2, option3: 2, option5: 2, option6: 2, option7: 2 }[left.kind] || 9) - ({ visual: 0, page: 1, line: 2, option3: 2, option5: 2, option6: 2, option7: 2 }[right.kind] || 9));
  const hasBbox = overlayEntries.length > 0;
  const activeHover = boxHover || overlayItems.find(item => item.key === hoveredRow) || null;
  return React.createElement('div', null,
    React.createElement('div', {
      style: {
        height: 'min(72vh, 780px)', minHeight: 360, display: 'flex', justifyContent: 'center', alignItems: 'center',
        overflow: 'hidden', border: '1px solid var(--nv-border)', borderRadius: 8, background: 'rgba(0,0,0,0.12)',
      },
    }, React.createElement('div', {
      style: {
        position: 'relative', height: '100%', width: 'auto', maxWidth: '100%', flex: '0 1 auto', aspectRatio: `${renderState.width} / ${renderState.height}`,
        overflow: 'hidden', border: '1px solid rgba(17,17,17,0.26)', background: '#fff',
      },
    },
      imageUrl
        ? React.createElement('img', {
            src: imageUrl,
            alt: `Ảnh trang ${pageNumber} sau parse`,
            onLoad: event => {
              const image = event.currentTarget;
              if (image.naturalWidth && image.naturalHeight) {
                setRenderState(prev => ({ ...prev, width: image.naturalWidth, height: image.naturalHeight, error: null }));
              }
            },
            onError: () => {
              setRenderState(prev => ({ ...prev, error: 'Không tải được ảnh trang từ backend.' }));
              if (onImageError) onImageError();
            },
            style: { position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain' },
          })
        : React.createElement('canvas', { ref: canvasRef, style: { position: 'absolute', inset: 0, width: '100%', height: '100%' } }),
      showBoxes && hasBbox && React.createElement('svg', {
        viewBox: '0 0 100 100', preserveAspectRatio: 'none',
        style: { position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'auto', zIndex: 2 },
      }, overlayEntries.map((entry, index) => {
        const item = entry.item;
        const box = entry.bbox;
        const colors = NEMO_BBOX_COLORS[entry.kind] || NEMO_BBOX_COLORS.line;
        const active = hoveredRow === item.key;
        return React.createElement('rect', {
          key: `${item.key}-${entry.kind}-${index}`, x: box[0] * 100, y: box[1] * 100,
          width: Math.max(0, box[2] - box[0]) * 100, height: Math.max(0, box[3] - box[1]) * 100,
          fill: active
            ? colors.fill.replace(/\d?\.?\d+\)$/, '0.20)')
            : (entry.kind === 'visual' ? 'rgba(22,163,74,0.045)' : colors.fill),
          stroke: colors.stroke, strokeWidth: active ? 0.55 : (entry.kind === 'page' ? 0.30 : 0.24),
          strokeDasharray: colors.dash || undefined,
          style: { cursor: entry.interactive ? 'crosshair' : 'default', pointerEvents: entry.interactive ? 'all' : 'none' },
          onMouseEnter: () => { setBoxHover(item); if (onHoverItem) onHoverItem(item.key); },
          onMouseLeave: () => { setBoxHover(null); if (onHoverItem) onHoverItem(null); },
          onClick: () => onHoverItem && onHoverItem(item.key),
        }, React.createElement('title', null, nemoBboxTooltip(item, colors.label)));
      })),
      activeHover && activeHover.text && React.createElement('div', {
        style: {
          position: 'absolute', left: 10, right: 10, bottom: 10, zIndex: 3,
          maxHeight: 110, overflow: 'auto', padding: '8px 10px', borderRadius: 5,
          background: 'rgba(0,0,0,0.78)', color: '#fff', fontSize: 11, lineHeight: 1.4,
          pointerEvents: 'none', whiteSpace: 'pre-wrap',
        },
      }, activeHover.text),
      renderState.loading && React.createElement('div', { style: { position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#555' } }, 'Đang dựng trang PDF…'),
      renderState.error && React.createElement('div', { style: { position: 'absolute', left: 12, right: 12, bottom: 12, padding: 10, background: 'rgba(0,0,0,0.75)', color: 'var(--nv-yellow)', fontSize: 11 } }, renderState.error),
    )),
    React.createElement('div', { style: { display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginTop: 8, fontSize: 11, color: 'var(--nv-text-muted)' } },
      React.createElement('span', null, hasBbox ? `${overlayItems.length} block · ${overlayEntries.length} lớp bbox` : 'Kết quả chưa có bbox chuẩn hóa'),
      hasBbox && React.createElement('button', {
        type: 'button',
        className: 'btn',
        onClick: () => setShowBoxes(value => !value),
        style: { padding: '4px 8px', fontSize: 10, background: 'var(--nv-surface)', color: 'var(--nv-text)' },
      }, showBoxes ? 'Ẩn bbox để đọc ảnh' : 'Hiện bbox'),
      hasBbox && React.createElement('span', null, showBoxes ? 'lia chuột lên vùng để xem block' : 'ảnh gốc đang hiển thị sạch'),
      Object.entries(NEMO_BBOX_COLORS).map(([kind, colors]) => React.createElement('span', { key: kind, style: { display: 'inline-flex', alignItems: 'center', gap: 5 } },
        React.createElement('span', { style: { width: 18, height: 9, border: `2px ${colors.dash ? 'dashed' : 'solid'} ${colors.stroke}`, background: colors.fill, display: 'inline-block' } }),
        colors.label,
      )),
    ),
  );
}

function ReconstructedPageView({ pageNumber, items, hoveredRow, onHoverItem, onOpenVisual, width, height }) {
  const overlayItems = (Array.isArray(items) ? items : []).filter(item => item && item.bbox);
  const entries = overlayItems.flatMap(nemoBboxEntries).sort((left, right) => {
    const order = { visual: 0, page: 1, line: 2 };
    return (order[left.kind] ?? 9) - (order[right.kind] ?? 9);
  });
  const textItems = overlayItems.filter(item => !nemoBboxIsVisual(item) && item.text);
  // A native page or a scan full-page OCR result can arrive as one large text
  // block. Painting that entire transcript inside one bbox makes every line
  // overlap and turns the reconstructed preview into an unreadable wall of
  // text. Keep the bbox for geometry inspection, but show its text in a
  // separate scrollable panel below the preview.
  const pageTextItems = textItems.filter(item => nemoBboxArea(item.bbox) >= 0.55);
  const inlineTextItems = textItems.filter(item => nemoBboxArea(item.bbox) < 0.55);
  const parsedPageText = pageTextItems
    .map(item => String(item.text || '').trim())
    .filter(Boolean)
    .join('\n\n');

  return React.createElement('div', null,
    React.createElement('div', {
      style: {
        height: 'min(72vh, 780px)', minHeight: 360, display: 'flex', justifyContent: 'center', alignItems: 'center',
        overflow: 'hidden', border: '1px solid var(--nv-border)', borderRadius: 8, background: 'rgba(0,0,0,0.12)',
      },
      title: `PDF giả dựng từ bbox · trang ${pageNumber}`,
    }, React.createElement('div', {
      style: {
        position: 'relative', height: '100%', width: 'auto', maxWidth: '100%', flex: '0 1 auto', aspectRatio: width && height ? `${width} / ${height}` : '0.772', overflow: 'hidden',
        border: '1px solid rgba(17,17,17,0.26)', background: '#fff',
        boxShadow: '0 2px 12px rgba(0,0,0,0.16)',
      },
    },
      React.createElement('div', { style: { position: 'absolute', inset: 0, background: '#fff' } }),
      React.createElement('svg', {
        viewBox: '0 0 100 100', preserveAspectRatio: 'none',
        style: { position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' },
      }, entries.map((entry, index) => {
        const box = entry.bbox;
        const colors = NEMO_BBOX_COLORS[entry.kind] || NEMO_BBOX_COLORS.line;
        const active = hoveredRow === entry.item.key;
        return React.createElement('rect', {
          key: `reconstructed-${entry.item.key}-${entry.kind}-${index}`,
          x: box[0] * 100, y: box[1] * 100,
          width: Math.max(0, box[2] - box[0]) * 100,
          height: Math.max(0, box[3] - box[1]) * 100,
          fill: entry.kind === 'page' ? 'transparent' : colors.fill,
          stroke: colors.stroke,
          strokeWidth: active ? 0.55 : (entry.kind === 'page' ? 0.30 : 0.24),
          strokeDasharray: colors.dash || undefined,
        }, React.createElement('title', null, colors.label));
      })),
      inlineTextItems.map(item => {
        const box = item.bbox;
        const active = hoveredRow === item.key;
        const heightPct = Math.max(0.7, (box[3] - box[1]) * 100);
        const fontSize = Math.max(7, Math.min(18, heightPct * 0.72));
        return React.createElement('div', {
          key: `reconstructed-text-${item.key}`,
          onMouseEnter: () => onHoverItem && onHoverItem(item.key),
          onMouseLeave: () => onHoverItem && onHoverItem(null),
          onClick: () => onHoverItem && onHoverItem(item.key),
          title: nemoBboxTooltip(item, 'Text'),
          style: {
            position: 'absolute', left: `${box[0] * 100}%`, top: `${box[1] * 100}%`,
            width: `${Math.max(0.2, box[2] - box[0]) * 100}%`,
            height: `${Math.max(0.6, box[3] - box[1]) * 100}%`,
            padding: '1px 2px', overflow: 'hidden', whiteSpace: 'normal', wordBreak: 'break-word',
            textOverflow: 'clip', fontSize: `${Math.max(9, fontSize)}px`, lineHeight: 1.12, fontWeight: 550,
            color: '#111827', background: active ? 'rgba(37,99,235,0.20)' : 'rgba(255,255,255,0.96)',
            borderBottom: `1px solid ${active ? '#2563eb' : 'rgba(37,99,235,0.55)'}`,
            cursor: 'crosshair', zIndex: 4,
          },
        }, item.text);
      }),
      overlayItems.filter(nemoBboxIsVisual).map(item => {
        const box = item.bbox;
        const active = hoveredRow === item.key;
        return React.createElement('button', {
          key: `reconstructed-visual-${item.key}`,
          type: 'button',
          onMouseEnter: () => onHoverItem && onHoverItem(item.key),
          onMouseLeave: () => onHoverItem && onHoverItem(null),
          onClick: () => {
            if (onOpenVisual && item.image_url) onOpenVisual(item);
            else if (onHoverItem) onHoverItem(item.key);
          },
          title: nemoBboxTooltip(item, item.content_type || 'visual'),
          style: {
            position: 'absolute', left: `${box[0] * 100}%`, top: `${box[1] * 100}%`,
            width: `${Math.max(0.2, box[2] - box[0]) * 100}%`,
            height: `${Math.max(0.6, box[3] - box[1]) * 100}%`,
            padding: 2, overflow: 'hidden', background: active ? 'rgba(22,163,74,0.24)' : 'rgba(22,163,74,0.11)',
            border: `1px solid ${active ? '#16a34a' : 'rgba(22,163,74,0.75)'}`,
            color: '#166534', fontSize: 10, textAlign: 'left', cursor: 'crosshair', zIndex: 5,
          },
        }, item.text || `Ảnh ${item.content_type || 'visual'}`);
      }),
    )),
    React.createElement('div', { style: { display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginTop: 8, fontSize: 10, color: 'var(--nv-text-muted)' } },
      Object.entries(NEMO_BBOX_COLORS).map(([kind, colors]) => React.createElement('span', { key: kind, style: { display: 'inline-flex', alignItems: 'center', gap: 4 } },
        React.createElement('span', { style: { width: 16, height: 8, border: `2px ${colors.dash ? 'dashed' : 'solid'} ${colors.stroke}`, background: colors.fill, display: 'inline-block' } }),
        colors.label,
      )),
    ),
    parsedPageText && React.createElement('div', {
      style: {
        marginTop: 12, padding: 12, border: '1px solid var(--nv-border)', borderRadius: 8,
        background: 'var(--nv-surface)',
      },
    },
      React.createElement('div', { style: { fontSize: 12, fontWeight: 600, marginBottom: 7 } }, 'Text đã parse · block cấp trang'),
      React.createElement('div', { style: { fontSize: 11, color: 'var(--nv-text-muted)', marginBottom: 8 } }, 'Không vẽ chồng transcript dài lên bbox, nội dung đầy đủ nằm trong khung cuộn này.'),
      React.createElement('pre', {
        className: 'mono',
        style: {
          maxHeight: 300, overflowY: 'auto', margin: 0, padding: 10,
          whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.5,
          background: 'var(--nv-bg)', border: '1px solid var(--nv-border)', borderRadius: 6,
          fontSize: 11,
        },
      }, parsedPageText),
    ),
  );
}

function NativePdfTextReference({ blob, pageNumber, showEmpty, compact }) {
  const [state, setState] = React.useState({ loading: false, text: null, error: null });

  React.useEffect(() => {
    let cancelled = false;
    let loadingTask = null;
    let pdf = null;
    if (!blob || !window.pdfjsLib) {
      setState({ loading: false, text: null, error: null });
      return undefined;
    }
    setState({ loading: true, text: null, error: null });
    (async () => {
      try {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        if (cancelled) return;
        loadingTask = window.pdfjsLib.getDocument({ data: bytes });
        pdf = await loadingTask.promise;
        const page = await pdf.getPage(Math.max(1, Number(pageNumber) || 1));
        const content = await page.getTextContent();
        const positioned = (content.items || []).map(item => {
          const rawText = String(item && item.str || '');
          const transform = Array.isArray(item && item.transform) ? item.transform : [];
          const text = rawText.trim();
          const fontSize = Math.max(
            1,
            Math.abs(Number(transform[3])) || 0,
            Math.abs(Number(transform[0])) || 0,
            Number(item && item.height) || 0,
          );
          return {
            text,
            rawText,
            x: Number.isFinite(Number(transform[4])) ? Number(transform[4]) : null,
            y: Number.isFinite(Number(transform[5])) ? Number(transform[5]) : null,
            width: Number.isFinite(Number(item && item.width)) ? Number(item.width) : 0,
            fontSize,
            hasEOL: Boolean(item && item.hasEOL),
          };
        }).filter(item => item.text);
        const rows = [];
        positioned.forEach(item => {
          const row = item.y == null
            ? null
            : rows.find(candidate => candidate.y != null && Math.abs(candidate.y - item.y) <= Math.max(2.5, item.fontSize * 0.25));
          if (row) {
            row.items.push(item);
          } else {
            rows.push({ y: item.y, items: [item] });
          }
        });

        const joinPdfTextRow = row => {
          const items = row.items.slice().sort((left, right) => {
            if (left.x == null && right.x == null) return 0;
            if (left.x == null) return 1;
            if (right.x == null) return -1;
            return left.x - right.x;
          });
          let output = '';
          let previous = null;
          items.forEach(item => {
            const value = item.text;
            if (!value) return;
            if (output && previous) {
              const previousWidth = previous.width > 0
                ? previous.width
                : Math.max(previous.fontSize * 0.45, previous.fontSize * previous.text.length * 0.45);
              const gap = item.x == null || previous.x == null
                ? 0
                : item.x - (previous.x + previousWidth);
              const spacingThreshold = Math.max(1, Math.min(3.5, Math.max(previous.fontSize, item.fontSize) * 0.22));
              const explicitSpace = /\s$/.test(previous.rawText) || /^\s/.test(item.rawText);
              const punctuationNoSpace = /^[,.;:!?%\)\]\}]/.test(value) || /[([{]$/.test(output);
              if (!punctuationNoSpace && (explicitSpace || gap > spacingThreshold)) output += ' ';
            }
            output += value;
            previous = item;
          });
          return output
            .replace(/\s+([,.;:!?%\)\]\}])/g, '$1')
            .replace(/([([{])\s+/g, '$1')
            .trim();
        };
        const text = rows
          .sort((left, right) => (right.y == null ? -Infinity : right.y) - (left.y == null ? -Infinity : left.y))
          .map(joinPdfTextRow)
          .filter(Boolean)
          .join('\n');
        if (!cancelled) setState({ loading: false, text: text || null, error: null });
        if (pdf) await pdf.destroy();
      } catch (error) {
        if (!cancelled) setState({ loading: false, text: null, error: String(error && error.message || error) });
        if (pdf) {
          try { await pdf.destroy(); } catch {}
        }
      }
    })();
    return () => {
      cancelled = true;
      if (loadingTask) loadingTask.destroy();
    };
  }, [blob, pageNumber]);

  if (!blob || !window.pdfjsLib) return null;
  if (state.loading) {
    return React.createElement('div', {
      style: { marginTop: 12, padding: 12, border: '1px solid var(--nv-border)', borderRadius: 8, background: 'var(--nv-surface)', color: 'var(--nv-text-muted)', fontSize: 12 },
    }, 'Đang đọc text layer của PDF gốc…');
  }
  if (state.text && compact) {
    return React.createElement('details', {
      style: { marginTop: 12, border: '1px solid #86efac', borderRadius: 8, background: '#f0fdf4' },
    },
      React.createElement('summary', { style: { cursor: 'pointer', padding: '10px 12px', color: '#166534', fontSize: 11, fontWeight: 700 } }, 'Đối chiếu text layer PDF gốc'),
      React.createElement('pre', {
        className: 'mono',
        style: { margin: '0 12px 12px', maxHeight: 300, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#111827', background: '#fff', border: '1px solid #bbf7d0', borderRadius: 6, padding: 10, fontSize: 11, lineHeight: 1.5 },
      }, state.text),
    );
  }
  if (state.text) {
    return React.createElement('div', {
      style: { marginTop: 12, padding: 12, border: '1px solid #86efac', borderRadius: 8, background: '#f0fdf4' },
    },
      React.createElement('div', { style: { fontSize: 12, fontWeight: 700, color: '#166534', marginBottom: 6 } }, 'Text native từ PDF gốc · hiển thị bổ sung'),
      React.createElement('div', { style: { fontSize: 11, color: '#166534', marginBottom: 8, lineHeight: 1.4 } }, 'Dùng để giữ lại text layer khi trace bbox không có block native trên màn hình. Đây không phải text đoán từ ảnh scan.'),
      React.createElement('pre', {
        className: 'mono',
        style: { margin: 0, maxHeight: 360, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#111827', background: '#fff', border: '1px solid #bbf7d0', borderRadius: 6, padding: 12, fontSize: 12, lineHeight: 1.55 },
      }, state.text),
    );
  }
  if (!showEmpty) return null;
  return React.createElement('div', {
    style: { marginTop: 12, padding: 12, border: '1px solid var(--nv-border)', borderRadius: 8, background: 'var(--nv-surface)', color: 'var(--nv-text-muted)', fontSize: 11, lineHeight: 1.45 },
  }, state.error
    ? `Không đọc được text layer PDF gốc: ${state.error}`
    : 'PDF gốc không có text layer ở trang này, nên đây có thể là trang scan. Text hiển thị ở trên là kết quả OCR/VLM nếu pipeline đã tạo block.'
  );
}

function PageContentPanel({ pageNumber, items, sourcePdfBlob, hoveredRow, onHoverItem, onOpenVisual }) {
  const allItems = (Array.isArray(items) ? items : []).filter(Boolean);
  const visualItems = allItems.filter(nemoBboxIsVisual);
  const textItems = allItems.filter(item => !nemoBboxIsVisual(item) && String(item.text || '').trim());
  const tableItems = textItems.filter(nemoBlockIsTable);
  const textInVisualItems = textItems.filter(item => !nemoBlockIsTable(item) && nemoTextIsInsideVisual(item, visualItems));
  const nativeItems = textItems.filter(item => nemoBlockIsNative(item) && !nemoBlockIsTable(item) && !textInVisualItems.includes(item));
  const ocrItems = textItems.filter(item => !nemoBlockIsTable(item) && !textInVisualItems.includes(item) && !nativeItems.includes(item));

  function visualForText(item) {
    let best = null;
    let score = 0;
    visualItems.forEach(visual => {
      const textArea = nemoBboxArea(item && item.bbox);
      const overlap = nemoBboxIntersection(item && item.bbox, visual && visual.bbox);
      const value = textArea ? overlap / textArea : 0;
      if (value > score) { score = value; best = visual; }
    });
    return best;
  }

  function itemLabel(item, group) {
    if (group === 'table') return 'Bảng · Markdown';
    if (group === 'native') return 'Văn bản native';
    if (group === 'image-text') return 'Text trong ảnh · OCR/VLM';
    if (String(item && item.ocr_mode || '').startsWith('scan_')) return 'Text OCR · scan';
    return 'Text OCR/VLM';
  }

  function renderTextCard(item, group, index) {
    const relatedVisual = group === 'image-text' ? visualForText(item) : null;
    const key = String(item.key || item.block_id || `${group}-${index}`);
    const active = hoveredRow === key;
    return React.createElement('div', {
      key,
      onMouseEnter: () => onHoverItem && onHoverItem(key),
      onMouseLeave: () => onHoverItem && onHoverItem(null),
      onClick: () => onHoverItem && onHoverItem(key),
      style: {
        padding: 12, marginBottom: 8, borderRadius: 8,
        border: `1px solid ${active ? 'var(--nv-green)' : 'var(--nv-border)'}`,
        background: active ? 'rgba(118,185,0,0.10)' : 'var(--nv-surface)',
        cursor: 'pointer',
      },
    },
      React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 7 } },
        React.createElement('span', { style: { fontSize: 10, fontWeight: 700, color: active ? 'var(--nv-green)' : 'var(--nv-text-muted)', textTransform: 'uppercase', letterSpacing: 0.35 } }, itemLabel(item, group)),
        item.reader_backend && React.createElement('span', { style: { fontSize: 10, color: 'var(--nv-text-muted)' } }, `· ${item.reader_backend}`),
        item.bbox && React.createElement('span', { className: 'mono', style: { fontSize: 9, color: 'var(--nv-text-dim)' } }, `· bbox ${item.bbox.slice(0, 4).map(value => Number(value).toFixed(2)).join(', ')}`),
      ),
      relatedVisual && relatedVisual.image_url && React.createElement('button', {
        type: 'button',
        onClick: event => { event.stopPropagation(); onOpenVisual && onOpenVisual(relatedVisual); },
        style: { display: 'block', width: 'min(280px, 100%)', padding: 4, marginBottom: 8, border: '1px solid var(--nv-border)', borderRadius: 6, background: '#f3f4f6', cursor: 'zoom-in' },
      }, React.createElement('img', { src: relatedVisual.image_url, alt: 'Crop chứa text trong ảnh', loading: 'lazy', style: { display: 'block', width: '100%', maxHeight: 130, objectFit: 'contain' } })),
      React.createElement('pre', {
        style: {
          margin: 0, padding: 10, maxHeight: group === 'table' ? 420 : 300, overflow: 'auto',
          whiteSpace: group === 'table' ? 'pre' : 'pre-wrap', wordBreak: 'break-word',
          color: '#111827', background: '#fff', border: '1px solid #d1d5db', borderRadius: 6,
          fontSize: 12, lineHeight: 1.55,
        },
      }, String(item.text || '').trim()),
    );
  }

  function renderGroup(title, description, group, groupItems) {
    if (!groupItems.length) return null;
    return React.createElement('section', { key: group, style: { marginBottom: 14 } },
      React.createElement('div', { style: { display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 7 } },
        React.createElement('div', { style: { fontSize: 12, fontWeight: 700 } }, `${title} · ${groupItems.length}`),
        React.createElement('div', { style: { fontSize: 10, color: 'var(--nv-text-muted)' } }, description),
      ),
      groupItems.map((item, index) => renderTextCard(item, group, index)),
    );
  }

  const hasText = textItems.length > 0;
  return React.createElement('div', {
    className: 'card',
    style: { marginTop: 14, padding: 14 },
  },
    React.createElement('div', { className: 'card-title', style: { marginBottom: 5 } }, `Nội dung đã parse · trang ${pageNumber}`),
    React.createElement('div', { style: { fontSize: 11, color: 'var(--nv-text-muted)', marginBottom: 12, lineHeight: 1.45 } },
      `${nativeItems.length} text native · ${ocrItems.length} text OCR/VLM · ${textInVisualItems.length} text trong ảnh · ${tableItems.length} bảng · ${visualItems.length} crop visual. Nội dung đọc ở đây, bbox trên ảnh chỉ dùng để đối chiếu vị trí.`,
    ),
    renderGroup('Văn bản native', 'text layer hoặc parser native', 'native', nativeItems),
    renderGroup('Văn bản OCR/VLM', 'text được đọc từ vùng OCR hoặc nguyên trang', 'ocr', ocrItems),
    renderGroup('Text trong ảnh', 'text nằm bên trong crop ảnh, sơ đồ hoặc biểu đồ', 'image-text', textInVisualItems),
    renderGroup('Bảng', 'giữ nguyên Markdown để đọc và copy', 'table', tableItems),
    !hasText && !visualItems.length && React.createElement('div', { className: 'empty-state', style: { padding: 24 } }, 'Trang chưa có text hoặc crop visual hợp lệ.'),
    !hasText && visualItems.length > 0 && React.createElement('div', { style: { padding: 10, border: '1px solid var(--nv-border)', borderRadius: 6, background: 'var(--nv-surface)', color: 'var(--nv-text-muted)', fontSize: 11 } }, 'Trang này chỉ có crop visual. Ảnh được hiển thị ở phần “Ảnh / biểu đồ / sơ đồ”, không tự coi nhãn “hình ảnh” là text.'),
    sourcePdfBlob && React.createElement(NativePdfTextReference, {
      blob: sourcePdfBlob,
      pageNumber,
      showEmpty: !nativeItems.length,
      compact: nativeItems.length > 0,
    }),
  );
}

function VisualCropGallery({ page, onOpenVisual }) {
  const seen = new Set();
  const items = (page && Array.isArray(page.blocks) ? page.blocks : [])
    .filter(item => nemoBboxIsVisual(item) && item.image_url)
    // The sidecar can contain both the exploded ``image`` row and its source
    // ``infographic/chart`` item at exactly the same bbox. Prefer the
    // semantic label and show one crop, not two identical thumbnails.
    .sort((left, right) => {
      const rank = item => item.content_type === 'image' ? 0 : 1;
      return rank(right) - rank(left);
    })
    .filter(item => {
      const bbox = Array.isArray(item.bbox) ? item.bbox.slice(0, 4).map(value => Number(value).toFixed(4)).join(',') : '';
      const key = bbox || item.image_url;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  if (!items.length) return null;
  return React.createElement('div', {
    className: 'card',
    style: { marginTop: 12, padding: 12 },
  },
    React.createElement('div', { className: 'card-title', style: { marginBottom: 4 } }, 'Ảnh / biểu đồ / sơ đồ đã giữ lại'),
    React.createElement('div', {
      style: { color: 'var(--nv-text-muted)', fontSize: 11, marginBottom: 10 },
    }, 'Full-page OCR vẫn giữ chữ trong vùng này nếu có, còn crop bên dưới là bằng chứng hình ảnh riêng.'),
    React.createElement('div', {
      style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10 },
    }, items.map((item, index) => React.createElement('button', {
      key: item.block_id || `visual-crop-${index}`,
      type: 'button',
      onClick: () => onOpenVisual && onOpenVisual(item),
      style: {
        display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0,
        padding: 6, textAlign: 'left', cursor: 'zoom-in',
        border: '1px solid var(--nv-border)', borderRadius: 6,
        background: 'var(--nv-surface)', color: 'var(--nv-text)',
      },
    },
      React.createElement('img', {
        src: item.image_url,
        alt: item.text || item.content_type || 'visual crop',
        loading: 'lazy',
        style: { display: 'block', width: '100%', height: 120, objectFit: 'contain', background: '#f3f4f6', borderRadius: 4 },
      }),
      React.createElement('span', { style: { fontSize: 11, lineHeight: 1.25 } }, item.text || item.content_type || 'visual'),
    ))),
  );
}

function OriginalPagePreview({
  selectedTracePage, displayPage, pageItems, hoveredRow, onHoverItem,
  isSpreadsheetFile, isOfficeDocumentFile, isImageFile, isTextFile, isAudioFile, isVideoFile,
  visualPageImageUrl, sourceFileUrl, sourceText, sourcePdfBlob, sourcePdfUrl,
}) {
  return React.createElement('div', null,
    React.createElement('div', { style: { fontSize: 13, fontWeight: 600, padding: '8px 10px', marginBottom: 8, border: '1px solid var(--nv-border)', borderRadius: 6, background: 'var(--nv-surface)' } }, isSpreadsheetFile ? `Native preview · sheet ${selectedTracePage.page_number}` : isOfficeDocumentFile ? `Preview trang đã chuyển PDF · trang ${selectedTracePage.page_number}` : isImageFile ? `Ảnh gốc · trang ${selectedTracePage.page_number}` : isTextFile ? `Nội dung gốc · trang ${selectedTracePage.page_number}` : isAudioFile ? 'Audio gốc' : isVideoFile ? 'Video gốc' : `Ảnh/PDF gốc · trang ${selectedTracePage.page_number}`),
    isSpreadsheetFile
      ? React.createElement('div', { style: { minHeight: 680, maxHeight: 680, overflow: 'auto', border: '1px solid var(--nv-border)', background: '#fff', padding: 12 } },
          React.createElement('div', { style: { fontSize: 11, color: 'var(--nv-text-muted)', marginBottom: 10 } }, 'Excel/CSV đang được đọc native; vùng dưới đây là Markdown canonical theo sheet/range, không phải ảnh OCR.'),
          (selectedTracePage.blocks || []).map((block, index) => React.createElement('pre', { key: index, style: { whiteSpace: 'pre-wrap', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', color: '#111827', fontSize: 11, lineHeight: 1.45, padding: 10, margin: '0 0 8px', background: '#f7f7f7', border: '1px solid #ddd', borderRadius: 5 } }, `${block.sheet_name || 'CSV'} · ${block.range || 'vùng dữ liệu'}\n\n${block.text || ''}`)),
        )
      : visualPageImageUrl
        ? React.createElement(PdfOverlayView, { imageUrl: visualPageImageUrl, sourceUrl: sourcePdfUrl, pageNumber: displayPage, items: pageItems, hoveredRow, onHoverItem })
        : isImageFile && sourceFileUrl
          ? React.createElement('img', { src: sourceFileUrl, alt: `Ảnh gốc · trang ${displayPage}`, style: { display: 'block', width: '100%', maxHeight: 680, objectFit: 'contain', border: '1px solid var(--nv-border)', background: '#fff' } })
          : isTextFile && sourceText != null
            ? React.createElement('pre', { style: { minHeight: 680, maxHeight: 680, overflow: 'auto', whiteSpace: 'pre-wrap', margin: 0, padding: 16, border: '1px solid var(--nv-border)', background: '#fff', fontSize: 12, lineHeight: 1.5 } }, sourceText)
            : isAudioFile && sourceFileUrl
              ? React.createElement('div', { style: { minHeight: 180, display: 'grid', placeItems: 'center', border: '1px solid var(--nv-border)', background: 'var(--nv-surface)', padding: 24 } }, React.createElement('audio', { controls: true, preload: 'metadata', src: sourceFileUrl, style: { width: '100%' } }))
              : isVideoFile && sourceFileUrl
                ? React.createElement('video', { controls: true, preload: 'metadata', src: sourceFileUrl, style: { display: 'block', width: '100%', maxHeight: 680, background: '#000' } })
                : sourcePdfBlob && window.pdfjsLib
                  ? React.createElement(PdfOverlayView, { blob: sourcePdfBlob, sourceUrl: sourcePdfUrl, pageNumber: displayPage, items: pageItems, hoveredRow, onHoverItem })
                  : sourcePdfUrl
                    ? React.createElement('object', { data: `${sourcePdfUrl}#page=${displayPage}`, type: 'application/pdf', style: { width: '100%', height: 680, border: '1px solid var(--nv-border)', background: '#fff' } },
                        React.createElement('div', { style: { padding: 24, color: 'var(--nv-text-muted)' } }, 'Trình duyệt không render được PDF tại chỗ. ', React.createElement('a', { href: sourcePdfUrl, target: '_blank', rel: 'noreferrer', style: { color: 'var(--nv-green)' } }, 'Mở PDF ở tab mới')))
                    : React.createElement('div', { className: 'empty-state', style: { border: '1px solid var(--nv-border)', padding: 30 } }, isOfficeDocumentFile ? 'Chưa có ảnh trang sau khi chuyển PPT/DOC sang PDF. Hãy bật Visual evidence khi upload để xem preview.' : 'Chưa có artifact gốc trong bộ nhớ trình duyệt. Preview chỉ khả dụng trên trình duyệt đã upload job này.'),
  );
}

function NativeSpreadsheetPageView({ page }) {
  const blocks = (page && Array.isArray(page.blocks)) ? page.blocks : [];
  const reader = blocks.find(block => block && block.reader_backend)?.reader_backend
    || (page && page.reader_backend)
    || 'native parser';
  return React.createElement('div', {
    style: {
      minHeight: 680, maxHeight: 680, overflow: 'auto', padding: 12,
      border: '1px solid var(--nv-border)', borderRadius: 8, background: '#f8fafc',
    },
  },
    React.createElement('div', {
      style: { marginBottom: 10, padding: '8px 10px', border: '1px solid #bbf7d0', borderRadius: 5, background: '#f0fdf4', color: '#166534', fontSize: 11, lineHeight: 1.45 },
    }, `Native sheet · parser ${reader} · không có bbox OCR để dựng PDF giả`),
    blocks.length
      ? blocks.map((block, index) => React.createElement('pre', {
          key: block.block_id || index,
          style: {
            whiteSpace: 'pre-wrap', overflow: 'auto', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            color: '#111827', fontSize: 11, lineHeight: 1.5, padding: 12, margin: '0 0 10px',
            background: '#fff', border: '1px solid #cbd5e1', borderRadius: 6,
          },
        }, `${block.sheet_name || 'CSV'} · ${block.range || 'vùng dữ liệu'}\n\n${block.text || ''}`))
      : React.createElement('div', { className: 'empty-state', style: { color: '#475569', padding: 24 } }, 'Chưa có block native cho sheet này.'),
  );
}

function PipelineOutputPopup({ popup, onClose, onOpenOutput }) {
  if (!popup) return null;
  return React.createElement('div', {
    role: 'dialog',
    'aria-modal': 'true',
    onClick: onClose,
    style: {
      position: 'fixed', inset: 0, zIndex: 1000, padding: 28,
      background: 'rgba(0,0,0,0.78)', display: 'flex', alignItems: 'center', justifyContent: 'center',
    },
  },
    React.createElement('div', {
      onClick: e => e.stopPropagation(),
      className: 'card',
      style: { width: 'min(1000px, 94vw)', maxHeight: '88vh', overflow: 'auto', padding: 18 },
    },
      React.createElement('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 12 } },
        React.createElement('div', null,
          React.createElement('div', { style: { fontSize: 15, fontWeight: 600 } }, popup.title || 'Output pipeline'),
          popup.subtitle && React.createElement('div', { style: { fontSize: 11, color: 'var(--nv-text-muted)', marginTop: 4 } }, popup.subtitle),
        ),
        React.createElement('button', { className: 'btn', onClick: onClose, style: { background: 'var(--nv-surface)', color: 'var(--nv-text)' } }, 'Đóng'),
      ),
      (popup.imageUrl || popup.imageB64) && React.createElement('div', { style: { marginBottom: 14, padding: 8, border: '1px solid var(--nv-border)', borderRadius: 8, background: '#fff', textAlign: 'center' } },
        React.createElement('img', { src: popup.imageUrl || `data:image/png;base64,${popup.imageB64}`, alt: popup.imageAlt || 'Ảnh crop của block', style: { maxWidth: '100%', maxHeight: '66vh', objectFit: 'contain' } }),
      ),
      popup.flow
        ? React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: 8, maxHeight: '72vh', overflowY: 'auto' } },
            popup.flow.map((stage, index) => React.createElement(React.Fragment, { key: stage.id || index },
              React.createElement('div', { style: { padding: 14, border: '1px solid var(--nv-border)', borderRadius: 8, background: 'var(--nv-surface)' } },
                React.createElement('div', { style: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 } },
                  React.createElement('div', null,
                    React.createElement('div', { style: { fontSize: 13, fontWeight: 600 } }, `${index + 1}. ${stage.label}`),
                    React.createElement('div', { style: { marginTop: 5, fontSize: 12, lineHeight: 1.45, color: 'var(--nv-text-muted)' } }, stage.description),
                    React.createElement('div', { className: 'mono', style: { marginTop: 7, fontSize: 10, color: 'var(--nv-text-dim)' } }, stage.executor || [stage.model, stage.function, stage.endpoint].filter(Boolean).join(' · ') || 'Thư viện / operator'),
                  ),
                  pipelineFlowBadge(stage.status),
                ),
                React.createElement('button', { className: 'btn', style: { marginTop: 10, padding: '5px 9px', fontSize: 10, background: 'var(--nv-bg)', color: 'var(--nv-green)' }, onClick: () => onOpenOutput && onOpenOutput({ title: `${stage.label} · output`, subtitle: stage.model || stage.function || 'Output của stage', value: stage.output || { note: 'Bước này không giữ payload riêng.' } }) }, 'Xem output'),
              ),
              index < popup.flow.length - 1 && React.createElement('div', { style: { textAlign: 'center', color: 'var(--nv-green)', fontSize: 18, lineHeight: 1 } }, '↓'),
            )),
          )
        : React.createElement('pre', { className: 'mono', style: { background: 'var(--nv-bg)', border: '1px solid var(--nv-border)', borderRadius: 6, padding: 14, whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: '72vh', fontSize: 11 } },
            typeof popup.value === 'string' ? popup.value : JSON.stringify(popup.value, null, 2)
          ),
    ),
  );
}

function pipelineFlowBadge(state) {
  const cls = { observed: 'badge-green', completed: 'badge-green', not_applicable: 'badge-dim', not_observed: 'badge-dim', configured: 'badge-blue', failed: 'badge-red' }[state] || 'badge-dim';
  const labels = { observed: 'đã chạy', completed: 'hoàn tất', not_applicable: 'không áp dụng', not_observed: 'chưa chạy', configured: 'đã cấu hình', failed: 'lỗi' };
  return React.createElement('span', { className: `badge ${cls}` }, labels[state] || state || '—');
}

function JobDetailView({ jobId, onBack }) {
  const [job, setJob] = React.useState(null);
  const [docs, setDocs] = React.useState([]);
  const [docTotal, setDocTotal] = React.useState(0);
  const [docTotalFiltered, setDocTotalFiltered] = React.useState(0);
  const [docOffset, setDocOffset] = React.useState(0);
  const [docLimit] = React.useState(100);
  const [docStatusFilter, setDocStatusFilter] = React.useState('');
  const [events, setEvents] = React.useState([]);
  const [throughput, setThroughput] = React.useState([]); // {t, completed, failed}
  const [sseStatus, setSseStatus] = React.useState('connecting');
  const [error, setError] = React.useState(null);
  const [jobUnavailable, setJobUnavailable] = React.useState(false);
  const startedAtRef = React.useRef(null);
  const [documentDetail, setDocumentDetail] = React.useState(null);
  const [selectedDocumentId, setSelectedDocumentId] = React.useState(null);
  const [pipelineConfig, setPipelineConfig] = React.useState(null);
  const [clusterOverview, setClusterOverview] = React.useState(null);
  const [sourcePdfUrl, setSourcePdfUrl] = React.useState(null);
  const [sourcePdfBlob, setSourcePdfBlob] = React.useState(null);
  const [sourceFileUrl, setSourceFileUrl] = React.useState(null);
  const [sourceText, setSourceText] = React.useState(null);
  const [selectedRow, setSelectedRow] = React.useState(0);
  const [selectedPage, setSelectedPage] = React.useState(null);
  const [hoveredRow, setHoveredRow] = React.useState(null);
  const [pipelineTrace, setPipelineTrace] = React.useState(null);
  const [pipelineTraceError, setPipelineTraceError] = React.useState(null);
  const [visualEvidence, setVisualEvidence] = React.useState(null);
  const [visualEvidenceError, setVisualEvidenceError] = React.useState(null);
  const [outputPopup, setOutputPopup] = React.useState(null);

  React.useEffect(() => {
    setJobUnavailable(false);
  }, [jobId]);

  // ------------------------------------------------------------------
  // Initial + on-status-change fetch of aggregate and docs.
  // ------------------------------------------------------------------
  const fetchAggregate = React.useCallback(async () => {
    try {
      const resp = await fetch(`/v1/dashboard/api/jobs/${jobId}`);
      if (jobDetailMissingResponse(resp)) {
        setJobUnavailable(true);
        setSseStatus('stopped');
        setError(JOB_DETAIL_MISSING_MESSAGE);
        return;
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setJob(data);
      if (data.started_at && !startedAtRef.current) {
        startedAtRef.current = new Date(data.started_at).getTime();
      }
      setError(null);
    } catch (e) {
      setError(`aggregate: ${e}`);
    }
  }, [jobId]);

  const fetchDocs = React.useCallback(async () => {
    try {
      const params = new URLSearchParams({
        offset: String(docOffset),
        limit: String(docLimit),
      });
      if (docStatusFilter) params.set('status', docStatusFilter);
      const resp = await fetch(`/v1/dashboard/api/jobs/${jobId}/documents?${params}`);
      if (jobDetailMissingResponse(resp)) {
        setJobUnavailable(true);
        setSseStatus('stopped');
        setError(JOB_DETAIL_MISSING_MESSAGE);
        return;
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setDocs(data.items || []);
      setDocTotal(data.total || 0);
      setDocTotalFiltered(data.total_filtered || 0);
      setError(null);
    } catch (e) {
      setError(`documents: ${e}`);
    }
  }, [jobId, docOffset, docLimit, docStatusFilter]);

  React.useEffect(() => { fetchAggregate(); }, [fetchAggregate]);
  React.useEffect(() => { fetchDocs(); }, [fetchDocs]);

  React.useEffect(() => {
    if (!selectedDocumentId && docs.length > 0) setSelectedDocumentId(docs[0].id);
    if (selectedDocumentId && !docs.some(d => d.id === selectedDocumentId)) {
      setSelectedDocumentId(docs.length > 0 ? docs[0].id : null);
    }
  }, [docs, selectedDocumentId]);

  React.useEffect(() => {
    if (!selectedDocumentId) { setDocumentDetail(null); return undefined; }
    let stopped = false;
    let timer = null;
    async function loadDocumentDetail() {
      try {
        const response = await fetch(`/v1/ingest/job/${jobId}/document/${selectedDocumentId}`);
        if (jobDetailMissingResponse(response)) {
          if (!stopped) {
            setJobUnavailable(true);
            setSseStatus('stopped');
            setDocumentDetail(prev => prev ? {
              ...prev,
              status: 'failed',
              job_lost: true,
              error: JOB_DETAIL_MISSING_MESSAGE,
            } : prev);
            setError(JOB_DETAIL_MISSING_MESSAGE);
          }
          return;
        }
        if (!response.ok && response.status !== 202) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!stopped) {
          setDocumentDetail(prev => {
            if (!prev || prev.status !== data.status || prev.result_data !== data.result_data) setSelectedRow(0);
            return data;
          });
          if (!['completed', 'failed'].includes(data.status)) timer = setTimeout(loadDocumentDetail, 2000);
        }
      } catch (e) {
        if (!stopped) {
          setError(`result data: ${e}`);
          timer = setTimeout(loadDocumentDetail, 3000);
        }
      }
    }
    loadDocumentDetail();
    return () => { stopped = true; if (timer) clearTimeout(timer); };
  }, [jobId, selectedDocumentId]);

  React.useEffect(() => {
    if (!selectedDocumentId) { setPipelineTrace(null); return undefined; }
    let stopped = false;
    async function loadPipelineTrace() {
      try {
        const response = await fetch(`/v1/dashboard/api/jobs/${jobId}/documents/${selectedDocumentId}/pipeline`);
        if (jobDetailMissingResponse(response)) {
          if (!stopped) {
            setJobUnavailable(true);
            setSseStatus('stopped');
            setPipelineTrace(null);
            setPipelineTraceError(JOB_DETAIL_MISSING_MESSAGE);
            setError(JOB_DETAIL_MISSING_MESSAGE);
          }
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!stopped) {
          setPipelineTrace(data);
          setPipelineTraceError(null);
          if (data.pages && data.pages.length > 0) {
            setSelectedPage(prev => data.pages.some(page => page.page_number === prev) ? prev : data.pages[0].page_number);
          }
        }
      } catch (e) {
        if (!stopped) setPipelineTraceError(String(e));
      }
    }
    loadPipelineTrace();
    return () => { stopped = true; };
  }, [jobId, selectedDocumentId, documentDetail && documentDetail.status, documentDetail && documentDetail.result_data && documentDetail.result_data.length]);

  React.useEffect(() => {
    if (!selectedDocumentId) {
      setVisualEvidence(null);
      setVisualEvidenceError(null);
      return undefined;
    }
    let stopped = false;
    async function loadVisualEvidence() {
      try {
        const response = await fetch(`/v1/dashboard/api/jobs/${jobId}/documents/${selectedDocumentId}/visual`);
        if (jobDetailMissingResponse(response)) {
          if (!stopped) {
            setJobUnavailable(true);
            setSseStatus('stopped');
            setVisualEvidence(null);
            setVisualEvidenceError(JOB_DETAIL_MISSING_MESSAGE);
            setError(JOB_DETAIL_MISSING_MESSAGE);
          }
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!stopped) {
          setVisualEvidence(data.available ? data : null);
          setVisualEvidenceError(null);
          if (data.pages && data.pages.length > 0) {
            setSelectedPage(prev => data.pages.some(page => page.page_number === prev) ? prev : data.pages[0].page_number);
          }
        }
      } catch (e) {
        if (!stopped) setVisualEvidenceError(String(e));
      }
    }
    loadVisualEvidence();
    return () => { stopped = true; };
  }, [jobId, selectedDocumentId, documentDetail && documentDetail.status]);

  React.useEffect(() => {
    let stopped = false;
    Promise.all([
      fetch('/v1/ingest/pipeline-config').then(r => { if (!r.ok) throw new Error(`pipeline config HTTP ${r.status}`); return r.json(); }),
      fetch('/v1/dashboard/api/overview').then(r => { if (!r.ok) throw new Error(`overview HTTP ${r.status}`); return r.json(); }),
    ]).then(([config, overview]) => {
      if (!stopped) { setPipelineConfig(config); setClusterOverview(overview); }
    }).catch(e => { if (!stopped) setError(`pipeline introspection: ${e}`); });
    return () => { stopped = true; };
  }, []);

  React.useEffect(() => {
    let stopped = false;
    let objectUrl = null;
    if (!window.NemoDebugStore) return undefined;
    setSourcePdfUrl(null);
    setSourcePdfBlob(null);
    setSourceFileUrl(null);
    setSourceText(null);
    window.NemoDebugStore.load(jobId).then(record => {
      if (stopped || !record || !record.file) return;
      // The debug store contains the original upload. Office files are
      // converted to PDF inside the worker, so their original PPTX/DOCX bytes
      // are never relabeled as application/pdf. Other formats get a typed
      // object URL for their native preview below.
      const filename = String(record.file.name || record.filename || '').toLowerCase();
      const extension = filename.includes('.') ? filename.split('.').pop() : '';
      const sourceType = record.file.type || record.contentType || 'application/octet-stream';
      const isPdf = filename ? extension === 'pdf' : sourceType === 'application/pdf';
      const isText = ['txt', 'md', 'json', 'sh', 'html'].includes(extension);
      let source = record.file instanceof Blob
        ? record.file
        : new Blob([record.file], { type: sourceType });
      if (isPdf && source.type !== 'application/pdf') {
        source = source.slice(0, source.size, 'application/pdf');
      }
      objectUrl = URL.createObjectURL(source);
      if (isPdf) setSourcePdfUrl(objectUrl);
      else setSourceFileUrl(objectUrl);
      if (isText) {
        source.text().then(value => { if (!stopped) setSourceText(value); }).catch(() => {});
      }
      if (!isPdf) return;
      setSourcePdfBlob(source);
    }).catch(() => {});
    return () => {
      stopped = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setSourcePdfUrl(null);
      setSourcePdfBlob(null);
      setSourceFileUrl(null);
      setSourceText(null);
    };
  }, [jobId]);

  // ------------------------------------------------------------------
  // Live per-job SSE — subscribes to the J4 per-job stream.
  // We slot incoming events into both the recent-events log and the
  // throughput series; aggregate-level events refresh the header.
  // ------------------------------------------------------------------
  React.useEffect(() => {
    if (jobUnavailable) return undefined;
    let es = null;
    let retryTimer = null;
    let stopped = false;

    function connect() {
      if (stopped || jobUnavailable) return;
      setSseStatus('connecting');
      es = new EventSource(`/v1/ingest/job/${jobId}/events`);

      const pushEvent = (kind, payload) => {
        setEvents(prev => [
          { id: `${kind}:${Date.now()}:${Math.random()}`, kind, payload, t: Date.now() },
          ...prev,
        ].slice(0, 200));
      };

      // Generic catch-all: we don't know exactly which event names the
      // backend emits, so we listen for the SSE event types we know
      // about ("completed", "failed", "job_progress", "job_finalized",
      // "job_partial", "job_failed", "job_started", "job_created").
      // The "message" handler covers anything else.
      const handlerFor = (kind) => (e) => {
        try {
          const data = JSON.parse(e.data);
          pushEvent(kind, data);

          if (['completed', 'failed', 'processing', 'pending'].includes(kind)) {
            // Update the in-row status optimistically.
            setDocs(prev => prev.map(d =>
              d.id === data.id || d.id === data.document_id
                ? {
                    ...d,
                    status: kind,
                    error: data.error,
                    elapsed_s: data.elapsed_s,
                    result_rows: data.result_rows,
                    pipeline_diagnostics: data.pipeline_diagnostics || d.pipeline_diagnostics,
                  }
                : d
            ));
            setThroughput(prev => {
              const tail = prev[prev.length - 1] || { completed: 0, failed: 0 };
              const next = {
                t: Date.now(),
                completed: tail.completed + (kind === 'completed' ? 1 : 0),
                failed: tail.failed + (kind === 'failed' ? 1 : 0),
              };
              return [...prev.slice(-119), next];
            });
            setSseStatus('connected');
          }

          if (kind.startsWith('job_')) {
            // Job lifecycle: refresh header from the event payload.
            setJob(prev => prev ? {
              ...prev,
              status: data.status || prev.status,
              counts: data.counts || prev.counts,
              expected_documents: data.expected_documents != null ? data.expected_documents : prev.expected_documents,
              started_at: data.started_at || prev.started_at,
              finalized_at: data.finalized_at || prev.finalized_at,
              elapsed_s: data.elapsed_s != null ? data.elapsed_s : prev.elapsed_s,
              trace_id: data.trace_id != null ? data.trace_id : prev.trace_id,
            } : prev);
            setSseStatus('connected');
          }
          if (['job_finalized', 'job_partial', 'job_failed'].includes(kind)) {
            stopped = true;
            setSseStatus('closed');
            if (es) es.close();
          }
        } catch {}
      };

      const eventTypes = [
        'completed', 'failed', 'processing', 'pending',
        'job_created', 'job_started', 'job_progress',
        'job_finalized', 'job_partial', 'job_failed',
      ];
      for (const t of eventTypes) es.addEventListener(t, handlerFor(t));

      es.onopen = () => setSseStatus('connected');
      es.onerror = () => {
        setSseStatus('disconnected');
        es.close();
        if (!stopped && !jobUnavailable) retryTimer = setTimeout(connect, 3000);
      };
    }

    connect();
    return () => {
      stopped = true;
      if (es) es.close();
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, [jobId, jobUnavailable]);

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function statusBadge(status) {
    const cls = {
      completed: 'badge-green',
      failed: 'badge-red',
      partial_success: 'badge-yellow',
      processing: 'badge-yellow',
      running: 'badge-yellow',
      pending: 'badge-blue',
    }[status] || 'badge-dim';
    return React.createElement('span', { className: `badge ${cls}` }, statusLabel(status));
  }

  function statusLabel(status) {
    return {
      completed: 'hoàn tất', failed: 'lỗi', partial_success: 'thành công một phần',
      processing: 'đang xử lý', running: 'đang chạy', pending: 'đang chờ',
    }[status] || status || '—';
  }

  function fmtTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString();
  }

  function progressPct() {
    if (!job || !job.expected_documents) return 0;
    const c = (job.counts && job.counts.completed) || 0;
    const f = (job.counts && job.counts.failed) || 0;
    return Math.min(100, Math.round(((c + f) / job.expected_documents) * 100));
  }

  function classifyFilename(filename) {
    const name = String(filename || '').toLowerCase();
    const ext = name.includes('.') ? name.split('.').pop() : '';
    const labels = {
      pdf: 'PDF · tài liệu PDF · application/pdf',
      docx: 'DOCX · Microsoft Word', pptx: 'PPTX · Microsoft PowerPoint',
      txt: 'TXT · văn bản thuần · text/plain', md: 'MD · văn bản thuần · text/plain',
      json: 'JSON · văn bản thuần · text/plain', sh: 'SH · văn bản thuần · text/plain',
      html: 'HTML · trang HTML · text/html', htm: 'HTML · trang HTML · text/html',
      xlsx: 'XLSX · workbook Excel · native cell parser',
      xls: 'XLS · workbook Excel cũ · native parser + LibreOffice fallback',
      csv: 'CSV · bảng phân cách · Python csv',
      jpg: 'JPG · hình ảnh · image/jpeg', jpeg: 'JPEG · hình ảnh · image/jpeg',
      png: 'PNG · hình ảnh · image/png', bmp: 'BMP · hình ảnh · image/bmp',
      tiff: 'TIFF · hình ảnh · image/tiff', tif: 'TIFF · hình ảnh · image/tiff',
      svg: 'SVG · hình ảnh · image/svg+xml',
      mp3: 'MP3 · âm thanh', wav: 'WAV · âm thanh', m4a: 'M4A · âm thanh',
      mp4: 'MP4 · video', mov: 'MOV · video', mkv: 'MKV · video', avi: 'AVI · video',
    };
    return labels[ext] || (ext ? `${ext.toUpperCase()} · định dạng theo phần mở rộng` : 'chưa xác định');
  }

  function isNativeReader(reader) {
    return ['native_pdf', 'native_spreadsheet', 'openpyxl', 'python_csv'].includes(String(reader || ''));
  }

  function inspectDocument(docId) {
    setSelectedDocumentId(docId);
    window.setTimeout(() => {
      const target = document.getElementById('document-inspection');
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 50);
  }

  async function deleteCurrentJob() {
    const confirmed = window.confirm(
      `Xóa job ${jobId}?\n\nThao tác này xóa lịch sử job, bản ghi tài liệu, result_data đã giữ lại và bản PDF trong trình duyệt. Các row trong VectorDB vẫn giữ nguyên vì backend hiện chưa có khóa an toàn để map chúng về job.`
    );
    if (!confirmed) return;
    try {
      const response = await fetch(`/v1/dashboard/api/jobs/${jobId}`, { method: 'DELETE' });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { detail = (await response.json()).detail || detail; } catch {}
        throw new Error(detail);
      }
      if (window.NemoDebugStore) await window.NemoDebugStore.remove(jobId).catch(() => {});
      if (onBack) onBack();
      else window.location.hash = 'jobs';
    } catch (e) {
      setError(`Xóa thất bại: ${e.message || e}`);
    }
  }

  function rowText(row) {
    if (Array.isArray(row)) return row.map(item => rowText(item)).filter(Boolean).join('\n\n');
    if (!row || typeof row !== 'object') return String(row || '');
    const metadata = rowMetadata(row);
    const direct = row.text || row.content || row.markdown || metadata.content || '';
    if (direct) return String(direct);
    for (const key of ['table', 'tables', 'chart', 'charts', 'infographic', 'infographics', 'images', 'page_elements', 'page_elements_v3']) {
      if (!Array.isArray(row[key])) continue;
      const nested = row[key].map(item => rowText(item)).filter(Boolean);
      if (nested.length) return nested.join('\n\n');
    }
    return '';
  }

  function rowMetadata(row) {
    if (!row || typeof row !== 'object') return {};
    if (row.metadata && typeof row.metadata === 'object') return row.metadata;
    if (typeof row.metadata === 'string') {
      try {
        const parsed = JSON.parse(row.metadata);
        if (parsed && typeof parsed === 'object') return parsed;
      } catch {}
    }
    return row;
  }

  function rowPage(row, fallback = 1) {
    if (!row || typeof row !== 'object') return fallback;
    const metadata = rowMetadata(row);
    const nested = metadata.content_metadata && typeof metadata.content_metadata === 'object'
      ? metadata.content_metadata : {};
    return Number(row.page_number || metadata.page_number || nested.page_number || fallback) || fallback;
  }

  function readerLabel(row) {
    const metadata = rowMetadata(row);
    const nested = metadata.content_metadata && typeof metadata.content_metadata === 'object' ? metadata.content_metadata : {};
    const backend = row && row._reader_backend || metadata.reader_backend || nested.reader_backend;
    if (backend === 'native_pdf') return 'native PDF · pypdfium2';
    if (['native_spreadsheet', 'openpyxl'].includes(backend)) return 'native spreadsheet · openpyxl';
    if (backend === 'python_csv') return 'native CSV · Python csv';
    if (backend === 'ocr') return 'OCR · model';
    return 'chưa xác định';
  }

  function normalizedBbox(value) {
    if (typeof value === 'string') {
      try { value = JSON.parse(value); } catch { return null; }
    }
    if (!Array.isArray(value) || value.length < 4) return null;
    const box = value.slice(0, 4).map(Number);
    if (box.some(v => !Number.isFinite(v))) return null;
    // The pipeline's public bbox contract is xyxy normalized to [0, 1].
    if (box.some(v => v < 0 || v > 1)) return null;
    return [Math.min(box[0], box[2]), Math.min(box[1], box[3]), Math.max(box[0], box[2]), Math.max(box[1], box[3])];
  }

  function directBbox(value) {
    if (!value || typeof value !== 'object') return null;
    for (const key of ['_bbox_xyxy_norm', 'bbox_xyxy_norm']) {
      const found = normalizedBbox(value[key]);
      if (found) return found;
    }
    const metadata = rowMetadata(value);
    if (metadata && metadata !== value) {
      for (const key of ['_bbox_xyxy_norm', 'bbox_xyxy_norm']) {
        const found = normalizedBbox(metadata[key]);
        if (found) return found;
      }
    }
    if (value.content_metadata && typeof value.content_metadata === 'object') {
      for (const key of ['_bbox_xyxy_norm', 'bbox_xyxy_norm']) {
        const found = normalizedBbox(value.content_metadata[key]);
        if (found) return found;
      }
    }
    return null;
  }

  function directText(value) {
    if (!value || typeof value !== 'object') return '';
    const metadata = rowMetadata(value);
    return String(value.text || value.content || value.markdown || metadata.content || '');
  }

  function collectVisualItems(row, rowIndex) {
    const items = [];
    const parentPage = rowPage(row);
    const nestedKeys = ['table', 'tables', 'chart', 'charts', 'infographic', 'infographics', 'images', 'page_elements', 'page_elements_v3'];
    const walk = (value, key, fallbackPage) => {
      if (Array.isArray(value)) {
        value.forEach((child, childIndex) => walk(child, `${key}[${childIndex}]`, fallbackPage));
        return;
      }
      if (!value || typeof value !== 'object') return;
      const page = rowPage(value, fallbackPage);
      const text = directText(value);
      const bbox = directBbox(value);
      if (text || bbox) items.push({ row: value, rowIndex, key, text, page, bbox });
      for (const nestedKey of nestedKeys) {
        if (!Array.isArray(value[nestedKey])) continue;
        value[nestedKey].forEach((child, childIndex) => walk(child, `${key}.${nestedKey}[${childIndex}]`, page));
      }
    };
    // The embedding pipeline already emits one canonical row per page-text
    // or structured element. Those rows still carry the original page arrays
    // for downstream compatibility; walking them again would duplicate every
    // block in the debug view. Recurse only for an un-exploded page record.
    const metadata = rowMetadata(row);
    const isCanonicalRow = Boolean(
      row && (row._content_type || row._embed_modality || metadata._content_type || metadata._embed_modality)
    );
    if (isCanonicalRow) {
      const text = directText(row);
      const bbox = directBbox(row);
      if (text || bbox) items.push({ row, rowIndex, key: String(rowIndex), text, page: parentPage, bbox });
    } else {
      walk(row, String(rowIndex), parentPage);
    }
    return items;
  }

  function pageExtractionSignal(row) {
    const metadata = rowMetadata(row);
    const hasText = metadata.has_text === true || metadata.has_text === 'true' || Boolean(rowText(row).trim());
    const needsOcr = metadata.needs_ocr_for_text === true || metadata.needs_ocr_for_text === 'true' || metadata.needs_ocr === true;
    return { hasText, needsOcr };
  }

  function findVector(value, path = '') {
    if (Array.isArray(value) && value.length > 0 && value.every(v => typeof v === 'number')) {
      return { path, vector: value };
    }
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) {
        const found = findVector(value[i], `${path}[${i}]`);
        if (found) return found;
      }
    } else if (value && typeof value === 'object') {
      for (const [key, nested] of Object.entries(value)) {
        if (key.toLowerCase().includes('embedding') || ['vector', 'vectors'].includes(key.toLowerCase())) {
          if (Array.isArray(nested) && nested.every(v => typeof v === 'number')) {
            return { path: path ? `${path}.${key}` : key, vector: nested };
          }
        }
        const found = findVector(nested, path ? `${path}.${key}` : key);
        if (found) return found;
      }
    }
    return null;
  }

  function configuredPipeline(poolName) {
    const pipelines = pipelineConfig && pipelineConfig.pipelines;
    return (pipelines && (pipelines[poolName] || pipelines.batch || pipelines.realtime)) || null;
  }

  function pipelineTimeline() {
    const selected = docs.find(d => d.id === selectedDocumentId) || documentDetail || {};
    const status = selected.status || (job && job.status) || 'pending';
    const terminal = ['completed', 'failed'].includes(status);
    const rows = (documentDetail && documentDetail.result_data) || [];
    const batch = configuredPipeline('batch');
    const extractParams = batch && batch.extract_params ? batch.extract_params : {};
    const embedEnabled = Boolean(batch && batch.embed_enabled);
    const selectedMetadata = selected && selected.metadata && typeof selected.metadata === 'object'
      ? selected.metadata
      : {};
    const selectedOcr = selected && selected.ocr && typeof selected.ocr === 'object'
      ? selected.ocr
      : {};
    const documentDiagnostics = documentDetail && documentDetail.pipeline_diagnostics
      && typeof documentDetail.pipeline_diagnostics === 'object'
      ? documentDetail.pipeline_diagnostics
      : {};
    const selectedOcrPipeline = selectedMetadata.ocr_pipeline
      || selectedOcr.pipeline
      || documentDiagnostics.ocr_pipeline
      || documentDiagnostics.pipeline_selector
      || extractParams.ocr_pipeline;
    const option6Semantic = selectedOcrPipeline === 'pipeline-option6';
    const option2Paddle = !option6Semantic && (
      ['pipeline-ppocrv6', 'pipeline-tesseract'].includes(selectedOcrPipeline)
      || Boolean(extractParams.vintern_ocr_invoke_url && extractParams.ocr_invoke_url)
    );
    const option5Semantic = selectedOcrPipeline === 'pipeline-option5';
    const option7Semantic = selectedOcrPipeline === 'pipeline-option7';
    const semanticLayout = option2Paddle || option5Semantic || option6Semantic || option7Semantic;
    const vectors = rows.filter(row => findVector(row)).length;
    const vdb = clusterOverview && clusterOverview.vectordb;
    const filename = selected.filename || '';
    const typeSummary = classifyFilename(filename);
    const nativePages = new Set(rows.filter(row => {
      const signal = pageExtractionSignal(row);
      return signal.hasText && !signal.needsOcr;
    }).map(rowPage)).size;
    const ocrPages = option2Paddle
      ? new Set(rows.filter(row => {
        const metadata = row && row.metadata && typeof row.metadata === 'object' ? row.metadata : {};
        return metadata.ocr_source === 'option2_page_detect_language_router'
          || (row && row.ocr && row.ocr.source === 'option2_page_detect_language_router');
      }).map(rowPage)).size
      : option7Semantic
      ? new Set(rows.filter(row => {
        const metadata = row && row.metadata && typeof row.metadata === 'object' ? row.metadata : {};
        const metadataTiming = metadata.ocr_timing && typeof metadata.ocr_timing === 'object' ? metadata.ocr_timing : {};
        const ocrTiming = row && row.ocr && row.ocr.timing && typeof row.ocr.timing === 'object' ? row.ocr.timing : {};
        const route = metadataTiming.route || ocrTiming.route;
        return metadata.ocr_pipeline === 'pipeline-option7'
          || route === 'native_visual_crops'
          || route === 'scan_full_page';
      }).map(rowPage)).size
      : new Set(rows.filter(row => pageExtractionSignal(row).needsOcr).map(rowPage)).size;
    const pageModel = option7Semantic
      ? 'NIM Page Elements v3 · semantic text/title/table bbox + visual evidence'
      : option6Semantic
      ? 'NIM Page Elements v3 · semantic bbox · batch 128'
      : option2Paddle
      ? 'NIM Page Elements v3 · page/block detection'
      : semanticLayout
        ? 'NIM Page Elements v3 · semantic bbox'
      : (extractParams.use_page_elements ? 'NIM Nemotron Page Elements v3' : 'tắt');
    const tableModel = option7Semantic
      ? 'Page Elements table bbox → Ministral whole-table Markdown · Table Structure tắt'
      : option6Semantic
      ? 'Table Structure NIM tắt · Qwen whole-table Markdown'
      : option2Paddle
      ? 'NIM Table Structure v1 · tách cell'
      : semanticLayout
        ? 'NIM Table Structure v1 · tách cell'
      : (extractParams.use_table_structure ? 'NIM Nemotron Table Structure v1' : 'tắt');
    const ocrModel = option7Semantic
      ? 'Ministral 3 3B FP8 · semantic text/title/table crop + full-page fallback'
      : option6Semantic
      ? 'Qwen3.5-2B VLM · model selected by OPTION6_MODEL · text + Markdown table · max 25'
      : option2Paddle
      ? 'Tesseract probe → Việt: Vintern · Anh/không chắc: Nemotron OCR'
      : option5Semantic
        ? 'Nemotron probe → Việt: VietOCR · Anh/không chắc: Nemotron OCR'
      : 'PP-OCRv6 medium det + medium rec';
    const embedModel = (batch && batch.embed_params && (batch.embed_params.embed_model_name || batch.embed_params.model_name)) || 'nvidia/llama-nemotron-embed-vl-1b-v2';
    return [
      { name: '1. Tiếp nhận tệp', state: selectedDocumentId ? 'completed' : 'pending', detail: `${filename || 'Tệp của document này'} đã được Job Tracker ghi nhận; kết quả được cấu hình giữ lại để kiểm tra.` },
      { name: '2. Phân loại định dạng', state: selectedDocumentId ? 'completed' : 'pending', detail: `FileClassifier: ${typeSummary}. Backend phân loại bằng suffix tên file, không đoán từ nội dung binary; file không nằm trong allow-list sẽ bị HTTP 415.` },
      { name: '3. Định tuyến worker', state: status === 'pending' ? 'pending' : 'completed', detail: 'Endpoint /whole đưa toàn bộ tài liệu vào Batch pool; đây là route explicit whole-document, không tự chọn realtime/batch.' },
      { name: '4. Xác định native PDF hay scan', state: status === 'failed' ? 'failed' : terminal ? 'completed' : 'processing', detail: option6Semantic
        ? `PDFium giữ nguyên text native (${nativePages} trang); block thiếu text mới đi qua Page Elements → Qwen3.5, bảng đi nguyên crop để trả Markdown, vùng ảnh/sơ đồ chỉ giữ crop ngắn. method=${extractParams.method || '—'}.`
        : option7Semantic
        ? `PDFium đọc text native (${nativePages} trang); Page Elements tạo text/title/table bbox và visual evidence → Ministral FP8 OCR semantic crop/whole-table. Table Structure tắt. Scan/layout yếu dùng full-page fallback; không gửi visual crop; method=${extractParams.method || '—'}.`
        : option5Semantic
        ? `PDFium kiểm tra text layer (${nativePages} trang native); trang scan đi qua Page Elements → Table Structure → probe document → route Nemotron/VietOCR. method=${extractParams.method || '—'}.`
        : option2Paddle
        ? `PDFium vẫn kiểm tra text layer (${nativePages} trang có text native); scan đi qua Page Elements → Table Structure → probe Tesseract vie+eng → Việt dùng Vintern, Anh/không chắc dùng Nemotron. method=${extractParams.method || '—'}.`
        : `PDFium mở từng trang và kiểm tra text layer: ${nativePages} trang có text native, ${ocrPages} trang cần OCR theo metadata result. method=${extractParams.method || '—'}, text extraction=${extractParams.extract_text ? 'bật' : 'tắt'}.` },
      { name: '5. Nhận diện nội dung', state: status === 'failed' ? 'failed' : terminal ? 'completed' : 'processing', detail: `Model/operator đang cấu hình: ${pageModel}; ${tableModel}; ${ocrPages > 0 ? ocrModel : `${ocrModel} (chưa có trang OCR trong result)`}. Kết quả hiện có ${rows.length || selected.result_rows || 0} row.` },
      { name: '6. Tạo embedding', state: status === 'failed' ? 'failed' : terminal ? 'completed' : 'processing', detail: embedEnabled ? `Model ${embedModel}; bước embedding đang bật; retained result có ${vectors} row chứa vector${vectors ? '' : ' (chưa thấy vector trong payload)'}.` : 'Bước embedding đang tắt trong pipeline đang chạy.' },
      { name: '7. Ghi vào VectorDB', state: status === 'failed' ? 'unknown' : terminal ? 'completed' : 'processing', detail: vdb && vdb.status === 'ok' ? `VectorDB đang khỏe; bảng ${vdb.table || '—'} có tổng ${vdb.total_rows || 0} row (số liệu toàn bảng, không phải riêng job này).` : 'Chưa lấy được trạng thái VectorDB.' },
      { name: '8. Giữ result_data', state: documentDetail && documentDetail.result_data ? 'completed' : terminal ? 'empty' : 'pending', detail: documentDetail && documentDetail.result_data ? `Có ${rows.length} row kết quả để đối chiếu text, metadata và embedding.` : 'Đang chờ kết quả cuối cùng được giữ lại.' },
    ];
  }

  function pipelineBadge(state) {
    const cls = { completed: 'badge-green', processing: 'badge-yellow', pending: 'badge-blue', failed: 'badge-red', unknown: 'badge-dim', empty: 'badge-dim' }[state] || 'badge-dim';
    const labels = { completed: 'hoàn tất', processing: 'đang xử lý', pending: 'đang chờ', failed: 'lỗi', unknown: 'chưa rõ', empty: 'trống' };
    return React.createElement('span', { className: `badge ${cls}` }, labels[state] || state);
  }

  const resultRows = (documentDetail && documentDetail.result_data) || [];
  const activeRow = resultRows[selectedRow] || null;
  const activePage = rowPage(activeRow);
  const activeVector = findVector(activeRow);
  const codeStyle = {
    background: 'var(--nv-bg)', border: '1px solid var(--nv-border)', borderRadius: 6,
    padding: 12, overflow: 'auto', maxHeight: 320, whiteSpace: 'pre-wrap', fontSize: 11,
  };

  // ------------------------------------------------------------------
  // Charts (pure-SVG so we don't depend on any chart library).
  // 'series' is an array of {t, completed, failed}; we plot cumulative
  // completed/failed counts as two line series within a 600x140 canvas.
  // ------------------------------------------------------------------
  function ThroughputChart({ series }) {
    const W = 600, H = 140, P = 24;
    if (series.length < 2) {
      return React.createElement('div', { className: 'empty-state', style: { padding: 24 } },
        'Đang chờ sự kiện…'
      );
    }
    const tMin = series[0].t;
    const tMax = series[series.length - 1].t;
    const dt = Math.max(1, tMax - tMin);
    const maxY = Math.max(1, ...series.map(s => s.completed + s.failed));
    const x = (t) => P + ((t - tMin) / dt) * (W - 2 * P);
    const y = (v) => H - P - (v / maxY) * (H - 2 * P);
    const pathFor = (key, color) => {
      const d = series.map((s, i) =>
        `${i === 0 ? 'M' : 'L'} ${x(s.t).toFixed(1)} ${y(s[key]).toFixed(1)}`
      ).join(' ');
      return React.createElement('path', {
        d, fill: 'none', stroke: color, strokeWidth: 2,
      });
    };
    return React.createElement('svg', { width: W, height: H, style: { display: 'block' } },
      React.createElement('rect', {
        x: 0, y: 0, width: W, height: H, fill: 'var(--nv-surface)',
      }),
      pathFor('completed', 'var(--nv-green)'),
      pathFor('failed', 'var(--nv-red)'),
      React.createElement('text', {
        x: P, y: H - 4, fontSize: 10, fill: 'var(--nv-text-muted)',
      }, new Date(tMin).toLocaleTimeString()),
      React.createElement('text', {
        x: W - P - 60, y: H - 4, fontSize: 10, fill: 'var(--nv-text-muted)',
      }, new Date(tMax).toLocaleTimeString()),
      React.createElement('text', {
        x: P, y: P, fontSize: 11, fill: 'var(--nv-green)', fontWeight: 600,
      }, `Hoàn tất: ${series[series.length - 1].completed}`),
      React.createElement('text', {
        x: P + 130, y: P, fontSize: 11, fill: 'var(--nv-red)', fontWeight: 600,
      }, `Lỗi: ${series[series.length - 1].failed}`),
    );
  }

  const pct = progressPct();
  const c = job ? ((job.counts && job.counts.completed) || 0) : 0;
  const f = job ? ((job.counts && job.counts.failed) || 0) : 0;
  const p = job ? ((job.counts && job.counts.processing) || 0) : 0;
  const exp = job ? (job.expected_documents || 0) : 0;
  const timeline = pipelineTimeline();
  const visualItems = resultRows.flatMap((row, index) => collectVisualItems(row, index));
  const backendTracePages = (pipelineTrace && pipelineTrace.pages) || [];
  const visualPages = (visualEvidence && visualEvidence.pages) || [];
  const sourceExtension = String(
    (pipelineTrace && pipelineTrace.file && pipelineTrace.file.extension)
      || (documentDetail && documentDetail.filename && documentDetail.filename.split('.').pop())
      || '',
  ).toLowerCase().replace(/^\./, '');
  const isSpreadsheetFile = ['xlsx', 'xls', 'csv'].includes(sourceExtension);
  const visualPageAsTracePage = page => {
    const existing = backendTracePages.find(item => item.page_number === page.page_number) || {};
    const blocks = (page.blocks || []).map((block, index) => ({
      block_id: block.id || `visual-${page.page_number}-${index}`,
      trace_source: 'visual_evidence',
      row_index: block.row_index,
      reading_order: block.reading_order || index + 1,
      content_type: isSpreadsheetFile ? 'spreadsheet_table' : (block.content_type || 'text'),
      reader_backend: block.reader_backend || (isSpreadsheetFile ? (sourceExtension === 'csv' ? 'python_csv' : 'openpyxl') : null),
      page_number: page.page_number,
      bbox: block.bbox,
      model_bbox: block.model_bbox || block.bbox,
      processed_bbox: block.processed_bbox || null,
      crop_bbox: block.crop_bbox || block.bbox,
      text: block.text || '',
      origin: block.origin || null,
      content_origin: block.content_origin || null,
      source: block.source || null,
      ocr_source: block.ocr_source || null,
      ocr_mode: block.ocr_mode || null,
      ocr_pipeline_name: block.ocr_pipeline_name || null,
      pipeline_name: block.pipeline_name || null,
      line_detector_score: block.line_detector_score ?? null,
      provenance: block.provenance || null,
      selected_backend: block.selected_backend || null,
      label_name: block.label_name || null,
      route: block.route || null,
      fallback_reason: block.fallback_reason || null,
      bbox_source: block.bbox_source || null,
      nemotron_original_text: block.nemotron_original_text || null,
      vietnamese_candidate_text: block.vietnamese_candidate_text || null,
      page_elements_score: block.page_elements_score ?? null,
      region_label: block.region_label || null,
      models: block.ocr_mode === 'page_elements_box'
        ? [
            { kind: 'detector', name: 'NIM Page Elements v3 · bbox gốc', function: 'page_elements_detect' },
            {
              kind: 'ocr',
              name: String(block.ocr_source) === 'tesseract-5' ? 'Tesseract 5 · đọc bbox Page Elements' : `OCR · ${block.ocr_source || 'recognizer'}`,
              function: 'page_elements_region_ocr',
            },
          ]
        : block.ocr_mode === 'table_cell'
          ? [
              { kind: 'structure', name: 'NIM Table Structure · cell bbox', function: 'table_structure_cells' },
              { kind: 'ocr', name: String(block.ocr_source) === 'tesseract-5' ? 'Tesseract 5 · đọc cell' : `OCR · ${block.ocr_source || 'recognizer'}`, function: 'ocr_cell_recognize' },
            ]
        : ['scan_full_page', 'scan_tile'].includes(String(block.ocr_mode || ''))
          ? [
              { kind: 'ocr', name: `Tesseract 5 · ${block.ocr_mode === 'scan_tile' ? 'tile recall' : 'full-page recall'}`, function: 'scan_full_page_tile_recall' },
            ]
        : nemoIsOption3(block)
        ? [
            { kind: 'ocr', name: 'Nemotron OCR v2 · baseline/fallback', function: 'invoke_image_inference_batches' },
            { kind: 'router', name: 'Unicode + langdetect language router', function: 'route_nemotron_text' },
            { kind: 'ocr', name: 'VietOCR vgg_seq2seq · quality-gated candidate', function: 'vietnamese_recognizer_batch' },
          ]
        : block.ocr_source
          ? [
            { kind: 'detector', name: 'OCR backend đã trả bbox', function: 'ocr_backend_bbox' },
            {
              kind: 'ocr',
              name: String(block.ocr_source) === 'tesseract-5' ? 'Tesseract 5 · đọc crop' : `OCR · ${block.ocr_source}`,
              function: 'ocr_region_recognize',
            },
          ]
        : block.reader_backend === 'native_pdf'
        ? [{ kind: 'library', name: 'pypdfium2', function: 'pdf_extraction' }]
        : ['image', 'chart', 'infographic', 'stamp'].includes(String(block.content_type || ''))
          ? [{ kind: 'visual', name: 'Crop ảnh theo bbox', function: 'page_image_crop' }]
        : [{ kind: 'ocr', name: 'OCR visual evidence', function: 'dashboard_visual_projection' }],
      output_keys: ['text', 'bbox'],
      outputs: {
        text: block.text || '',
        bbox: block.bbox,
        model_bbox: block.model_bbox || block.bbox,
        processed_bbox: block.processed_bbox || null,
        crop_bbox: block.crop_bbox || block.bbox,
        confidence: block.confidence,
        origin: block.origin,
        ocr_source: block.ocr_source || null,
        ocr_mode: block.ocr_mode || null,
        line_detector_score: block.line_detector_score ?? null,
        page_elements_score: block.page_elements_score ?? null,
        region_label: block.region_label || null,
        provenance: block.provenance || null,
        selected_backend: block.selected_backend || null,
        route: block.route || null,
        fallback_reason: block.fallback_reason || null,
        bbox_source: block.bbox_source || null,
        nemotron_original_text: block.nemotron_original_text || null,
        vietnamese_candidate_text: block.vietnamese_candidate_text || null,
        model: block.model || null,
        overlaps_regions: block.overlaps_regions || [],
      },
      overlaps_regions: block.overlaps_regions || [],
      contains_text_blocks: block.contains_text_blocks || [],
      image_index: null,
      image_url: visualEvidence && visualEvidence.block_image_endpoint
        ? visualEvidence.block_image_endpoint
            .replace('{page_number}', encodeURIComponent(String(page.page_number)))
            .replace('{block_id}', encodeURIComponent(String(block.id || `visual-${page.page_number}-${index}`)))
        : null,
      embedding: null,
    }));
    const spreadsheetStages = [
      { id: 'split', label: 'Tách workbook thành sheet / vùng dữ liệu', status: 'observed', executor: 'SpreadsheetExtractActor', function: 'spreadsheet_bytes_to_chunks_df', library: sourceExtension === 'csv' ? 'Python csv' : 'openpyxl', output: { page_count: 1, sheet: page.blocks && page.blocks[0] && page.blocks[0].sheet_name || 'CSV' } },
      { id: 'extract', label: 'Đọc cell native / bản ghi CSV', status: 'observed', executor: 'SpreadsheetExtractActor', function: sourceExtension === 'csv' ? '_csv_to_rows' : '_xlsx_to_rows', library: sourceExtension === 'csv' ? 'csv.reader' : 'openpyxl', output: { blocks: blocks.length, reader_backend: blocks[0] && blocks[0].reader_backend || (sourceExtension === 'csv' ? 'python_csv' : 'openpyxl') } },
      { id: 'page_elements', label: 'Nhận diện bố cục bằng model', status: 'not_applicable', executor: 'SpreadsheetExtractActor', output: { note: 'Không chạy cho Excel/CSV native.' } },
      { id: 'table_structure', label: 'Tách cấu trúc bảng bằng model', status: 'not_applicable', executor: 'SpreadsheetExtractActor', output: { note: 'Grid đã có sẵn từ cell/range native.' } },
      { id: 'ocr', label: 'OCR text', status: 'not_applicable', executor: 'SpreadsheetExtractActor', output: { note: 'Không OCR text trong cell native.' } },
      { id: 'clean', label: 'Chuẩn hóa grid / giữ provenance', status: 'observed', executor: 'SpreadsheetExtractActor', function: 'canonicalize_spreadsheet_text', output: { block_count: blocks.length } },
      { id: 'explode', label: 'Sinh block Markdown', status: 'observed', executor: 'SpreadsheetExtractActor', function: 'spreadsheet_bytes_to_chunks_df', output: { block_count: blocks.length } },
    ];
    return {
      ...existing,
      page_number: page.page_number,
      source_id: page.source_id || existing.source_id,
      reader_backend: page.blocks && page.blocks.some(block => block.reader_backend === 'native_pdf')
        ? 'native_pdf'
        : (blocks.find(block => block.reader_backend) || {}).reader_backend || (existing.reader_backend || (isSpreadsheetFile ? (sourceExtension === 'csv' ? 'python_csv' : 'openpyxl') : 'ocr')),
      text_chars: blocks.reduce((total, block) => total + String(block.text || '').length, 0),
      block_count: blocks.length,
      content_types: Array.from(new Set(blocks.map(block => block.content_type))),
      blocks,
      stages: Array.isArray(existing.stages) && existing.stages.length
        ? existing.stages
        : (isSpreadsheetFile ? spreadsheetStages : (existing.stages || [])),
    };
  };
  // The sidecar contributes page images, visual crops and scan line boxes,
  // but it is not a replacement for the backend trace.  Merging both keeps
  // native text below an image visible even when result_data is not retained
  // in the document-detail response.
  const visualTracePages = visualPages.map(visualPageAsTracePage);
  const tracePages = visualPages.length
    ? nemoMergeTracePages(backendTracePages, visualTracePages)
    : backendTracePages;
  const isOfficeDocumentFile = ['ppt', 'pptx', 'doc', 'docx'].includes(sourceExtension);
  const isTextFile = ['txt', 'md', 'json', 'sh', 'html'].includes(sourceExtension);
  const isImageFile = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'svg'].includes(sourceExtension);
  const isAudioFile = ['mp3', 'wav', 'm4a'].includes(sourceExtension);
  const isVideoFile = ['mp4', 'mov', 'mkv', 'avi'].includes(sourceExtension);
  const fileKindLabel = isSpreadsheetFile ? 'Excel/CSV' : sourceExtension ? sourceExtension.toUpperCase() : 'Tài liệu';
  const pageUnitLabel = isSpreadsheetFile ? 'sheet' : 'trang';
  const displayPage = selectedPage || (tracePages[0] && tracePages[0].page_number) || activePage;
  const selectedTracePage = tracePages.find(page => page.page_number === displayPage) || null;
  const selectedVisualPage = visualPages.find(page => page.page_number === displayPage) || null;
  const visualPageImageUrl = visualEvidence && visualEvidence.image_endpoint && selectedVisualPage && selectedVisualPage.image_available
    ? visualEvidence.image_endpoint.replace('{page_number}', encodeURIComponent(String(displayPage)))
    : null;
  // Use the canonical blocks from the trace for the page view.  The old
  // recursive visual walker is kept for compatibility with older payloads,
  // but must not drive the new page view or it can show nested content twice.
  const pageItems = selectedTracePage
    ? selectedTracePage.blocks.map(block => ({
        key: String(block.block_id || block.row_index || `${displayPage}-${block.reading_order || 0}`),
        block_id: block.block_id || null,
        trace_source: block.trace_source || null,
        rowIndex: block.row_index,
        row: resultRows[block.row_index],
        page: displayPage,
        text: block.text,
        content_type: block.content_type,
        reader_backend: block.reader_backend
          || (block.trace_source === 'visual_evidence' ? null : selectedTracePage.reader_backend || null),
        origin: block.origin || null,
        content_origin: block.content_origin || null,
        source: block.source || null,
        ocr_source: block.ocr_source || null,
        ocr_mode: block.ocr_mode || null,
        ocr_pipeline_name: block.ocr_pipeline_name || null,
        pipeline_name: block.pipeline_name || null,
        line_detector_score: block.line_detector_score ?? null,
        bbox: block.bbox,
        model_bbox: block.model_bbox || block.bbox,
        processed_bbox: block.processed_bbox || null,
        crop_bbox: block.crop_bbox || block.bbox,
        image_url: block.image_url || null,
        label_name: block.label_name || null,
        provenance: block.provenance || null,
        selected_backend: block.selected_backend || null,
      }))
    : visualItems.filter(item => item.page === displayPage);
  const selectedDocumentMetadata = documentDetail && documentDetail.metadata && typeof documentDetail.metadata === 'object'
    ? documentDetail.metadata
    : {};
  const selectedDocumentDiagnostics = documentDetail && documentDetail.pipeline_diagnostics
    && typeof documentDetail.pipeline_diagnostics === 'object'
    ? documentDetail.pipeline_diagnostics
    : {};
  const selectedDocumentPipeline = selectedDocumentMetadata.ocr_pipeline
    || selectedDocumentDiagnostics.ocr_pipeline
    || selectedDocumentDiagnostics.pipeline_selector
    || selectedDocumentDiagnostics.pipeline;
  const hasVisualCrops = pageItems.some(item => nemoBboxIsVisual(item) && item.image_url);
  function openVisualCrop(item) {
    if (!item || !item.image_url) return;
    setOutputPopup({
      title: `${item.text || item.content_type || 'Visual'} · trang ${displayPage}`,
      subtitle: 'Crop visual được giữ riêng từ Page Elements',
      imageUrl: item.image_url,
      imageAlt: item.text || item.content_type || 'visual crop',
    });
  }
  const nativePages = new Set(resultRows.filter(row => pageExtractionSignal(row).hasText && !pageExtractionSignal(row).needsOcr).map(rowPage)).size;
  const ocrPages = new Set(resultRows.filter(row => pageExtractionSignal(row).needsOcr).map(rowPage)).size;

  function stageDescription(stage) {
    if (stage.executor === 'SpreadsheetExtractActor') {
      return {
        split: 'Tách workbook theo sheet và vùng dữ liệu; CSV được xem như một sheet duy nhất.',
        extract: 'Đọc cell, công thức và bản ghi trực tiếp bằng parser native; không chạy OCR cho text trong cell.',
        clean: 'Chuẩn hóa vùng grid, giữ sheet name và cell range để truy nguyên về file gốc.',
        explode: 'Sinh Markdown theo từng vùng bảng và chia chunk theo nhóm dòng để embedding.',
        embedding: 'Chuyển Markdown của từng vùng bảng thành vector để ghi vào VectorDB.',
      }[stage.id] || 'Bước native của pipeline Excel/CSV.';
    }
    return {
      split: 'Tách file PDF thành các trang riêng để từng trang có thể được đọc và theo dõi độc lập.',
      extract: 'Đọc text layer và tạo ảnh trang bằng thư viện pypdfium2. Nếu PDF có text layer thì đây là đường native PDF.',
      page_elements: 'NIM tìm vùng text/title/table/chart/image và bbox. Pipeline 7 dùng text/title/table bbox để lập crop OCR; chart/image chỉ giữ bbox làm evidence, không tạo visual crop gửi VLM.',
      stamp_detection: 'Ooredoo YOLOS tìm vùng mộc/con dấu trên ảnh trang scan; kết quả là bbox để cắt ảnh và đưa riêng qua OCR.',
      table_structure: 'NIM phân tích các ô, hàng và cột của vùng bảng. Bước này chỉ chạy khi Page Elements phát hiện bảng.',
      nemotron_baseline: 'Nemotron OCR v2 nhận tất cả semantic crops theo batch; mỗi recognition item giữ text, score và bbox local để map về trang.',
      language_router: 'Router chỉ nhận raw text Nemotron: strong Vietnamese Unicode trước, sau đó langdetect vi/en deterministic; English và uncertain giữ Nemotron.',
      vietnamese_recognizer: 'VietOCR vgg_seq2seq nhận một logical batch các crop route Vietnamese. Quality Gate quyết định thay text hoặc fallback Nemotron.',
      ministral_vlm: 'Ministral 3 3B FP8 chỉ OCR semantic text/title/table crop hoặc full-page fallback; không phân loại visual và không nhận visual crop riêng.',
      ocr: 'Option Tesseract: NIM Page Elements cung cấp vùng text/table, Tesseract đọc từng vùng/cell; scan chỉ fallback toàn trang + tile khi vùng không trả text. Chart và ảnh được giữ riêng theo bbox.',
      clean: 'Đối chiếu bbox native với vùng table/chart, bỏ phần native bị trùng và giữ raw_text để kiểm tra lại.',
      explode: 'Tách kết quả trang thành các block độc lập để hiển thị, tìm kiếm và embedding.',
      embedding: 'Chuyển text của block thành vector số để ghi vào VectorDB và phục vụ truy vấn.',
    }[stage.id] || 'Bước xử lý nội bộ của pipeline ingest.';
  }

  function readableContentType(type) {
    return { text: 'Văn bản', table: 'Bảng', spreadsheet_table: 'Bảng native', chart: 'Biểu đồ', infographic: 'Infographic', stamp: 'Mộc / con dấu', image: 'Hình ảnh' }[type] || type || 'Không xác định';
  }

  function pageTiming(page) {
    const finishedAt = (documentDetail && documentDetail.completed_at) || (job && job.finalized_at);
    return {
      'Trang': page ? page.page_number : displayPage,
      'Trạng thái': documentDetail ? statusLabel(documentDetail.status) : '—',
      'Kết quả trang được ghi nhận lúc': finishedAt ? fmtTime(finishedAt) : 'Chưa hoàn tất',
      'Phạm vi thời gian': 'Đây là thời điểm document/job hoàn tất; backend chưa lưu timestamp riêng cho từng trang.',
    };
  }

  function filePipelineSummary() {
    const visualPageCount = visualPages.length;
    const visualBlockCount = visualEvidence ? visualEvidence.block_count : 0;
    const visualSidecarObserved = isSpreadsheetFile && visualPageCount > 0;
    return (pipelineTrace && pipelineTrace.file && pipelineTrace.file.stages || []).map(stage => {
      const output = { ...(stage.output || {}) };
      if (visualSidecarObserved && ['split', 'pages', 'retain'].includes(stage.id)) {
        output.page_count = visualPageCount;
        output.block_count = visualBlockCount;
        output.evidence = 'visual evidence native đã giữ lại';
      }
      return {
        'Bước': stage.label,
        'Ý nghĩa': ({
          receive: 'Job Tracker lưu file và tạo document để worker xử lý.',
          classify: `Xác định file là ${isSpreadsheetFile ? 'Excel/CSV' : 'PDF'} dựa trên phần mở rộng và MIME được cho phép.`,
          route: 'Đưa file vào worker đúng pipeline theo loại dữ liệu.',
          split: isSpreadsheetFile ? 'Đọc workbook theo sheet/range; CSV được xem như một sheet native.' : 'Tách PDF thành từng trang.',
          pages: isSpreadsheetFile ? 'Mỗi sheet/range native được biểu diễn thành block Markdown; không chạy OCR.' : 'Chạy pipeline riêng cho từng trang.',
          vdb: 'Ghi các vector đã tạo vào VectorDB.',
          retain: 'Giữ result_data/visual evidence để frontend mở text, metadata, bbox và output.',
        }[stage.id] || 'Bước quản lý dữ liệu của job.'),
        'Kết quả': output,
      };
    });
  }

  return React.createElement(React.Fragment, null,

    React.createElement(PipelineOutputPopup, { popup: outputPopup, onClose: () => setOutputPopup(null), onOpenOutput: value => setOutputPopup(value) }),

    /* Header bar */
    React.createElement('div', {
      style: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }
    },
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, background: 'var(--nv-surface)', color: 'var(--nv-text)' },
        onClick: () => onBack && onBack(),
      }, '← Về danh sách job'),
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, background: 'rgba(255,80,80,0.12)', color: 'var(--nv-red)' },
        onClick: deleteCurrentJob,
      }, 'Xóa job'),
      React.createElement('span', {
        className: 'mono',
        style: { fontSize: 13, color: 'var(--nv-text-muted)' }
      }, jobId),
      job && statusBadge(job.status),
      React.createElement('span', {
        className: `status-dot ${sseStatus === 'connected' ? 'ok' : sseStatus === 'connecting' ? 'unknown' : 'error'}`,
        style: { marginLeft: 'auto' },
      }),
      React.createElement('span', { style: { fontSize: 12, color: 'var(--nv-text-muted)' } },
        `Luồng SSE: ${sseStatus}`
      ),
    ),

    error && React.createElement('div', {
      className: 'card',
      style: { marginBottom: 16, color: 'var(--nv-red)' }
    }, error),

    /* Aggregate header card */
    job && React.createElement('div', {
      className: 'card',
      style: { marginBottom: 24 }
    },
      React.createElement('div', {
        style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }
      },
        React.createElement('div', null,
          React.createElement('div', { style: { fontSize: 18, fontWeight: 600 } },
            job.label || `Job ${jobId.substring(0, 12)}…`
          ),
          React.createElement('div', { style: { fontSize: 12, color: 'var(--nv-text-muted)', marginTop: 4 } },
            `Tạo lúc ${fmtTime(job.created_at)}  •  Bắt đầu ${fmtTime(job.started_at)}  •  Kết thúc ${fmtTime(job.finalized_at)}`
          ),
          React.createElement('div', {
            style: { display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', fontSize: 12, color: 'var(--nv-text-muted)', marginTop: 6 }
          },
            React.createElement('span', { style: { fontWeight: 600 } }, 'Trace ID'),
            React.createElement('span', {
              className: 'mono',
              title: job.trace_id || '',
              style: { color: 'var(--nv-text)', userSelect: 'text' },
            }, job.trace_id || '—'),
          ),
        ),
        React.createElement('div', {
          className: 'mono',
          style: { fontSize: 13, color: 'var(--nv-text-muted)' }
        }, job.elapsed_s != null ? `đã chạy ${job.elapsed_s.toFixed(1)} giây` : ''),
      ),
      React.createElement('div', { className: 'progress-bar', style: { height: 24 } },
        React.createElement('div', {
          className: 'progress-fill',
          style: { width: pct + '%' },
        }),
        React.createElement('div', { className: 'progress-label' },
          `${c + f} / ${exp} (${pct}%)`
        ),
      ),
      React.createElement('div', {
        style: { display: 'flex', gap: 24, marginTop: 16, fontSize: 13 }
      },
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-green)', fontWeight: 600 } }, `${c}`),
          ' hoàn tất',
        ),
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-yellow)', fontWeight: 600 } }, `${p}`),
          ' đang xử lý',
        ),
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-red)', fontWeight: 600 } }, `${f}`),
          ' lỗi',
        ),
        React.createElement('span', null,
          React.createElement('span', { style: { color: 'var(--nv-blue)', fontWeight: 600 } }, `${exp - c - f - p}`),
          ' đang chờ',
        ),
      ),
    ),

    documentDetail && React.createElement(Option5Diagnostics, {
      diagnostics: documentDetail.pipeline_diagnostics,
    }),

    /* Page-first inspection: the original page is the reading surface.  The
     * bbox reconstruction remains available as a collapsed geometry debug
     * view, while parsed text is rendered in a separate readable panel. */
    React.createElement('div', { className: 'card', style: { marginBottom: 24 } },
      React.createElement('div', { className: 'card-title' }, 'Lịch sử ingest theo trang'),
      pipelineTraceError && React.createElement('div', { style: { color: 'var(--nv-red)', fontSize: 12, marginBottom: 12 } }, `Không tải được trace: ${pipelineTraceError}`),
      visualEvidenceError && React.createElement('div', { style: { color: 'var(--nv-yellow)', fontSize: 12, marginBottom: 12 } }, `Visual evidence chưa tải được: ${visualEvidenceError}`),
      !pipelineTrace
        ? React.createElement('div', { className: 'empty-state', style: { padding: 24 } }, 'Đang tải pipeline tài liệu…')
        : React.createElement(React.Fragment, null,
            React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 14 } },
                React.createElement('div', { style: { flex: 1, minWidth: 260 } },
                  React.createElement('div', { style: { fontSize: 13, fontWeight: 600 } }, pipelineTrace.file.filename || 'Tài liệu'),
              React.createElement('div', { style: { fontSize: 11, color: 'var(--nv-text-muted)', marginTop: 3 } },
                  `${fileKindLabel} · ${tracePages.length} ${pageUnitLabel} · ${visualEvidence ? visualEvidence.block_count : (pipelineTrace.file.result_rows || resultRows.length)} block đầu ra · chọn ${pageUnitLabel} để xem chi tiết`
                ),
              ),
              React.createElement('button', {
                className: 'btn', style: { padding: '6px 10px', fontSize: 11, background: 'var(--nv-surface)', color: 'var(--nv-text)' },
                onClick: () => setOutputPopup({ title: 'Pipeline toàn file', subtitle: 'Giải thích các bước trước khi vào từng trang', value: filePipelineSummary() }),
              }, 'Xem pipeline file'),
              React.createElement('select', {
                className: 'input', style: { padding: '7px 10px', minWidth: 190, fontWeight: 600 },
                value: displayPage || '',
                'aria-label': isSpreadsheetFile ? 'Chọn sheet' : 'Chọn trang PDF',
                onChange: e => {
                  const page = Number(e.target.value);
                  setSelectedPage(page);
                  const firstRow = resultRows.findIndex(row => rowPage(row) === page);
                  if (firstRow >= 0) setSelectedRow(firstRow);
                },
              }, tracePages.map(page => React.createElement('option', { key: page.page_number, value: page.page_number }, `${isSpreadsheetFile ? 'Sheet' : 'Trang'} ${page.page_number} · ${page.block_count} block`))),
            ),
            selectedTracePage && React.createElement(React.Fragment, null,
              React.createElement('div', { style: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', padding: '10px 12px', marginBottom: 14, border: '1px solid var(--nv-border)', borderRadius: 6, background: 'var(--nv-surface)' } },
                React.createElement('div', { style: { flex: 1, minWidth: 240 } },
                  React.createElement('div', { style: { fontSize: 15, fontWeight: 600 } }, `${isSpreadsheetFile ? 'Sheet' : 'Trang'} ${selectedTracePage.page_number}`),
                  React.createElement('div', { style: { fontSize: 11, color: 'var(--nv-text-muted)', marginTop: 3 } },
                    `${selectedTracePage.block_count} block · ${selectedTracePage.text_chars} ký tự · ${isNativeReader(selectedTracePage.reader_backend) ? 'đọc native bằng thư viện' : 'có vùng OCR'} · ${selectedTracePage.ocr_backend === 'option2_nemotron_language_routed_vietnamese_ocr' ? 'Option 2 · Nemotron → router → VietOCR' : selectedTracePage.ocr_backend === 'option3_nemotron_language_routed_vietnamese_ocr' ? 'Option 3 · Nemotron → router → VietOCR' : selectedTracePage.ocr_backend === 'option5_nemotron_language_routed_vietnamese_ocr' ? 'pipeline 5 (fast)' : selectedTracePage.ocr_backend === 'option6_page_detect_qwen35_vlm' ? 'pipeline 6 · Qwen 3.5 VLM' : selectedTracePage.ocr_backend === 'option7_ministral_vlm' ? 'pipeline 7 · Ministral 3 3B VLM' : selectedTracePage.content_types.map(readableContentType).join(', ') || 'chưa xác định'}`
                  ),
                ),
              React.createElement('button', {
                className: 'btn', style: { padding: '6px 9px', fontSize: 11, background: 'var(--nv-bg)', color: 'var(--nv-green)' },
                  onClick: () => setOutputPopup({ title: `Pipeline ${isSpreadsheetFile ? 'sheet' : 'trang'} ${selectedTracePage.page_number}`, subtitle: 'Flow xử lý của vùng đang chọn; mỗi bước có output riêng', flow: (selectedTracePage.stages || []).map(stage => ({ ...stage, description: stageDescription(stage) })) }),
                }, `Xem pipeline ${isSpreadsheetFile ? 'sheet' : 'trang'}`),
                React.createElement('button', {
                  className: 'btn', style: { padding: '6px 9px', fontSize: 11, background: 'var(--nv-bg)', color: 'var(--nv-text)' },
                  onClick: () => setOutputPopup({ title: `Thời gian trang ${selectedTracePage.page_number}`, subtitle: 'Thời gian được lưu trong backend', value: pageTiming(selectedTracePage) }),
                }, 'Xem thời gian'),
              ),
              React.createElement('div', null,
                React.createElement('div', { style: { fontSize: 13, fontWeight: 600, padding: '8px 10px', marginBottom: 8, border: '1px solid var(--nv-border)', borderRadius: 6, background: 'var(--nv-surface)' } }, isSpreadsheetFile ? `Dữ liệu native đã parse · sheet ${selectedTracePage.page_number}` : `Trang gốc · trang ${selectedTracePage.page_number}`),
                isSpreadsheetFile
                  ? React.createElement(NativeSpreadsheetPageView, { page: selectedTracePage })
                  : React.createElement(OriginalPagePreview, {
                    selectedTracePage,
                    displayPage,
                    pageItems,
                    hoveredRow,
                    onHoverItem: setHoveredRow,
                    isSpreadsheetFile,
                    isOfficeDocumentFile,
                    isImageFile,
                    isTextFile,
                    isAudioFile,
                    isVideoFile,
                    visualPageImageUrl,
                    sourceFileUrl,
                    sourceText,
                    sourcePdfBlob,
                    sourcePdfUrl,
                  }),
                !isSpreadsheetFile && React.createElement(PageContentPanel, {
                  pageNumber: displayPage,
                  items: pageItems,
                  sourcePdfBlob,
                  hoveredRow,
                  onHoverItem: setHoveredRow,
                  onOpenVisual: openVisualCrop,
                }),
                !isSpreadsheetFile && React.createElement('details', { style: { marginTop: 12 } },
                  React.createElement('summary', { style: { cursor: 'pointer', color: 'var(--nv-text-muted)', fontSize: 11, padding: '8px 0' } }, 'Xem debug bbox · không dùng lớp này để đọc nội dung'),
                  React.createElement(ReconstructedPageView, {
                    pageNumber: displayPage,
                    items: pageItems,
                    hoveredRow,
                    onHoverItem: setHoveredRow,
                    onOpenVisual: openVisualCrop,
                    width: selectedVisualPage && selectedVisualPage.width,
                    height: selectedVisualPage && selectedVisualPage.height,
                  }),
                ),
              ),
              hasVisualCrops && React.createElement(VisualCropGallery, {
                page: selectedTracePage,
                onOpenVisual: openVisualCrop,
              }),
            ),
          ),
    ),

    /* Documents table */
    React.createElement('div', { className: 'section' },
      React.createElement('div', {
        style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }
      },
        React.createElement('div', { className: 'section-title', style: { marginBottom: 0 } }, 'Tài liệu'),
        React.createElement('div', { style: { display: 'flex', gap: 8, fontSize: 12, color: 'var(--nv-text-muted)' } },
          `${docs.length} / ${docTotalFiltered} (${docTotal} tổng số)`,
          React.createElement('select', {
            className: 'input',
            style: { padding: '4px 8px', fontSize: 12 },
            value: docStatusFilter,
            onChange: (e) => { setDocOffset(0); setDocStatusFilter(e.target.value); },
          },
            React.createElement('option', { value: '' }, 'Tất cả trạng thái'),
            ['pending', 'processing', 'completed', 'failed'].map(s =>
              React.createElement('option', { key: s, value: s }, ({
                pending: 'đang chờ', processing: 'đang xử lý', completed: 'hoàn tất', failed: 'lỗi',
              })[s])
            ),
          ),
          React.createElement('button', {
            className: 'btn btn-primary',
            style: { padding: '6px 12px', fontSize: 12 },
            onClick: () => { fetchAggregate(); fetchDocs(); },
          }, 'Làm mới'),
        ),
      ),
      docs.length === 0
        ? React.createElement('div', { className: 'empty-state' }, 'Không có tài liệu để hiển thị')
        : React.createElement('div', { className: 'table-wrap' },
            React.createElement('table', null,
              React.createElement('thead', null,
                React.createElement('tr', null,
                  ['ID tài liệu', 'Tên file', 'Trạng thái', 'Gửi lúc', 'Thời gian', 'Số row', 'Lỗi', ''].map(h =>
                    React.createElement('th', { key: h }, h)
                  )
                )
              ),
              React.createElement('tbody', null,
                docs.map(d =>
                  React.createElement('tr', { key: d.id },
                    React.createElement('td', { className: 'mono', style: { fontSize: 11 } },
                      (d.id || '').substring(0, 12) + '…'
                    ),
                    React.createElement('td', {
                      style: { maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
                      title: d.filename || '',
                    }, d.filename || '—'),
                    React.createElement('td', null, statusBadge(d.status)),
                    React.createElement('td', { title: d.submitted_at || '' },
                      d.submitted_at ? new Date(d.submitted_at).toLocaleTimeString() : '—'
                    ),
                    React.createElement('td', { className: 'mono' },
                      d.elapsed_s != null ? d.elapsed_s.toFixed(1) + 's' : '—'
                    ),
                    React.createElement('td', { className: 'mono' },
                      d.result_rows != null ? d.result_rows.toLocaleString() : '—'
                    ),
                    React.createElement('td', {
                      style: {
                        color: d.error ? 'var(--nv-red)' : 'inherit',
                        maxWidth: 300, overflow: 'hidden',
                        textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      },
                      title: d.error || '',
                    }, d.error || '—'),
                    React.createElement('td', null,
                      ['completed', 'failed'].includes(d.status) &&
                        React.createElement('button', {
                          className: 'btn',
                          style: { padding: '2px 8px', fontSize: 11, background: 'var(--nv-surface)', color: 'var(--nv-text)' },
                          onClick: () => inspectDocument(d.id),
                        }, 'Xem'),
                    ),
                  )
                )
              )
            )
          )
    ),

    /* Pagination */
    docTotalFiltered > docLimit && React.createElement('div', {
      style: { display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 }
    },
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, opacity: docOffset === 0 ? 0.5 : 1 },
        disabled: docOffset === 0,
        onClick: () => setDocOffset(Math.max(0, docOffset - docLimit)),
      }, '← Trước'),
      React.createElement('span', { style: { padding: '6px 12px', fontSize: 12, color: 'var(--nv-text-muted)' } },
        `Trang ${Math.floor(docOffset / docLimit) + 1} / ${Math.max(1, Math.ceil(docTotalFiltered / docLimit))}`
      ),
      React.createElement('button', {
        className: 'btn',
        style: { padding: '6px 12px', fontSize: 12, opacity: docOffset + docLimit >= docTotalFiltered ? 0.5 : 1 },
        disabled: docOffset + docLimit >= docTotalFiltered,
        onClick: () => setDocOffset(docOffset + docLimit),
      }, 'Tiếp →'),
    ),
  );
}
