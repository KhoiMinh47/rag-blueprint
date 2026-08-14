/* VDB Explorer — table info, live query interface, results table */

function VdbView() {
  const [tables, setTables] = React.useState(null);
  const [tableError, setTableError] = React.useState(null);
  const [query, setQuery] = React.useState('');
  const [topK, setTopK] = React.useState(10);
  const [results, setResults] = React.useState(null);
  const [searching, setSearching] = React.useState(false);
  const [queryError, setQueryError] = React.useState(null);

  React.useEffect(() => {
    fetch('/v1/dashboard/api/vdb/tables')
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(setTables)
      .catch(e => setTableError(e.message));
  }, []);

  const doSearch = React.useCallback(() => {
    if (!query.trim()) return;
    setSearching(true);
    setQueryError(null);
    setResults(null);
    fetch('/v1/dashboard/api/vdb/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query.trim(), top_k: topK }),
    })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(data => { setResults(data); setSearching(false); })
      .catch(e => { setQueryError(e.message); setSearching(false); });
  }, [query, topK]);

  const onKeyDown = React.useCallback((e) => {
    if (e.key === 'Enter') doSearch();
  }, [doSearch]);

  const hits = React.useMemo(() => {
    if (!results || !results.results || !results.results.length) return [];
    const first = results.results[0];
    return Array.isArray(first.hits) ? first.hits : [];
  }, [results]);

  const hitColumns = React.useMemo(() => {
    if (hits.length === 0) return [];
    const cols = new Set();
    hits.forEach(h => Object.keys(h).forEach(k => cols.add(k)));
    const priority = ['score', 'source_id', 'page_number', 'content_type', 'text'];
    const ordered = priority.filter(c => cols.has(c));
    cols.forEach(c => { if (!priority.includes(c)) ordered.push(c); });
    return ordered;
  }, [hits]);

  function formatCell(val, col) {
    if (val == null) return '—';
    if (col === 'score') return Number(val).toFixed(4);
    if (col === 'text' || col === 'content') {
      const s = String(val);
      return s.length > 200 ? s.substring(0, 200) + '…' : s;
    }
    return String(val);
  }

  return React.createElement(React.Fragment, null,

    /* Table info */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'Cơ sở dữ liệu vector'),
      tableError
        ? React.createElement('div', { className: 'card' },
            React.createElement('div', { style: { color: 'var(--nv-red)' } },
              'Tải thông tin VDB thất bại: ' + tableError
            )
          )
        : !tables
          ? React.createElement('div', { className: 'empty-state' }, 'Đang tải…')
          : tables.error
            ? React.createElement('div', { className: 'card' },
                React.createElement('div', { style: { color: 'var(--nv-yellow)' } }, tables.error)
              )
            : React.createElement('div', { className: 'card-grid' },
                (tables.tables || []).map((t, i) =>
                  React.createElement('div', { key: i, className: 'card' },
                    React.createElement('div', { className: 'card-title' }, 'Bảng'),
                    React.createElement('div', { style: { marginBottom: 8 } },
                      React.createElement('span', { className: 'mono', style: { fontSize: 16, fontWeight: 600 } }, t.name || '—'),
                    ),
                    React.createElement('div', { className: 'pod-card-detail' },
                      React.createElement('span', { className: `status-dot ${t.exists ? 'ok' : 'error'}` }),
                      t.exists ? 'Đang tồn tại' : 'Không tìm thấy',
                    ),
                    React.createElement('div', { className: 'pod-card-detail' },
                      `Số row: ${(t.total_rows || 0).toLocaleString()}`,
                    ),
                  )
                )
              ),
    ),

    /* Query interface */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' }, 'Truy vấn'),
      React.createElement('div', { className: 'query-bar' },
        React.createElement('input', {
          className: 'input',
          type: 'text',
          placeholder: 'Nhập nội dung cần tìm…',
          value: query,
          onChange: e => setQuery(e.target.value),
          onKeyDown: onKeyDown,
        }),
        React.createElement('input', {
          className: 'input input-small',
          type: 'number',
          min: 1,
          max: 1000,
          value: topK,
          onChange: e => setTopK(Math.max(1, parseInt(e.target.value) || 1)),
          title: 'top_k',
        }),
        React.createElement('button', {
          className: 'btn btn-primary',
          onClick: doSearch,
          disabled: searching || !query.trim(),
        }, searching ? 'Đang tìm…' : 'Tìm kiếm'),
      ),
      queryError && React.createElement('div', {
        style: { color: 'var(--nv-red)', fontSize: 13, marginBottom: 12 },
      }, 'Lỗi: ' + queryError),
    ),

    /* Results table */
    React.createElement('div', { className: 'section' },
      React.createElement('div', { className: 'section-title' },
        results ? `Kết quả (${hits.length} mục)` : 'Kết quả'
      ),
      !results && !searching
        ? React.createElement('div', { className: 'empty-state' }, 'Thực hiện truy vấn để xem kết quả')
        : searching
          ? React.createElement('div', { className: 'empty-state' }, 'Đang tìm…')
          : hits.length === 0
            ? React.createElement('div', { className: 'empty-state' }, 'Không tìm thấy kết quả')
            : React.createElement('div', { className: 'table-wrap' },
                React.createElement('table', null,
                  React.createElement('thead', null,
                    React.createElement('tr', null,
                      hitColumns.map(col =>
                        React.createElement('th', { key: col }, col)
                      )
                    )
                  ),
                  React.createElement('tbody', null,
                    hits.map((hit, idx) =>
                      React.createElement('tr', { key: idx },
                        hitColumns.map(col =>
                          React.createElement('td', {
                            key: col,
                            className: ['score', 'page_number'].includes(col) ? 'mono' : '',
                            style: col === 'text' || col === 'content'
                              ? { maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }
                              : {},
                            title: col === 'text' || col === 'content' ? String(hit[col] || '') : undefined,
                          }, formatCell(hit[col], col))
                        )
                      )
                    )
                  )
                )
              ),
    ),
  );
}
