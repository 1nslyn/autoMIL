/* Nodes: every attempt as a sortable table, and technique effectiveness when it was tracked. */
(function (AM) {
  'use strict';
  const { h, fmt } = AM;

  const COLUMNS = [
    { key: 'id', label: 'Node', get: (n) => n.id, render: (n) => h('td.mono', n.id) },
    { key: 'status', label: 'Status', get: (n) => n.status, render: (n) => h('td', h('span.status-dot', { class: AM.statusClass(n.status) }), AM.statusLabel(n.status)) },
    { key: 'description', label: 'Description', get: (n) => n.description || '', render: (n) => h('td.desc-cell', { title: n.description || '' }, n.description || '') },
    { key: 'primary_value', label: 'Value', num: true, get: (n) => n.primary_value, render: (n) => h('td.num', n.type === 'executed' ? fmt.num(n.primary_value) : '–') },
    { key: 'parent_delta', label: 'vs parent', num: true, get: (n, g) => delta(n, g), render: (n, g) => { const d = delta(n, g); return h('td.num', { class: d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : '' }, d == null ? '–' : fmt.delta(d)); } },
    { key: 'primary_se', label: 'SE', num: true, get: (n) => n.primary_se, render: (n) => h('td.num', n.primary_se != null ? fmt.num(n.primary_se) : '–') },
    { key: 'kind', label: 'Kind', get: (n) => n.kind || '', render: (n) => h('td', n.kind || '') },
    { key: 'elapsed_min', label: 'Minutes', num: true, get: (n) => n.elapsed_min, render: (n) => h('td.num', n.elapsed_min != null ? fmt.num(n.elapsed_min, 1) : '–') },
    { key: 'vram_gb', label: 'VRAM GB', num: true, get: (n) => n.vram_gb, render: (n) => h('td.num', n.vram_gb != null ? fmt.num(n.vram_gb, 1) : '–') },
    { key: 'created_at', label: 'Created', get: (n) => n.created_at || '', render: (n) => h('td', fmt.when(n.created_at)) },
  ];

  function delta(n, graph) {
    const parent = n.parent_id ? graph.nodes[n.parent_id] : null;
    if (!parent || parent.primary_value == null || n.primary_value == null || n.type !== 'executed') return null;
    return n.primary_value - parent.primary_value;
  }

  function techniques(stats) {
    const keys = Object.keys(stats || {});
    if (!keys.length) return null;
    keys.sort((a, b) => (stats[b].times_tried || 0) - (stats[a].times_tried || 0));
    return h('section', { style: { marginTop: '40px' } }, h('h2', 'Techniques'),
      h('p', 'Aggregated over every attempt that declared the technique.'),
      h('div.table-wrap', h('table', h('thead', h('tr', h('th', 'Technique'), h('th.num', 'Tried'), h('th.num', 'Best gain vs parent'), h('th.num', 'Mean gain vs parent'))),
        h('tbody', keys.map((k) => {
          const s = stats[k];
          const best = s.best_parent_delta === -Infinity || s.best_parent_delta == null ? 0 : s.best_parent_delta;
          return h('tr', h('td', k), h('td.num', String(s.times_tried || 0)), h('td.num', fmt.delta(best)), h('td.num', fmt.delta(s.avg_parent_delta)));
        })))));
  }

  AM.views.nodes = async function (root, ctx) {
    const { run } = ctx;
    const source = run.source;
    let graph = await source.graph(run.id);
    let sortKey = 'created_at';
    let sortDir = 'asc';
    let statusFilter = 'all';
    const tbody = h('tbody');
    const thead = h('thead');
    const count = h('span.muted');
    const select = h('select', ['all', 'executed', 'keep', 'discard', 'pending', 'running', 'crash'].map((s) => h('option', { value: s }, s === 'all' ? 'all nodes' : s === 'executed' ? 'executed only' : AM.statusLabel(s))));
    select.addEventListener('change', () => { statusFilter = select.value; render(); });
    const page = h('div.page',
      h('div.page-title', h('h1', 'Nodes'), h('p', 'Every proposal and attempt in the tree. Click a row for the full record of that node.')),
      h('div.table-tools', h('label', 'Show ', select), count),
      h('div.table-wrap', h('table', thead, tbody)),
      h('div', { id: 'techniques' }),
    );
    root.append(page);

    function render() {
      let rows = Object.values(graph.nodes);
      if (statusFilter === 'executed') rows = rows.filter((n) => n.type === 'executed');
      else if (statusFilter !== 'all') rows = rows.filter((n) => n.status === statusFilter);
      const column = COLUMNS.find((c) => c.key === sortKey);
      rows.sort((a, b) => {
        const va = column.get(a, graph);
        const vb = column.get(b, graph);
        const cmp = va == null ? 1 : vb == null ? -1 : typeof va === 'number' ? va - vb : String(va).localeCompare(String(vb));
        return sortDir === 'asc' ? cmp : -cmp;
      });
      AM.clear(thead).append(h('tr', COLUMNS.map((c) => h('th.sortable', { class: (c.num ? 'num ' : '') + (c.key === sortKey ? sortDir : ''), onclick: () => {
        if (sortKey === c.key) sortDir = sortDir === 'asc' ? 'desc' : 'asc'; else { sortKey = c.key; sortDir = c.num ? 'desc' : 'asc'; }
        render();
      } }, c.label))));
      AM.append(AM.clear(tbody), rows.map((n) => h('tr.clickable', { onclick: () => AM.nodeDrawer.open(run, n.id) }, COLUMNS.map((c) => c.render(n, graph)))));
      count.textContent = `${rows.length} of ${Object.keys(graph.nodes).length}`;
      const tech = techniques(graph.technique_stats);
      const holder = document.getElementById('techniques');
      AM.clear(holder);
      if (tech) holder.append(tech);
    }
    render();
    const unsubscribe = source.subscribe(async (frame) => {
      if (frame.type === 'graph_update') { graph = await source.graph(run.id); render(); }
    });
    return () => { unsubscribe(); AM.nodeDrawer.close(); };
  };
})(window.AM);
