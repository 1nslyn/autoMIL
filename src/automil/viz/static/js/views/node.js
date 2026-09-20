/* The node drawer: one attempt in full, with the verdict in words and the link to the agent's turn. */
(function (AM) {
  'use strict';
  const { h, fmt } = AM;

  function section(title, ...children) {
    return [h('h4', title), ...children];
  }

  function metricsList(metrics) {
    const entries = Object.entries(metrics || {}).filter(([k]) => k !== 'primary_value');
    if (!entries.length) return h('div.muted.small', 'no validation metrics recorded');
    return h('dl.kv', entries.map(([k, v]) => [h('dt', k), h('dd', typeof v === 'number' ? fmt.num(v) : String(v))]));
  }

  function foldValues(node) {
    if (Array.isArray(node.fold_primary_values)) {
      return new Map(node.fold_primary_values.map((f) => [f.fold_index, f.primary_value]));
    }
    const folds = node.metadata && node.metadata.validation_folds;
    if (Array.isArray(folds)) return new Map(folds.map((f) => [f.fold_index, f.primary_value]));
    return new Map();
  }

  function foldTable(detail) {
    const child = foldValues(detail.node);
    const parent = detail.parent ? foldValues(detail.parent) : new Map();
    if (!child.size) return null;
    const indices = Array.from(child.keys()).sort((a, b) => a - b);
    return h('table.fold-table',
      h('thead', h('tr', h('th', 'fold'), h('th.num', 'this node'), parent.size ? h('th.num', 'parent') : null, parent.size ? h('th.num', 'delta') : null)),
      h('tbody', indices.map((i) => {
        const c = child.get(i);
        const p = parent.get(i);
        return h('tr', h('td', String(i)), h('td.num', fmt.num(c)),
          parent.size ? h('td.num', fmt.num(p)) : null,
          parent.size ? h('td.num', { class: p != null && c - p > 0 ? 'delta-pos' : p != null && c - p < 0 ? 'delta-neg' : '' }, p != null ? fmt.delta(c - p) : '–') : null);
      })),
    );
  }

  function verdictBox(detail) {
    const v = detail.verdict;
    if (!v) return h('div.muted.small', `No keep or discard verdict: ${detail.verdict_unavailable || 'not judged'}.`);
    const rows = [];
    if (v.parent_id) {
      rows.push([h('dt', 'parent'), h('dd', v.parent_id + ' at ' + fmt.num(v.parent_primary_value))]);
      rows.push([h('dt', 'change'), h('dd', fmt.delta(v.delta))]);
      rows.push([h('dt', 'bar to clear'), h('dd', fmt.num(v.bar) + ` (max of δ ${fmt.num(v.accept_margin)} and ${fmt.num(v.se_multiplier, 1)} × ${v.basis} SE ${v.basis_se != null ? fmt.num(v.basis_se) : 'n/a'})`)]);
      if (v.guard && v.guard.verdict !== 'none') rows.push([h('dt', 'guard ' + (v.guard.metric || '')), h('dd', `${v.guard.delta != null ? fmt.delta(v.guard.delta) : 'unreported'}: ${v.guard.verdict}${v.guard.decisive ? ', decided the outcome' : ''}`)]);
    }
    if (!v.consistent) rows.push([h('dt', 'note'), h('dd', `stored status ${v.stored_status} differs from the recomputed decision`)]);
    return h('div.verdict', { class: v.decision }, h('div', v.explanation), rows.length ? h('dl.kv', { style: { marginTop: '8px' } }, rows) : null);
  }

  function timing(detail) {
    const t = detail.timing || {};
    const rows = [
      ['submitted', fmt.when(t.submitted_at)], ['launched', fmt.when(t.launched_at)], ['finished', fmt.when(t.completed_at)],
      ['queue wait', fmt.duration(t.queue_wait_s)], ['run time', fmt.duration(t.run_s)], ['slot', t.slot || '–'],
    ];
    const r = detail.result || {};
    if (r.peak_vram_mb != null) rows.push(['peak VRAM', `${fmt.int(r.peak_vram_mb)} MB`]);
    return h('dl.kv', rows.map(([k, v]) => [h('dt', k), h('dd', v)]));
  }

  function filesList(run, detail) {
    const overlay = detail.overlay || {};
    const items = [];
    if (overlay.run_command_override) items.push(h('div.small', h('span.muted', 'run command: '), h('code', overlay.run_command_override)));
    if (!overlay.files || !overlay.files.length) {
      items.push(h('div.muted.small', 'no source files changed (configuration-only attempt)'));
      return h('div', items);
    }
    const list = h('div.file-list');
    for (const file of overlay.files) {
      const holder = h('div');
      const button = h('button.quiet', { text: file.path });
      let open = false;
      button.addEventListener('click', async () => {
        if (open) { holder.querySelector('pre') && holder.querySelector('pre').remove(); open = false; return; }
        try {
          const payload = await run.source.file(run.id, detail.node_id, file.path);
          holder.append(h('pre', payload.text + (payload.truncated ? '\n…[file truncated]' : '')));
          open = true;
        } catch (err) {
          holder.append(h('div.error-box', `The file could not be loaded (${err.message}).`));
        }
      });
      holder.append(button);
      list.append(holder);
    }
    items.push(list);
    if (overlay.base_commit) items.push(h('div.small.muted', 'base commit ', h('code', String(overlay.base_commit).slice(0, 12))));
    return h('div', items);
  }

  function logTail(detail) {
    const log = detail.run_log || {};
    if (!log.available) return h('div.muted.small', log.reason === 'not terminal' ? 'The run log is shown once the attempt has finished.' : 'No run log recorded.');
    const details = h('details', h('summary', { class: 'small' }, `last ${log.tail_lines} of ${fmt.int(log.n_lines_total)} lines` + (log.redacted_lines ? `, ${log.redacted_lines} redacted` : '')),
      h('pre.log-tail', log.tail.join('\n')));
    return details;
  }

  function agentLinks(run, detail) {
    const links = detail.agent || [];
    if (!links.length) return h('div.muted.small', 'No transcript turn names this node.');
    const labels = { propose: 'proposed', submit: 'submitted', resubmit: 'resubmitted', mention: 'mentioned' };
    return h('ul', { style: { paddingLeft: '1.1em', margin: 0 } }, links.map((l) =>
      h('li.small', `${labels[l.kind] || l.kind} at turn ${l.turn} `,
        h('a', { href: `#/run/${encodeURIComponent(run.key)}/session/${l.session_id}/turn/${l.turn}`, text: `(${fmt.when(l.at)})` })),
    ));
  }

  const drawer = {
    el: null,
    current: null,
    init() {
      if (this.el) return;
      this.el = document.getElementById('drawer');
      document.addEventListener('keydown', (event) => { if (event.key === 'Escape') this.close(); });
    },
    async open(run, nodeId, options) {
      this.init();
      const opts = options || {};
      this.current = nodeId;
      const body = h('div.drawer-body', h('div.skeleton', { style: { height: '120px' } }));
      AM.clear(this.el).append(
        h('div.drawer-head', h('h2', nodeId), h('span.small.muted', 'loading'), h('button.quiet.close', { text: 'Close', onclick: () => this.close() })),
        body,
      );
      this.el.classList.add('open');
      let detail;
      try {
        detail = await run.source.node(run.id, nodeId);
      } catch (err) {
        AM.clear(body).append(h('div.error-box', `This node could not be loaded (${err.message}).`));
        return;
      }
      if (this.current !== nodeId) return;
      const node = detail.node;
      const graph = await run.source.graph(run.id);
      const parentDelta = detail.verdict && detail.verdict.delta != null ? detail.verdict.delta : null;
      const metric = (graph.meta && graph.meta.scoring && graph.meta.scoring.formula) || 'primary value';
      AM.clear(this.el).append(
        h('div.drawer-head', h('h2', nodeId), h('span.small', h('span.status-dot', { class: AM.statusClass(node.status) }), AM.statusLabel(node.status)),
          h('button.quiet.close', { text: 'Close', onclick: () => this.close() })),
        h('div.drawer-body',
          node.type === 'executed' ? h('div.big-number', fmt.num(node.primary_value), h('small', metric + (parentDelta != null ? `, ${fmt.delta(parentDelta)} vs parent` : ''))) : null,
          h('p', { style: { marginTop: '8px' } }, node.description || ''),
          node.kind ? h('div.small.muted', [node.kind, node.tier ? `tier ${node.tier}` : null, node.techniques && node.techniques.length ? node.techniques.join(', ') : null].filter(Boolean).join(' · ')) : null,
          ...section('Decision', verdictBox(detail)),
          ...section('Validation metrics', metricsList(node.metrics)),
          foldTable(detail) ? h('div', ...section('Per fold', foldTable(detail))) : null,
          ...section('Timing', timing(detail)),
          ...section('Files changed', filesList(run, detail)),
          ...section('Run log', logTail(detail)),
          ...section('In the transcript', agentLinks(run, detail)),
          ...section('Lineage', h('div.small', AM.lineage(graph.nodes, nodeId).map((n, i, arr) => [
            h('span.tag.link', { text: n.id, onclick: () => this.open(run, n.id, opts) }), i < arr.length - 1 ? ' → ' : '',
          ]))),
          node.type === 'executed' ? h('div.small', { style: { marginTop: '16px' } }, h('a', { href: AM.runHref(run.key, 'tree') + `/node/${nodeId}`.replace('/tree/node', '/node'), text: 'Show in the tree' })) : null,
        ),
      );
      if (opts.onVerdict) {
        const v = detail.verdict;
        opts.onVerdict(v && v.parent_id ? { node_id: nodeId, parent_value: v.parent_primary_value, bar: v.bar } : null);
      }
    },
    close() {
      if (!this.el) return;
      this.el.classList.remove('open');
      this.current = null;
    },
  };

  AM.nodeDrawer = drawer;
})(window.AM);
