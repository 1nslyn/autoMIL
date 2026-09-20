/* The tree view: the 3D tree by default, the flat lineage tree on request, beside the value chart. */
(function (AM) {
  'use strict';
  const { h, esc, fmt } = AM;

  function buildHierarchy(graph) {
    const nodes = graph.nodes || {};
    const children = new Map();
    const roots = [];
    for (const node of Object.values(nodes)) {
      const parent = node.parent_id && nodes[node.parent_id] ? node.parent_id : null;
      if (parent) {
        if (!children.has(parent)) children.set(parent, []);
        children.get(parent).push(node);
      } else {
        roots.push(node);
      }
    }
    const order = (a, b) => (a.created_at || '').localeCompare(b.created_at || '') || a.id.localeCompare(b.id);
    for (const list of children.values()) list.sort(order);
    roots.sort(order);
    const virtual = { id: '__root__', children: roots, virtual: true };
    return d3.hierarchy(virtual, (d) => (d.virtual ? d.children : children.get(d.id) || []));
  }

  /* The flat lineage tree in an svg; returns {update, fit, focus, transform, destroy}. */
  function renderTree(svg, graph, options) {
    const opts = Object.assign({ selected: null, onSelect: null, compact: false, interactive: true }, options || {});
    const sel = d3.select(svg).classed('tree-svg', true);
    sel.selectAll('*').remove();
    const rootLayer = sel.append('g');
    const layer = rootLayer.append('g');
    const nodes = graph.nodes || {};
    const spine = new Set(AM.lineage(nodes, graph.meta && graph.meta.best_node_id).map((n) => n.id));
    const hierarchy = buildHierarchy(graph);
    const dx = opts.compact ? 18 : 50;
    const dy = opts.compact ? 48 : 84;
    d3.tree().nodeSize([dx, dy])(hierarchy);
    const points = hierarchy.descendants().filter((d) => !d.data.virtual);
    const links = hierarchy.links().filter((l) => !l.source.data.virtual);
    const radius = opts.compact ? 4 : 8;

    layer.selectAll('path.link').data(links).enter().append('path')
      .attr('class', (l) => 'link' + (spine.has(l.source.data.id) && spine.has(l.target.data.id) ? ' spine' : ''))
      .attr('d', (l) => {
        const midY = (l.source.y + l.target.y) / 2;
        return `M${l.source.x},${l.source.y}V${midY}H${l.target.x}V${l.target.y}`;
      });

    const nodeSel = layer.selectAll('g.node').data(points, (d) => d.data.id).enter().append('g')
      .attr('class', (d) => 'node' + (spine.has(d.data.id) ? ' spine' : ''))
      .attr('transform', (d) => `translate(${d.x},${d.y})`);
    nodeSel.append('circle')
      .attr('r', (d) => (d.data.id === (graph.meta && graph.meta.best_node_id) ? radius + 3 : radius))
      .attr('fill', (d) => AM.statusColor(d.data.status))
      .attr('stroke-dasharray', (d) => (d.data.status === 'cancelled' ? '2 2' : null));
    if (!opts.compact) {
      nodeSel.append('text').attr('dx', radius + 5).attr('dy', 4).text((d) => d.data.id.replace('node_', ''));
    }
    if (opts.interactive) {
      nodeSel.style('cursor', 'pointer')
        .on('click', (event, d) => { if (opts.onSelect) opts.onSelect(d.data.id); })
        .on('mousemove', (event, d) => {
          const n = d.data;
          AM.tooltip.show(
            `<span class="mono">${esc(n.id)}</span> ${esc(AM.statusLabel(n.status))}<br>${esc(fmt.truncate(n.description, 90))}` +
            (n.primary_value != null && n.type === 'executed' ? `<br>${esc(fmt.num(n.primary_value))}` : ''),
            event.clientX, event.clientY,
          );
        })
        .on('mouseleave', () => AM.tooltip.hide());
    }

    function update(selectedId) {
      nodeSel.classed('selected', (d) => d.data.id === selectedId);
    }
    update(opts.selected);

    const box = svg.getBoundingClientRect();
    const width = box.width || 800;
    const height = box.height || 500;
    const xs = points.map((d) => d.x);
    const ys = points.map((d) => d.y);
    const pad = 48;
    const minX = Math.min(...xs) - pad;
    const maxX = Math.max(...xs) + pad + (opts.compact ? 0 : 44);
    const minY = Math.min(...ys) - pad;
    const maxY = Math.max(...ys) + pad;
    const scale = Math.min(1.5, Math.min(width / (maxX - minX || 1), height / (maxY - minY || 1)));
    const tx = (width - (maxX + minX) * scale) / 2;
    const ty = (height - (maxY + minY) * scale) / 2;
    const fitTransform = d3.zoomIdentity.translate(tx, ty).scale(scale);
    let zoom = null;
    if (opts.interactive) {
      zoom = d3.zoom().scaleExtent([0.2, 3]).on('zoom', (event) => {
        rootLayer.attr('transform', event.transform);
        sel.classed('labels-off', event.transform.k < 0.7);
      });
      sel.call(zoom).on('dblclick.zoom', null);
      sel.call(zoom.transform, opts.transform || fitTransform);
    } else {
      rootLayer.attr('transform', fitTransform);
    }
    return {
      update,
      fit() { if (zoom) sel.transition().duration(300).call(zoom.transform, fitTransform); },
      transform() { return d3.zoomTransform(svg); },
      focus(id) {
        const point = points.find((d) => d.data.id === id);
        if (!point || !zoom) return;
        const t = d3.zoomTransform(svg);
        const target = d3.zoomIdentity.translate(width / 2 - point.x * t.k, height / 2 - point.y * t.k).scale(t.k);
        sel.transition().duration(300).call(zoom.transform, target);
      },
      destroy() { sel.selectAll('*').remove(); },
    };
  }

  function legend() {
    const items = [['keep', 'kept'], ['discard', 'discarded'], ['pending', 'proposed'], ['running', 'running'], ['crash', 'crashed'], ['candidate', 'gate stage']];
    return h('div.legend', items.map(([status, label]) => h('span', h('span.status-dot', { class: AM.statusClass(status) }), label)),
      h('span', h('span', { style: { display: 'inline-block', width: '18px', borderTop: '3px solid var(--accent)', verticalAlign: 'middle', marginRight: '6px' } }), 'lineage of the best node'));
  }

  function preferredMode() {
    const saved = AM.storage('am.tree');
    if (saved === '2d') return '2d';
    return AM.tree3d && AM.tree3d.available() ? '3d' : '2d';
  }

  AM.views.tree = async function (root, ctx) {
    const { run } = ctx;
    const source = run.source;
    let graph = await source.graph(run.id);
    let selected = ctx.route.params.node || null;
    let mode = preferredMode();
    const body = h('div.pane-body');
    const chartSvg = h('svg', { role: 'img', 'aria-label': 'Primary value by attempt' });
    const fitButton = h('button.quiet', { text: 'Fit' });
    const counts = h('span.muted');
    const toggle = h('div.segmented');
    const page = h('div.page.wide', h('div.workspace',
      h('div.pane', h('div.pane-head', h('h3', 'Experiment tree'), counts, h('span.spacer'), toggle, fitButton), body, legend()),
      h('div.pane', h('div.pane-head', h('h3', 'Primary value by attempt'), h('span.muted', { id: 'chart-metric' })), h('div.pane-body', chartSvg)),
    ));
    root.append(page);

    let tree = null;
    let chart = null;
    function summarize(g) {
      const all = Object.values(g.nodes);
      const executed = all.filter((n) => n.type === 'executed');
      const kept = executed.filter((n) => ['keep', 'candidate', 'registered'].includes(n.status)).length;
      counts.textContent = `${executed.length} attempts, ${kept} kept, ${all.length - executed.length} proposed`;
    }
    function select(id, focus) {
      selected = id;
      if (tree) tree.update(id);
      if (chart) chart.update(id);
      if (id) {
        AM.nodeDrawer.open(run, id, { onVerdict: (band) => chart && chart.setBand(band) });
        if (focus && tree) tree.focus(id);
      }
    }
    function drawTree(keepTransform) {
      const transform = keepTransform && tree && mode === '2d' && tree.transform ? tree.transform() : null;
      if (tree) tree.destroy();
      AM.clear(body);
      if (mode === '3d' && AM.tree3d && AM.tree3d.available()) {
        const holder = h('div.gl');
        body.append(holder);
        try {
          tree = AM.tree3d.renderTree3D(holder, graph, { selected, onSelect: (id) => select(id, false) });
        } catch (err) {
          console.error('3D tree failed, falling back to the flat tree', err);
          mode = '2d';
          AM.clear(body);
        }
      }
      if (mode === '2d') {
        const svg = h('svg', { role: 'img', 'aria-label': 'Experiment tree' });
        body.append(svg);
        tree = renderTree(svg, graph, { selected, onSelect: (id) => select(id, false), transform });
      }
      renderToggle();
    }
    function renderToggle() {
      AM.clear(toggle);
      if (!(AM.tree3d && AM.tree3d.available())) return;
      for (const [value, label] of [['3d', '3D'], ['2d', 'Lineage']]) {
        toggle.append(h('button', { class: mode === value ? 'active' : '', text: label, onclick: () => {
          if (mode === value) return;
          mode = value;
          AM.storage('am.tree', value);
          drawTree(false);
        } }));
      }
    }
    function drawChart() {
      chart = AM.progressChart(chartSvg, graph, { selected, onSelect: (id) => select(id, true) });
      document.getElementById('chart-metric').textContent = (graph.meta && graph.meta.scoring && graph.meta.scoring.formula) || '';
      summarize(graph);
    }
    drawTree(false);
    drawChart();
    fitButton.addEventListener('click', () => tree && tree.fit());
    if (selected) select(selected, true);

    const onResize = () => { if (mode === '2d') drawTree(true); drawChart(); };
    const onTheme = () => { if (tree && tree.setTheme) tree.setTheme(); else drawTree(true); drawChart(); };
    window.addEventListener('resize', onResize);
    window.addEventListener('am:theme', onTheme);
    const unsubscribe = source.subscribe(async (frame) => {
      if (frame.type === 'graph_update') {
        graph = await source.graph(run.id);
        drawTree(true);
        drawChart();
      }
    });
    return () => {
      window.removeEventListener('resize', onResize);
      window.removeEventListener('am:theme', onTheme);
      unsubscribe();
      if (tree) tree.destroy();
      AM.nodeDrawer.close();
    };
  };

  AM.renderTree = renderTree;
})(window.AM);
