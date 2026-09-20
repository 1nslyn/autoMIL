/* The lineage tree: every node under its parent, the kept lineage of the best node emphasised. */
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

  /* Draws the tree into svg; returns {update(selectedId), fit()}. */
  function renderTree(svg, graph, options) {
    const opts = Object.assign({ selected: null, onSelect: null, compact: false, interactive: true }, options || {});
    const sel = d3.select(svg).classed('tree-svg', true);
    sel.selectAll('*').remove();
    const rootLayer = sel.append('g');
    const layer = rootLayer.append('g');
    const nodes = graph.nodes || {};
    const spine = new Set(AM.lineage(nodes, graph.meta && graph.meta.best_node_id).map((n) => n.id));
    const hierarchy = buildHierarchy(graph);
    const dx = opts.compact ? 18 : 46;
    const dy = opts.compact ? 48 : 74;
    d3.tree().nodeSize([dx, dy])(hierarchy);
    const points = hierarchy.descendants().filter((d) => !d.data.virtual);
    const links = hierarchy.links().filter((l) => !l.source.data.virtual);
    const radius = opts.compact ? 4 : 6;

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
      .attr('r', radius)
      .attr('fill', (d) => AM.statusColor(d.data.status))
      .attr('stroke-dasharray', (d) => (d.data.status === 'cancelled' ? '2 2' : null));
    if (!opts.compact) {
      nodeSel.append('text').attr('dx', radius + 4).attr('dy', 4).text((d) => d.data.id.replace('node_', ''));
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

    /* zoom and fit */
    const box = svg.getBoundingClientRect();
    const width = box.width || 800;
    const height = box.height || 500;
    const xs = points.map((d) => d.x);
    const ys = points.map((d) => d.y);
    const pad = 40;
    const minX = Math.min(...xs) - pad;
    const maxX = Math.max(...xs) + pad + (opts.compact ? 0 : 40);
    const minY = Math.min(...ys) - pad;
    const maxY = Math.max(...ys) + pad;
    const scale = Math.min(1.4, Math.min(width / (maxX - minX || 1), height / (maxY - minY || 1)));
    const tx = (width - (maxX + minX) * scale) / 2;
    const ty = (height - (maxY + minY) * scale) / 2;
    const fitTransform = d3.zoomIdentity.translate(tx, ty).scale(scale);
    let zoom = null;
    if (opts.interactive) {
      zoom = d3.zoom().scaleExtent([0.2, 3]).on('zoom', (event) => {
        rootLayer.attr('transform', event.transform);
        sel.classed('labels-off', event.transform.k < 0.75);
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
    };
  }

  function legend() {
    const items = [['keep', 'kept'], ['discard', 'discarded'], ['pending', 'proposed'], ['running', 'running'], ['crash', 'crashed'], ['candidate', 'gate stage']];
    return h('div.legend', items.map(([status, label]) => h('span', h('span.status-dot', { class: AM.statusClass(status) }), label)),
      h('span', h('span', { style: { display: 'inline-block', width: '18px', borderTop: '2px solid var(--accent)', verticalAlign: 'middle', marginRight: '6px' } }), 'lineage of the best node'));
  }

  AM.views.tree = async function (root, ctx) {
    const { run } = ctx;
    const source = run.source;
    let graph = await source.graph(run.id);
    let selected = ctx.route.params.node || null;
    const treeSvg = h('svg', { role: 'img', 'aria-label': 'Experiment tree' });
    const chartSvg = h('svg', { role: 'img', 'aria-label': 'Primary value by attempt' });
    const fitButton = h('button.quiet', { text: 'Fit' });
    const counts = h('span.muted');
    const page = h('div.page.wide', h('div.workspace',
      h('div.pane', h('div.pane-head', h('h3', 'Experiment tree'), counts, h('span.spacer'), fitButton), h('div.pane-body', treeSvg), legend()),
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
    function draw(keepTransform) {
      const transform = keepTransform && tree ? tree.transform() : null;
      tree = renderTree(treeSvg, graph, { selected, onSelect: (id) => select(id, false), transform });
      chart = AM.progressChart(chartSvg, graph, { selected, onSelect: (id) => select(id, true) });
      document.getElementById('chart-metric').textContent = (graph.meta && graph.meta.scoring && graph.meta.scoring.formula) || '';
      summarize(graph);
    }
    draw(false);
    fitButton.addEventListener('click', () => tree && tree.fit());
    if (selected) select(selected, true);

    const onResize = () => draw(true);
    window.addEventListener('resize', onResize);
    window.addEventListener('am:theme', onResize);
    const unsubscribe = source.subscribe(async (frame) => {
      if (frame.type === 'graph_update') {
        graph = await source.graph(run.id);
        draw(true);
      }
    });
    return () => {
      window.removeEventListener('resize', onResize);
      window.removeEventListener('am:theme', onResize);
      unsubscribe();
      AM.nodeDrawer.close();
    };
  };

  AM.renderTree = renderTree;
})(window.AM);
