/* Primary value by attempt: each executed node in submission order, the best value so far, the baseline. */
(function (AM) {
  'use strict';
  const { esc, fmt } = AM;
  const KEEP_CLASS = new Set(['keep', 'candidate', 'registered']);
  const FAILED = new Set(['crash', 'oom', 'timeout', 'cancelled', 'partial']);

  function progressChart(svg, graph, options) {
    const opts = Object.assign({ selected: null, onSelect: null }, options || {});
    const sel = d3.select(svg).classed('chart-svg', true);
    sel.selectAll('*').remove();
    const box = svg.getBoundingClientRect();
    const width = box.width || 600;
    const height = box.height || 400;
    const margin = { top: 16, right: 18, bottom: 40, left: 52 };
    const innerW = Math.max(60, width - margin.left - margin.right);
    const innerH = Math.max(60, height - margin.top - margin.bottom);
    const g = sel.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const ordered = AM.executedInOrder(graph);
    if (!ordered.length) {
      sel.append('text').attr('x', width / 2).attr('y', height / 2).attr('text-anchor', 'middle').text('no executed attempt yet');
      return { update() {}, setBand() {} };
    }
    const scored = ordered.filter((n) => !FAILED.has(n.status) && n.primary_value != null);
    const values = scored.map((n) => n.primary_value);
    const baseline = ordered.find((n) => !n.parent_id);
    const baseValue = baseline && baseline.primary_value != null ? baseline.primary_value : null;
    if (baseValue != null) values.push(baseValue);
    let [lo, hi] = values.length ? d3.extent(values) : [0, 1];
    if (lo === hi) { lo -= 0.01; hi += 0.01; }
    const padY = (hi - lo) * 0.12;
    const x = d3.scalePoint().domain(ordered.map((_, i) => i)).range([0, innerW]).padding(0.6);
    const y = d3.scaleLinear().domain([lo - padY, hi + padY]).range([innerH, 0]).nice();
    const failedY = innerH + 14;

    g.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).enter().append('line')
      .attr('x1', 0).attr('x2', innerW).attr('y1', (d) => y(d)).attr('y2', (d) => y(d));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3f')).tickSize(0)).select('.domain').remove();
    const tickEvery = ordered.length > 40 ? 10 : ordered.length > 16 ? 5 : 1;
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickValues(x.domain().filter((i) => i % tickEvery === 0 || i === ordered.length - 1)).tickFormat((i) => String(i + 1)).tickSize(3));
    g.append('text').attr('class', 'label').attr('x', innerW).attr('y', innerH + 32).attr('text-anchor', 'end').text('attempt, in submission order');

    if (baseValue != null) {
      g.append('line').attr('class', 'baseline').attr('x1', 0).attr('x2', innerW).attr('y1', y(baseValue)).attr('y2', y(baseValue));
      g.append('text').attr('x', innerW).attr('y', y(baseValue) - 4).attr('text-anchor', 'end').text('baseline');
    }

    /* best kept value so far, as a step line */
    let best = -Infinity;
    const steps = [];
    ordered.forEach((n, i) => {
      if (KEEP_CLASS.has(n.status) && n.primary_value != null && n.primary_value > best) best = n.primary_value;
      if (best > -Infinity) steps.push([i, best]);
    });
    if (steps.length) {
      const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveStepAfter);
      g.append('path').attr('class', 'best').attr('d', line(steps));
    }

    const band = g.append('rect').attr('class', 'band').attr('width', 0).attr('height', 0);

    const points = g.selectAll('g.pt').data(ordered).enter().append('g').attr('class', 'pt');
    points.filter((n) => !FAILED.has(n.status) && n.primary_value != null).append('circle')
      .attr('class', 'point')
      .attr('cx', (n, i) => x(ordered.indexOf(n)))
      .attr('cy', (n) => y(n.primary_value))
      .attr('r', 6)
      .attr('fill', (n) => AM.statusColor(n.status))
      .classed('kept', (n) => KEEP_CLASS.has(n.status));
    points.filter((n) => FAILED.has(n.status) || n.primary_value == null).append('path')
      .attr('class', 'point')
      .attr('d', 'M-5,-5L5,5M-5,5L5,-5')
      .attr('transform', (n) => `translate(${x(ordered.indexOf(n))},${failedY})`)
      .attr('stroke', (n) => AM.statusColor(n.status))
      .attr('stroke-width', 1.5);
    g.append('text').attr('x', -6).attr('y', failedY + 4).attr('text-anchor', 'end').text('failed');

    points.style('cursor', 'pointer')
      .on('click', (event, n) => { if (opts.onSelect) opts.onSelect(n.id); })
      .on('mousemove', (event, n) => {
        const parent = n.parent_id ? graph.nodes[n.parent_id] : null;
        const delta = parent && parent.primary_value != null && n.primary_value != null ? n.primary_value - parent.primary_value : null;
        AM.tooltip.show(
          `<span class="mono">${esc(n.id)}</span> ${esc(AM.statusLabel(n.status))}<br>${esc(fmt.truncate(n.description, 90))}<br>` +
          `${esc(fmt.num(n.primary_value))}${delta != null ? ' (' + esc(fmt.delta(delta)) + ' vs parent)' : ''}`,
          event.clientX, event.clientY,
        );
      })
      .on('mouseleave', () => AM.tooltip.hide());

    function update(selectedId) {
      points.selectAll('.point').classed('selected', function () {
        return d3.select(this.parentNode).datum().id === selectedId;
      });
      if (!selectedId) band.attr('width', 0);
    }
    function setBand(info) {
      /* info: {node_id, parent_value, bar} from the node's verdict */
      if (!info || info.parent_value == null || info.bar == null) { band.attr('width', 0); return; }
      const index = ordered.findIndex((n) => n.id === info.node_id);
      if (index < 0) { band.attr('width', 0); return; }
      const half = Math.max(6, x.step() / 2);
      const top = y(info.parent_value + info.bar);
      const bottom = y(info.parent_value);
      band.attr('x', x(index) - half).attr('width', half * 2).attr('y', Math.min(top, bottom)).attr('height', Math.abs(bottom - top));
    }
    update(opts.selected);
    return { update, setBand };
  }

  AM.progressChart = progressChart;
})(window.AM);
