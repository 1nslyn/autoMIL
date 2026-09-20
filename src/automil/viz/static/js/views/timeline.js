/* The discovery timeline: the agent's actions, the experiment runs and the best value, on one time axis. */
(function (AM) {
  'use strict';
  const { h, esc, fmt } = AM;
  const KEEP_CLASS = new Set(['keep', 'candidate', 'registered']);
  const MARK_KINDS = ['prompt', 'propose', 'submit', 'reconcile', 'rank', 'notification', 'compact'];

  function parse(iso) {
    const t = iso ? new Date(iso).getTime() : NaN;
    return Number.isNaN(t) ? null : t;
  }

  function draw(svg, timeline, graph, run, domainOverride) {
    const sel = d3.select(svg).classed('timeline-svg', true);
    sel.selectAll('*').remove();
    const box = svg.getBoundingClientRect();
    const width = box.width || 900;
    const margin = { top: 24, right: 20, bottom: 30, left: 110 };
    const innerW = Math.max(100, width - margin.left - margin.right);
    const nodes = timeline.nodes.filter((n) => parse(n.submitted_at) || parse(n.launched_at) || parse(n.completed_at));
    nodes.sort((a, b) => (parse(a.submitted_at) || parse(a.launched_at) || 0) - (parse(b.submitted_at) || parse(b.launched_at) || 0));
    const stamps = [];
    for (const n of nodes) for (const key of ['submitted_at', 'launched_at', 'completed_at']) { const t = parse(n[key]); if (t) stamps.push(t); }
    for (const e of timeline.events) { const t = parse(e.at); if (t) stamps.push(t); }
    for (const s of timeline.sessions) for (const key of ['opened_at', 'ended_at']) { const t = parse(s[key]); if (t) stamps.push(t); }
    if (!stamps.length) {
      sel.attr('height', 80).append('text').attr('x', 20).attr('y', 40).text('no timed events yet');
      return null;
    }
    const now = Date.now();
    const fullDomain = [Math.min(...stamps), Math.max(...stamps, timeline.sessions.some((s) => s.live) ? now : 0)];
    const domain = domainOverride || fullDomain;
    const x = d3.scaleUtc().domain(domain).range([0, innerW]);

    const laneAgentH = 46;
    const rowH = Math.max(8, Math.min(14, 420 / Math.max(nodes.length, 1)));
    const laneRunsH = Math.max(40, nodes.length * rowH);
    const laneValueH = 90;
    const gap = 26;
    const height = margin.top + laneAgentH + gap + laneRunsH + gap + laneValueH + margin.bottom;
    sel.attr('height', height).attr('viewBox', `0 0 ${width} ${height}`);
    const g = sel.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const clip = sel.append('defs').append('clipPath').attr('id', 'tl-clip').append('rect').attr('width', innerW).attr('height', height);
    const body = g.append('g').attr('clip-path', 'url(#tl-clip)');

    /* axis */
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${laneAgentH + gap + laneRunsH + gap + laneValueH})`)
      .call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(innerW / 110))).tickSize(3));

    /* agent lane */
    let y0 = 0;
    g.append('text').attr('class', 'lane-label').attr('x', -10).attr('y', y0 + 14).attr('text-anchor', 'end').text('agent');
    for (const s of timeline.sessions) {
      const start = parse(s.opened_at) || parse(s.first_at);
      const end = parse(s.ended_at) || (s.live ? now : null);
      if (!start) continue;
      body.append('rect').attr('class', 'session-span').attr('x', x(start)).attr('y', y0).attr('width', Math.max(2, x(end || start) - x(start))).attr('height', 8)
        .append('title').text(`session ${s.session_id}`);
    }
    const marks = timeline.events.filter((e) => MARK_KINDS.includes(e.kind) && parse(e.at));
    const markY = { prompt: 14, propose: 24, submit: 24, reconcile: 34, rank: 34, notification: 14, compact: 34 };
    body.selectAll('rect.mark').data(marks).enter().append('rect').attr('class', 'mark')
      .attr('x', (e) => x(parse(e.at)) - 2).attr('y', (e) => y0 + markY[e.kind]).attr('width', 4).attr('height', 8)
      .attr('fill', (e) => ({ prompt: AM.cssVar('--ink'), propose: AM.cssVar('--sand'), submit: AM.cssVar('--accent'), reconcile: AM.cssVar('--cyan-light'), rank: AM.cssVar('--cyan-light'), notification: AM.cssVar('--amber-light'), compact: AM.cssVar('--faint') }[e.kind]))
      .on('mousemove', (event, e) => AM.tooltip.show(`${esc(e.kind)} ${e.node_id ? '<span class="mono">' + esc(e.node_id) + '</span>' : ''}<br>${esc(fmt.truncate(e.label, 80))}<br>${esc(fmt.when(e.at))}`, event.clientX, event.clientY))
      .on('mouseleave', () => AM.tooltip.hide())
      .on('click', (event, e) => {
        if (e.turn != null && e.session_id) AM.navigate(`#/run/${encodeURIComponent(run.key)}/session/${e.session_id}/turn/${e.turn}`);
        else if (e.node_id) AM.nodeDrawer.open(run, e.node_id);
      });

    /* runs lane */
    y0 = laneAgentH + gap;
    g.append('text').attr('class', 'lane-label').attr('x', -10).attr('y', y0 + 12).attr('text-anchor', 'end').text('experiments');
    const rows = body.selectAll('g.row').data(nodes).enter().append('g').attr('class', 'row').attr('transform', (n, i) => `translate(0,${y0 + i * rowH})`);
    rows.filter((n) => parse(n.submitted_at) && (parse(n.launched_at) || parse(n.completed_at))).append('rect').attr('class', 'bar queued')
      .attr('x', (n) => x(parse(n.submitted_at))).attr('y', rowH * 0.3).attr('height', rowH * 0.4)
      .attr('width', (n) => Math.max(1, x(parse(n.launched_at) || parse(n.completed_at)) - x(parse(n.submitted_at))));
    rows.filter((n) => parse(n.launched_at)).append('rect').attr('class', 'bar')
      .attr('x', (n) => x(parse(n.launched_at))).attr('y', rowH * 0.15).attr('height', rowH * 0.7)
      .attr('width', (n) => Math.max(2, x(parse(n.completed_at) || (n.status === 'running' ? now : parse(n.launched_at))) - x(parse(n.launched_at))))
      .attr('fill', (n) => AM.statusColor(n.status))
      .attr('stroke', (n) => (n.status === 'discard' ? AM.cssVar('--status-discard-stroke') : 'none'));
    rows.append('rect').attr('x', 0).attr('width', innerW).attr('height', rowH).attr('fill', 'transparent')
      .on('mousemove', (event, n) => AM.tooltip.show(`<span class="mono">${esc(n.node_id)}</span> ${esc(AM.statusLabel(n.status))}<br>queued ${esc(fmt.when(n.submitted_at))}<br>ran ${esc(fmt.duration(fmt.between(n.launched_at, n.completed_at)))}${n.primary_value != null ? '<br>' + esc(fmt.num(n.primary_value)) : ''}`, event.clientX, event.clientY))
      .on('mouseleave', () => AM.tooltip.hide())
      .on('click', (event, n) => AM.nodeDrawer.open(run, n.node_id))
      .style('cursor', 'pointer');
    if (rowH >= 11) {
      g.selectAll('text.rowlabel').data(nodes).enter().append('text').attr('x', -10).attr('y', (n, i) => y0 + i * rowH + rowH * 0.75).attr('text-anchor', 'end')
        .attr('font-size', 9).text((n) => n.node_id.replace('node_', ''));
    }

    /* value lane: best kept value at completion time */
    y0 = laneAgentH + gap + laneRunsH + gap;
    g.append('text').attr('class', 'lane-label').attr('x', -10).attr('y', y0 + 12).attr('text-anchor', 'end').text('best value');
    const done = nodes.filter((n) => parse(n.completed_at) && n.primary_value != null && !['crash', 'cancelled', 'partial'].includes(n.status))
      .sort((a, b) => parse(a.completed_at) - parse(b.completed_at));
    const values = done.map((n) => n.primary_value);
    const baseline = nodes.find((n) => !n.parent_id);
    if (baseline && baseline.primary_value != null) values.push(baseline.primary_value);
    if (values.length) {
      let [lo, hi] = d3.extent(values);
      if (lo === hi) { lo -= 0.01; hi += 0.01; }
      const y = d3.scaleLinear().domain([lo, hi]).range([y0 + laneValueH, y0 + 4]).nice();
      g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(3).tickFormat(d3.format('.3f')).tickSize(0)).select('.domain').remove();
      let best = baseline && baseline.primary_value != null ? baseline.primary_value : -Infinity;
      const steps = [];
      if (best > -Infinity) steps.push([domain[0], best]);
      for (const n of done) {
        if (KEEP_CLASS.has(n.status) && n.primary_value > best) best = n.primary_value;
        if (best > -Infinity) steps.push([parse(n.completed_at), best]);
      }
      if (steps.length) steps.push([domain[1], steps[steps.length - 1][1]]);
      body.append('path').attr('class', 'best').attr('fill', 'none').attr('stroke', AM.cssVar('--accent')).attr('stroke-width', 2)
        .attr('d', d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveStepAfter)(steps));
      body.selectAll('circle.pt').data(done).enter().append('circle').attr('class', 'point')
        .attr('cx', (n) => x(parse(n.completed_at))).attr('cy', (n) => y(n.primary_value)).attr('r', 3.5)
        .attr('fill', (n) => AM.statusColor(n.status)).attr('stroke', AM.cssVar('--paper'))
        .on('click', (event, n) => AM.nodeDrawer.open(run, n.node_id)).style('cursor', 'pointer');
    }
    return { x, fullDomain, height };
  }

  AM.views.timeline = async function (root, ctx) {
    const { run } = ctx;
    const source = run.source;
    let [timeline, graph] = await Promise.all([source.timeline(run.id), source.graph(run.id)]);
    const svg = h('svg', { role: 'img', 'aria-label': 'Discovery timeline' });
    const overview = h('svg', { style: { height: '44px' }, 'aria-label': 'Time range' });
    const reset = h('button.quiet', { text: 'Whole run' });
    const page = h('div.page.wide',
      h('div.page-title', h('h1', 'Timeline'), h('p', 'What the agent did, when each attempt waited and ran, and how the best validation value moved. Drag on the strip below to zoom; click a bar or a mark to open it.')),
      timeline.warnings && timeline.warnings.length ? h('div.notice', timeline.warnings.join(' ')) : null,
      h('div.pane', { style: { minHeight: '0' } }, h('div.pane-head', h('h3', 'Overview'), h('span.spacer'), reset), h('div', { style: { padding: '4px 12px' } }, overview)),
      h('div.pane.timeline-wrap', { style: { marginTop: '12px' } }, h('div', svg),
        h('div.timeline-legend', ...['prompt', 'propose', 'submit', 'reconcile', 'notification', 'compact'].map((k) => h('span', h('span.glyph', { class: k }), k)),
          h('span', h('span.status-dot.status-keep'), 'kept run'), h('span', h('span.status-dot.status-discard'), 'discarded run'), h('span', h('span.status-dot.status-crash'), 'crashed'))),
    );
    root.append(page);
    let domain = null;
    let brush = null;
    function redraw() {
      const result = draw(svg, timeline, graph, run, domain);
      if (!result) return;
      const box = overview.getBoundingClientRect();
      const width = box.width || 900;
      const ov = d3.select(overview).attr('viewBox', `0 0 ${width} 44`);
      ov.selectAll('*').remove();
      const ox = d3.scaleUtc().domain(result.fullDomain).range([110, width - 20]);
      ov.append('g').attr('class', 'axis').attr('transform', 'translate(0,30)').call(d3.axisBottom(ox).ticks(Math.max(3, Math.floor(width / 130))).tickSize(3));
      for (const n of timeline.nodes) {
        const s = parse(n.launched_at);
        const e = parse(n.completed_at) || (n.status === 'running' ? Date.now() : null);
        if (!s || !e) continue;
        ov.append('rect').attr('x', ox(s)).attr('y', 8).attr('width', Math.max(1, ox(e) - ox(s))).attr('height', 16).attr('fill', AM.statusColor(n.status)).attr('opacity', 0.7);
      }
      brush = d3.brushX().extent([[110, 4], [width - 20, 28]]).on('end', (event) => {
        if (!event.selection) { domain = null; redraw(); return; }
        domain = event.selection.map(ox.invert);
        draw(svg, timeline, graph, run, domain);
      });
      ov.append('g').attr('class', 'brush').call(brush);
      if (domain) ov.select('.brush').call(brush.move, domain.map(ox));
    }
    redraw();
    reset.addEventListener('click', () => { domain = null; redraw(); });
    const onResize = () => redraw();
    window.addEventListener('resize', onResize);
    window.addEventListener('am:theme', onResize);
    const unsubscribe = source.subscribe(async (frame) => {
      if (['graph_update', 'sessions_update', 'transcript_delta'].includes(frame.type)) {
        [timeline, graph] = await Promise.all([source.timeline(run.id), source.graph(run.id)]);
        redraw();
      }
    });
    return () => { window.removeEventListener('resize', onResize); window.removeEventListener('am:theme', onResize); unsubscribe(); };
  };
})(window.AM);
