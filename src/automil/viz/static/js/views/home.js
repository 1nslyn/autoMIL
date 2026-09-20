/* Home: what autoMIL is, how a run works, what it leaves behind, the benchmark, how to start. */
(function (AM) {
  'use strict';
  const { h, fmt } = AM;

  const METHODS = ['CLAM', 'nnMIL', 'ABMIL', 'DTFD'];

  async function heroVisual(source, index) {
    const run = (index.runs || [])[0];
    const box = h('div.hero-visual');
    if (!run) {
      box.append(h('div.empty', { style: { padding: '24px' } }, 'No run is recorded in this copy of the site yet.'));
      return { box, caption: null, destroy() {} };
    }
    let handle = null;
    try {
      const graph = await source.graph(run.run_id);
      if (AM.tree3d && AM.tree3d.available()) {
        const holder = h('div.gl');
        box.append(holder);
        requestAnimationFrame(() => {
          try {
            handle = AM.tree3d.renderTree3D(holder, graph, { compact: true, rotate: true });
          } catch (err) {
            AM.clear(box);
            const svg = h('svg');
            box.append(svg);
            AM.renderTree(svg, graph, { compact: true, interactive: false });
          }
        });
      } else {
        const svg = h('svg');
        box.append(svg);
        requestAnimationFrame(() => AM.renderTree(svg, graph, { compact: true, interactive: false }));
      }
    } catch (err) {
      box.append(h('div.error-box', 'The tree could not be drawn.'));
    }
    const caption = h('div.hero-caption', `${run.title || run.run_id}: ${run.n_executed} attempts, best ${fmt.num(run.best_primary_value)} from a baseline of ${fmt.num(run.baseline_primary_value)}. `,
      h('a', { href: AM.runHref(run.run_id, 'tree'), text: 'Open this run' }));
    return { box, caption, destroy() { if (handle) handle.destroy(); } };
  }

  async function excerpt(source, index) {
    const run = (index.runs || [])[0];
    if (!run || !run.n_sessions) return h('div.muted.small', 'No session is recorded in this copy of the site.');
    try {
      const sessions = await source.sessions(run.run_id);
      const session = sessions.sessions[0];
      const chunk = await source.chunk(run.run_id, session.session_id, 0);
      const assistant = chunk.turns.filter((t) => t.kind === 'assistant');
      const first = assistant.find((t) => t.text.length > 240 && t.tool_calls.some((c) => c.automil && c.automil.created.length))
        || assistant.find((t) => t.text.length > 400) || assistant[0] || chunk.turns[0];
      const box = h('div.excerpt');
      box.append(h('div.turn-head', h('span.who', 'Agent'), h('span.idx', `turn #${first.index} of ${chunk.n_turns}`), h('span', fmt.when(first.at))));
      if (first.text) box.append(AM.markdown.element(fmt.truncate(first.text, 1100)));
      const calls = first.tool_calls || [];
      const creations = calls.filter((c) => c.automil && c.automil.created.length);
      if (creations.length) {
        box.append(h('div.small.muted', { style: { marginTop: '8px' } }, 'This turn created ',
          ...creations.flatMap((c) => c.automil.created.map((cr) => h('span.tag', { text: cr.node_id, style: { marginRight: '4px' } })))));
      } else if (calls.length) {
        box.append(h('div.small.muted', { style: { marginTop: '8px' } }, `${calls.length} tool call${calls.length > 1 ? 's' : ''} in this turn: ${calls.slice(0, 3).map((c) => c.name).join(', ')}${calls.length > 3 ? ', \u2026' : ''}`));
      }
      box.append(h('p.small', { style: { marginTop: '12px', marginBottom: 0 } }, h('a', { href: `#/run/${encodeURIComponent(run.run_id)}/session/${session.session_id}/turn/${first.index}`, text: 'Read this session from the start' })));
      return box;
    } catch (err) {
      return h('div.muted.small', 'The transcript could not be loaded.');
    }
  }

  function lane(kind, title, actor, steps) {
    return h('div.lane-row', { class: kind },
      h('div', h('h3', title), h('div.actor', actor)),
      h('div.lane-steps', steps.map(([head, text]) => h('div.lane-step', h('b', head), text))),
    );
  }

  function lanesStrip() {
    return h('div.lanes-strip',
      lane('research', 'Research', 'the agent',
        [['Read', 'the code base, the tree and its own notes'], ['Propose', 'a change under the current best node, with its kind declared'], ['Submit', 'the changed files; the agent runs nothing itself']]),
      lane('execution', 'Execution', 'the orchestrator',
        [['Queue', 'nodes leave in priority order, as many at once as GPU memory allows'], ['Isolate', 'a detached git worktree at the pinned commit, changed files on top'], ['Train', 'the script writes result.json; only its validation block is passed on']]),
      lane('evidence', 'Evidence', 'the framework',
        [['Recompute', 'the validation value from the per-fold results'], ['Compare', 'the child must beat its parent by max(δ, k · SE) on paired folds'], ['Decide', 'kept: it becomes the next parent; discarded: it stays as a record']]),
      lane('test', 'Final test', 'once, at the end',
        [['Set aside', 'test metrics go to a closed directory during the search'], ['Select', 'the search ends on validation evidence alone'], ['Reveal', 'automil certify reports the chosen node']]),
    );
  }

  const ROWS = [
    { label: 'LUAD KRAS', dataset: 'tcga_luad', task: 'kras' },
    { label: 'PDAC immune', dataset: 'cptac_pdac', task: 'immune_class' },
    { label: 'HNSC grade', dataset: 'tcga_hnsc', task: 'grade' },
    { label: 'LUAD survival', dataset: 'tcga_luad', task: 'os' },
    { label: 'HNSC survival', dataset: 'tcga_hnsc', task: 'os' },
  ];
  const ENCODER_KEYS = ['uni_v2', 'virchow2', 'hoptimus1'];
  const METHOD_KEYS = ['clam', 'nnmil', 'abmil', 'dtfd'];

  function methodKey(model) {
    const m = String(model || '').toLowerCase();
    if (m.startsWith('clam')) return 'clam';
    if (m.includes('nnmil') || m.includes('simple_mil')) return 'nnmil';
    if (m.includes('abmil')) return 'abmil';
    if (m.includes('dtfd')) return 'dtfd';
    if (m.includes('titan')) return 'titan';
    return m;
  }

  function cellGrid(index) {
    const recorded = new Set((index.runs || []).map((r) => {
      const dataset = String((r.project && r.project.name) || '').toLowerCase();
      return `${dataset}:${String(r.task || '').toLowerCase()}:${String(r.encoder || '').toLowerCase()}:${methodKey(r.mil_model)}`;
    }));
    const cols = METHOD_KEYS.length * ENCODER_KEYS.length + 1;
    const cellW = 36;
    const cellH = 26;
    const left = 118;
    const top = 58;
    const width = left + cols * cellW + 20;
    const height = top + ROWS.length * cellH + 8;
    const svg = h('svg.cell-grid', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'The 65 cells: five task and cohort pairs by thirteen method configurations' });
    const ns = 'http://www.w3.org/2000/svg';
    function text(x, y, value, cls, anchor) {
      const t = document.createElementNS(ns, 'text');
      t.setAttribute('x', x); t.setAttribute('y', y);
      if (cls) t.setAttribute('class', cls);
      if (anchor) t.setAttribute('text-anchor', anchor);
      t.textContent = value;
      svg.appendChild(t);
    }
    METHODS.forEach((m, i) => text(left + (i * ENCODER_KEYS.length + 1.5) * cellW, 18, m, 'head', 'middle'));
    const short = ['U', 'V', 'H'];
    METHOD_KEYS.forEach((m, i) => ENCODER_KEYS.forEach((e, j) => text(left + (i * ENCODER_KEYS.length + j + 0.5) * cellW, 40, short[j], null, 'middle')));
    text(left + (cols - 0.5) * cellW, 18, 'TITAN', 'head', 'middle');
    text(left + (cols - 0.5) * cellW, 40, 'slide', null, 'middle');
    ROWS.forEach((row, r) => {
      text(left - 10, top + r * cellH + cellH * 0.68, row.label, null, 'end');
      for (let col = 0; col < cols; col += 1) {
        const rect = document.createElementNS(ns, 'rect');
        rect.setAttribute('x', left + col * cellW + 3);
        rect.setAttribute('y', top + r * cellH + 3);
        rect.setAttribute('width', cellW - 6);
        rect.setAttribute('height', cellH - 6);
        rect.setAttribute('rx', 3);
        const key = col === cols - 1
          ? `${row.dataset}:${row.task}:titan:titan`
          : `${row.dataset}:${row.task}:${ENCODER_KEYS[col % ENCODER_KEYS.length]}:${METHOD_KEYS[Math.floor(col / ENCODER_KEYS.length)]}`;
        rect.setAttribute('class', 'cell' + (recorded.has(key) ? ' recorded' : ''));
        svg.appendChild(rect);
      }
    });
    return svg;
  }

  AM.views.home = async function (root) {
    const source = AM.state.sources.bundled;
    const index = source ? await source.index() : { runs: [] };
    const first = (index.runs || [])[0];
    const openRun = first ? AM.runHref(first.run_id, 'tree') : '#/runs';
    const hero = await heroVisual(source, index);

    root.append(h('div.band.hero-band', h('div.inner', h('section.hero',
      h('div',
        h('h1', 'A coding agent runs the experiment loop on your existing code base.'),
        h('p.lead', 'It proposes a change, submits it as an experiment, reads the validation result and keeps or discards it. Every attempt, decision and transcript is recorded and shown here.'),
        h('div.actions', h('a.button.primary.large', { href: openRun, text: first ? 'Open a recorded run' : 'Runs' }), h('a.button.large', { href: '#/?scroll=get-started', text: 'Install' })),
      ),
      h('div', hero.box, hero.caption),
    ))));

    root.append(h('div.band.tint', { id: 'how' }, h('div.inner',
      h('h2', 'How a run works'),
      h('p.lead', 'autoMIL overlays one directory onto an existing repository. Four lanes share the work: the agent proposes, the orchestrator runs, the framework judges, and the test set waits until the end.'),
      lanesStrip(),
      h('p.figure-link', 'The same mechanism as drawn in the preprint: ', h('a', { href: './static/img/fig1_overview.png', target: '_blank', rel: 'noopener', text: 'Figure 1' }), '.'),
    )));

    root.append(h('div.band', { id: 'record' }, h('div.inner',
      h('h2', 'What a run leaves behind'),
      h('p.lead', 'Everything on this site is read from files in the project directory, the same files the framework works from.'),
      h('div.cols-2.uneven',
        h('div.record-list',
          h('div.record-item', h('div.path', 'automil/graph.json'), h('div.what', 'The tree: every proposal and attempt, its parent, its validation value and the keep or discard status.')),
          h('div.record-item', h('div.path', 'automil/orchestrator/archive/<node>/'), h('div.what', 'The changed files, the launched spec, the validation result and the run log of one attempt.')),
          h('div.record-item', h('div.path', 'automil/sessions/<session>/'), h('div.what', 'The agent session as the runtime recorded it: every message, tool call, result and subagent.')),
          h('div.record-item', h('div.path', 'automil/plan.md, learnings.md'), h('div.what', 'What the agent planned to try and what it concluded from each batch.')),
        ),
        await excerpt(source, index),
      ),
    )));

    root.append(h('div.band.tint', { id: 'benchmark' }, h('div.inner',
      h('h2', 'The benchmark'),
      h('p.lead', 'Does the ranking of pathology MIL methods change when every method gets the same search opportunity? Four aggregators on three encoders, plus TITAN at slide level, across five task and cohort pairs, on frozen features and splits.'),
      h('div.stat-row',
        h('div.stat', h('div.value', '3'), h('div.label', 'cohorts: TCGA-LUAD, CPTAC-PDAC, TCGA-HNSC')),
        h('div.stat', h('div.value', '13'), h('div.label', 'method configurations per task')),
        h('div.stat', h('div.value', '65'), h('div.label', 'cells, each with a native and a searched arm')),
        h('div.stat', h('div.value', '30'), h('div.label', 'attempts per searched cell on three folds')),
      ),
      cellGrid(index),
      h('p.grid-note', 'Columns per aggregator: U = UNI2-h, V = Virchow2, H = H-optimus-1. Filled cells are recorded on this site. The campaign is in progress: these are rehearsal cells and no ranking result is reported. The ten best recipes of a cell are re-run on the two remaining folds; the winner is the best five-fold validation mean, with the native baseline in the pool.'),
    )));

    root.append(h('div.band', { id: 'get-started' }, h('div.inner',
      h('h2', 'Get started'),
      h('div.cols-2',
        h('ol.steps',
          h('li', 'Install the framework.', h('span.note', 'Python 3.10 or newer; uv recommended.')),
          h('li', 'Overlay it onto a project whose training script writes a result.json.', h('span.note', 'automil check validates the setup.')),
          h('li', 'Start the orchestrator and the dashboard, each in its own terminal.'),
          h('li', 'Run the agent in a third terminal and let it work.', h('span.note', 'Claude Code with the /automil skill; Codex and OpenCode runtimes are supported too.')),
          h('li', 'On a remote host, forward the dashboard port, or connect this site to it.', h('span.note', h('a', { href: '#/remote', text: 'Remote access in detail' }))),
        ),
        h('div.code-panel', { html:
          '<span class="c"># install</span>\nuv tool install git+https://github.com/leoyin1127/autoMIL.git\n\n' +
          '<span class="c"># overlay onto your project</span>\ncd /path/to/your/project\nautomil init\nautomil check\n\n' +
          '<span class="c"># run (three terminals)</span>\nautomil orchestrator start\nautomil viz start\nclaude --dangerously-skip-permissions   <span class="c"># then /automil-setup, /automil</span>\n\n' +
          '<span class="c"># from your own machine</span>\nssh -N -L 8420:127.0.0.1:8420 user@gpu-host' }),
      ),
    )));

    root.append(h('div.band.tint', { id: 'cite' }, h('div.inner',
      h('h2', 'Authors'),
      h('p.authors', 'Shuolin Yin, Yeonwoo Seo and Jun Ma. The framework is released under the Apache-2.0 licence; the code and the issue tracker are on GitHub.'),
      h('pre', '@software{automil,\n  title  = {autoMIL},\n  author = {Yin, Shuolin and Seo, Yeonwoo and Ma, Jun},\n  year   = {2026},\n  url    = {https://github.com/leoyin1127/autoMIL}\n}'),
    )));
    return () => hero.destroy();
  };
})(window.AM);
