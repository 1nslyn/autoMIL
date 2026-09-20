/* Home: what autoMIL is, how a run works, what it leaves behind, how to start. */
(function (AM) {
  'use strict';
  const { h, fmt } = AM;

  async function heroVisual(source, index) {
    const run = (index.runs || [])[0];
    const box = h('div.hero-visual');
    if (!run) {
      box.append(h('div.empty', { style: { padding: '24px' } }, 'No run is recorded in this copy of the site yet.'));
      return box;
    }
    const svg = h('svg', { 'aria-label': 'Experiment tree of a recorded run' });
    box.append(svg, h('div.caption', `${run.title || run.run_id}: ${run.n_executed} attempts, best ${fmt.num(run.best_primary_value)} from a baseline of ${fmt.num(run.baseline_primary_value)}`));
    try {
      const graph = await source.graph(run.run_id);
      requestAnimationFrame(() => AM.renderTree(svg, graph, { compact: true, interactive: false }));
    } catch (err) {
      box.append(h('div.error-box', 'The tree could not be drawn.'));
    }
    return box;
  }

  async function excerpt(source, index) {
    const run = (index.runs || [])[0];
    if (!run || !run.n_sessions) return h('div.muted.small', 'No session is recorded in this copy of the site.');
    try {
      const sessions = await source.sessions(run.run_id);
      const session = sessions.sessions[0];
      const chunk = await source.chunk(run.run_id, session.session_id, 0);
      const first = chunk.turns.find((t) => t.kind === 'assistant' && t.tool_calls.some((c) => c.automil && c.automil.created.length)) || chunk.turns[1] || chunk.turns[0];
      const box = h('div.excerpt');
      box.append(h('div.turn-head', h('span.who', 'Agent'), h('span.idx', `turn #${first.index} of ${chunk.n_turns}`), h('span', fmt.when(first.at))));
      if (first.text) box.append(AM.markdown.element(fmt.truncate(first.text, 900)));
      const creations = first.tool_calls.filter((c) => c.automil && c.automil.created.length);
      if (creations.length) {
        box.append(h('div.small.muted', { style: { marginTop: '8px' } }, 'This turn created ',
          ...creations.flatMap((c) => c.automil.created.map((cr) => h('span.tag', { text: cr.node_id, style: { marginRight: '4px' } })))));
      }
      box.append(h('p.small', { style: { marginTop: '12px', marginBottom: 0 } }, h('a', { href: `#/run/${encodeURIComponent(run.run_id)}/session/${session.session_id}/turn/${first.index}`, text: 'Read this session from the start' })));
      return box;
    } catch (err) {
      return h('div.muted.small', 'The transcript could not be loaded.');
    }
  }

  AM.views.home = async function (root) {
    const source = AM.state.sources.bundled;
    const index = source ? await source.index() : { runs: [] };
    const first = (index.runs || [])[0];
    const openRun = first ? AM.runHref(first.run_id, 'tree') : '#/runs';
    const page = h('div.page');
    root.append(page);

    page.append(h('section.hero',
      h('div',
        h('h1', 'A coding agent runs the experiment loop of multiple instance learning on your existing code base.'),
        h('p', 'It proposes a change, submits it as an experiment, reads the validation result and keeps or discards it. Every attempt, decision and transcript is recorded.'),
        h('div.actions', h('a.button.primary', { href: openRun, text: first ? 'Open a recorded run' : 'Runs' }), h('a.button', { href: '#/?scroll=get-started', text: 'Install' })),
      ),
      await heroVisual(source, index),
    ));

    page.append(h('section.section', { id: 'how' },
      h('h2', 'How a run works'),
      h('p', 'autoMIL overlays one directory onto an existing repository. The agent reads the code, the framework runs the experiments and judges them, and the tree of attempts grows from the baseline.'),
      h('figure.figure', h('img', { src: './static/img/fig1_overview.png', alt: 'The four lanes of a run: research, execution, evidence and the final test', loading: 'lazy' }),
        h('figcaption', 'Figure 1 of the preprint: the research lane (agent), the execution lane (orchestrator), the evidence lane (framework) and the final test.')),
      h('div.lanes',
        h('div.lane.research', h('h3', 'Research'), h('div.actor', 'the agent'), h('p', 'Picks the current best node as parent, declares the kind of change, writes the changed files and submits them. It runs nothing itself.')),
        h('div.lane.execution', h('h3', 'Execution'), h('div.actor', 'the orchestrator'), h('p', 'Takes nodes from the queue as GPU memory allows, gives each a detached git worktree at the pinned commit with the changed files on top, and passes the validation block on.')),
        h('div.lane.evidence', h('h3', 'Evidence'), h('div.actor', 'the framework'), h('p', 'Recomputes the validation value from per-fold results and keeps a child only if it beats its parent by the larger of a declared margin and one paired standard error. A companion metric can veto.')),
        h('div.lane.test', h('h3', 'Final test'), h('div.actor', 'once, at the end'), h('p', 'Test metrics are written to a closed directory that no agent-facing command reads. After selection, automil certify reveals the chosen node.')),
      ),
    ));

    page.append(h('section.section', { id: 'record' },
      h('h2', 'What a run leaves behind'),
      h('p', 'Everything on this site is read from files in the project directory. The same files are what the framework itself works from.'),
      h('div.cols-2',
        h('dl.record-list',
          h('dt', 'automil/graph.json'), h('dd', 'The tree: every proposal and attempt, its parent, its validation value and the keep or discard status.'),
          h('dt', 'automil/orchestrator/archive/<node>/'), h('dd', 'The changed files, the launched spec, the validation result and the run log of one attempt.'),
          h('dt', 'automil/sessions/<session>/'), h('dd', 'The agent session as the runtime recorded it: every message, tool call, result and subagent.'),
          h('dt', 'automil/plan.md, learnings.md'), h('dd', 'What the agent planned to try and what it concluded from each batch.'),
        ),
        await excerpt(source, index),
      ),
    ));

    page.append(h('section.section', { id: 'benchmark' },
      h('h2', 'The benchmark'),
      h('p', 'The study asks whether the ranking of pathology MIL methods changes when every method gets the same search opportunity. Three cohorts (TCGA-LUAD KRAS mutation, CPTAC-PDAC immune subtype, TCGA-HNSC tumour grade), four aggregators (CLAM, nnMIL, ABMIL, DTFD-MIL) on three encoders (UNI2-h, Virchow2, H-optimus-1) plus TITAN at slide level: 65 cells on frozen features and splits.'),
      h('p', 'Each searched cell gets 30 attempts on three folds; the ten best recipes are re-run on the two remaining folds and the winner is the best five-fold validation mean, with the native baseline in the pool. The campaign is in progress. The runs recorded here are rehearsal cells, and no ranking result is reported.'),
    ));

    page.append(h('section.section', { id: 'get-started' },
      h('h2', 'Get started'),
      h('ol.steps',
        h('li', 'Install the framework.', h('pre', 'uv tool install git+https://github.com/leoyin1127/autoMIL.git')),
        h('li', 'Overlay it onto a project whose training script writes a result.json.', h('pre', 'cd /path/to/your/project\nautomil init\nautomil check')),
        h('li', 'Start the orchestrator and the dashboard, each in its own terminal.', h('pre', 'automil orchestrator start\nautomil viz start')),
        h('li', 'Run the agent in a third terminal and let it work.', h('pre', 'claude --dangerously-skip-permissions\n# in the session: /automil-setup, then /automil')),
        h('li', 'On a remote host, forward the dashboard port from your own machine, or connect this site to it.', h('pre', 'ssh -N -L 8420:127.0.0.1:8420 user@gpu-host'), h('p', h('a', { href: '#/remote', text: 'Remote access in detail' }))),
      ),
    ));

    page.append(h('section.section', { id: 'cite' },
      h('h2', 'Authors'),
      h('p', 'Shuolin Yin, Yeonwoo Seo and Jun Ma. The framework is released under the Apache-2.0 licence; the code and the issue tracker are on GitHub.'),
      h('pre', '@software{automil,\n  title  = {autoMIL},\n  author = {Yin, Shuolin and Seo, Yeonwoo and Ma, Jun},\n  year   = {2026},\n  url    = {https://github.com/leoyin1127/autoMIL}\n}'),
    ));
  };
})(window.AM);
