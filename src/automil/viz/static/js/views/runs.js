/* Runs: every recorded run, and the control that connects a local server. */
(function (AM) {
  'use strict';
  const { h, fmt } = AM;

  function deltaCell(entry) {
    if (entry.best_primary_value == null || entry.baseline_primary_value == null) return h('td.num', '–');
    const delta = entry.best_primary_value - entry.baseline_primary_value;
    return h('td.num', { class: delta > 0 ? 'delta-pos' : delta < 0 ? 'delta-neg' : '' }, fmt.delta(delta));
  }

  function runsTable(index, prefix) {
    const runs = index.runs || [];
    if (!runs.length) return h('div.empty', 'No runs in this record yet.');
    const rows = runs.map((entry) => {
      const key = prefix + entry.run_id;
      return h('tr.clickable', { onclick: () => AM.navigate(AM.runHref(key, 'tree')) },
        h('td', h('a', { href: AM.runHref(key, 'tree'), text: entry.title || entry.run_id })),
        h('td', [entry.task, entry.encoder, entry.mil_model].filter(Boolean).join(' / ') || '–'),
        h('td.num', String(entry.n_executed)),
        h('td.num', fmt.num(entry.baseline_primary_value)),
        h('td.num', fmt.num(entry.best_primary_value)),
        deltaCell(entry),
        h('td.num', String(entry.n_sessions) + (entry.live_sessions ? ' (live)' : '')),
        h('td', entry.campaign_phase || (entry.n_running ? `${entry.n_running} running` : entry.live_sessions ? 'in progress' : 'recorded')),
        h('td', fmt.when(entry.ended_at, { hour: undefined, minute: undefined })),
      );
    });
    return h('div.table-wrap', h('table',
      h('thead', h('tr',
        h('th', 'Run'), h('th', 'Task / encoder / model'), h('th.num', 'Attempts'), h('th.num', 'Baseline'),
        h('th.num', 'Best'), h('th.num', 'Gain'), h('th.num', 'Sessions'), h('th', 'State'), h('th', 'Last node'))),
      h('tbody', rows),
    ));
  }

  function connectPanel() {
    const saved = AM.storage('am.local') || 'http://127.0.0.1:8420';
    const input = h('input', { type: 'url', value: saved, placeholder: 'http://127.0.0.1:8420', 'aria-label': 'Local server address' });
    const status = h('div.small.muted', { id: 'local-status' });
    const list = h('div', { id: 'local-runs' });
    const button = h('button.primary', { text: 'Connect' });
    const forget = h('button.quiet', { text: 'Disconnect' });

    async function connect(origin, quiet) {
      status.textContent = 'Connecting…';
      let source;
      try {
        source = AM.openLocal(origin);
        const index = await source.index();
        AM.state.sources.local = source;
        AM.storage('am.local', source.base);
        status.textContent = `Connected to ${source.base}: ${index.runs.length} run(s).`;
        AM.clear(list).append(runsTable(index, 'local:'));
      } catch (err) {
        AM.state.sources.local = null;
        AM.clear(list);
        status.textContent = quiet
          ? `No server answered at ${origin}.`
          : `Could not read ${origin}/record/index.json (${err.message}). Is the tunnel up? See Remote access.`;
      }
    }
    button.addEventListener('click', () => connect(input.value.trim()));
    input.addEventListener('keydown', (event) => { if (event.key === 'Enter') connect(input.value.trim()); });
    forget.addEventListener('click', () => {
      AM.storage('am.local', null);
      AM.state.sources.local = null;
      status.textContent = 'Disconnected.';
      AM.clear(list);
    });
    if (AM.storage('am.local')) connect(saved, true);
    return h('section.section', { id: 'local' },
      h('h2', 'Your local run'),
      h('p', 'Start the dashboard next to your project (', h('code', 'automil viz start'), ') and, when it runs on a remote host, forward its port with SSH. Then connect this page to it: the same views read your live tree and transcript. Safari blocks this; open http://localhost:8420 directly there.'),
      h('div.connect', input, button, forget),
      status,
      list,
    );
  }

  AM.views.runs = async function (root) {
    const bundled = AM.state.sources.bundled;
    const page = h('div.page');
    root.append(page);
    page.append(h('div.page-title', h('h1', 'Runs'), h('p', 'Each run is one project: its experiment tree, the agent sessions that built it, and the files every attempt left behind.')));
    if (bundled) {
      const index = await bundled.index();
      page.append(h('section', h('h2', index.mode === 'live' ? 'This host' : 'Recorded runs'), runsTable(index, '')));
    } else {
      page.append(h('div.notice', 'No record is bundled with this page.'));
    }
    page.append(connectPanel());
  };
})(window.AM);
