/* Boot: open the bundled record, restore a connected local server, route views. */
(function (AM) {
  'use strict';

  const { h, esc, fmt } = AM;
  let cleanup = null;
  let unsubscribe = null;

  function setConnection(state, label) {
    const el = document.getElementById('conn');
    el.className = 'conn ' + state;
    document.getElementById('conn-text').textContent = label;
  }

  /* Which source serves a run id: bundled runs, or the connected local server for local:<id>. */
  function resolveRun(runId) {
    if (runId.startsWith('local:')) {
      const source = AM.state.sources.local;
      return { source, id: runId.slice(6), local: true };
    }
    return { source: AM.state.sources.bundled, id: runId, local: false };
  }

  async function findRun(runId) {
    const resolved = resolveRun(runId);
    if (!resolved.source) return null;
    const index = await resolved.source.index();
    const entry = (index.runs || []).find((r) => r.run_id === resolved.id);
    if (!entry) return null;
    return Object.assign(resolved, { entry, index, key: runId });
  }

  function renderRunbar(run, view) {
    const bar = document.getElementById('runbar');
    AM.clear(bar);
    if (!run) {
      bar.hidden = true;
      return;
    }
    bar.hidden = false;
    const e = run.entry;
    const meta = [e.task, e.encoder, e.mil_model].filter(Boolean).join(' / ');
    AM.append(bar, [
      h('span.run-title', { text: e.title || e.run_id }),
      h('span.run-meta', { text: meta ? `${meta} · ${e.n_executed} attempts` : `${e.n_executed} attempts` }),
      run.local ? h('span.tag', { text: 'local server' }) : null,
      h('span.tabs',
        ...[['tree', 'Tree'], ['timeline', 'Timeline'], ['agent', 'Agent'], ['nodes', 'Nodes'], ['notes', 'Notes']].map(([name, label]) =>
          h('a', { href: AM.runHref(run.key, name), class: name === view ? 'active' : '', text: label }),
        ),
      ),
    ]);
  }

  function markNav(name) {
    for (const link of document.querySelectorAll('#nav a[data-route]')) {
      link.classList.toggle('active', link.dataset.route === name);
    }
  }

  async function route() {
    const parsed = AM.parseRoute(location.hash);
    const app = document.getElementById('app');
    if (cleanup) { try { cleanup(); } catch (err) { console.error(err); } cleanup = null; }
    if (unsubscribe) { unsubscribe(); unsubscribe = null; }
    AM.setState({ route: parsed });
    markNav(parsed.name);
    let run = null;
    if (parsed.params.run) {
      try {
        run = await findRun(parsed.params.run);
      } catch (err) {
        console.error(err);
      }
      if (!run) {
        renderRunbar(null);
        AM.clear(app).append(h('div.page', h('div.error-box', { text: `No run named ${parsed.params.run} in this record.` }),
          h('p', h('a', { href: '#/runs', text: 'All runs' }))));
        return;
      }
      renderRunbar(run, parsed.name);
      const source = run.source;
      setConnection(source.connection, source.kind === 'live' ? (source.connection === 'live' ? 'Live' : 'Connecting') : 'Recorded');
      unsubscribe = source.subscribe((frame) => {
        if (frame.type === 'connection') {
          setConnection(frame.state, frame.state === 'live' ? 'Live' : 'Offline');
        }
      });
    } else {
      renderRunbar(null);
      const bundled = AM.state.sources.bundled;
      setConnection(bundled && bundled.kind === 'live' ? bundled.connection : 'recorded',
        bundled && bundled.kind === 'live' ? (bundled.connection === 'live' ? 'Live' : 'Connecting') : 'Recorded');
      if (bundled && bundled.kind === 'live') {
        unsubscribe = bundled.subscribe((frame) => {
          if (frame.type === 'connection') setConnection(frame.state, frame.state === 'live' ? 'Live' : 'Offline');
        });
      }
    }
    const view = AM.views[parsed.name] || AM.views.home;
    AM.clear(app);
    try {
      cleanup = await view(app, { route: parsed, run }) || null;
    } catch (err) {
      console.error(err);
      app.append(h('div.page', h('div.error-box', { text: `This view failed to load: ${err.message}` })));
    }
    if (parsed.query.scroll) {
      const target = document.getElementById(parsed.query.scroll);
      if (target) target.scrollIntoView();
    }
  }

  async function boot() {
    AM.initTheme();
    let bundled = null;
    try {
      bundled = await AM.openBundled();
    } catch (err) {
      console.error(err);
    }
    AM.state.sources.bundled = bundled;
    const savedLocal = AM.storage('am.local');
    if (savedLocal) {
      try { AM.state.sources.local = AM.openLocal(savedLocal); } catch (err) { AM.storage('am.local', null); }
    }
    if (bundled) {
      const index = await bundled.index();
      AM.state.index = index;
      const note = document.getElementById('footer-note');
      if (index.mode === 'live') note.textContent = 'This page is served by automil viz on this host.';
      else if (index.generated_at) note.textContent = `Record exported ${fmt.when(index.generated_at)}.`;
    }
    window.addEventListener('hashchange', route);
    await route();
  }

  AM.resolveRun = resolveRun;
  AM.findRun = findRun;
  AM.setConnection = setConnection;
  document.addEventListener('DOMContentLoaded', boot);
})(window.AM);
