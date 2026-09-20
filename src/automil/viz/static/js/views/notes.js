/* Notes: the plan and the learnings the agent wrote during the run. */
(function (AM) {
  'use strict';
  const { h, fmt } = AM;

  AM.views.notes = async function (root, ctx) {
    const { run } = ctx;
    const source = run.source;
    let notes = await source.notes(run.id);
    const body = h('div');
    const tabs = h('div.notes-tabs');
    let current = 'plan_md';
    const page = h('div.page.narrow',
      h('div.page-title', h('h1', 'Notes'), h('p', 'The agent keeps two files as it works: a plan of what it will try and why, and the learnings it draws from each batch of results.')),
      tabs, body,
    );
    root.append(page);
    function render() {
      AM.append(AM.clear(tabs), [
        h('button', { class: current === 'plan_md' ? 'active' : '', text: 'plan.md', onclick: () => { current = 'plan_md'; render(); } }),
        h('button', { class: current === 'learnings_md' ? 'active' : '', text: 'learnings.md', onclick: () => { current = 'learnings_md'; render(); } }),
        notes.mtime ? h('span.muted.small', { style: { marginLeft: 'auto', alignSelf: 'center' } }, `last written ${fmt.when(notes.mtime * 1000)}`) : null,
      ]);
      AM.clear(body);
      const text = notes[current];
      body.append(text ? AM.markdown.element(text) : h('div.empty', 'This file is empty or was not written.'));
    }
    render();
    const unsubscribe = source.subscribe(async (frame) => {
      if (frame.type === 'graph_update' || frame.type === 'transcript_delta') {
        source.invalidate(`runs/${run.id}/notes.json`);
        notes = await source.notes(run.id);
        render();
      }
    });
    return () => unsubscribe();
  };
})(window.AM);
