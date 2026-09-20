/* The agent's sessions: every turn, tool call, result and subagent, with links to the nodes it touched. */
(function (AM) {
  'use strict';
  const { h, fmt } = AM;
  const CHUNK = 100;

  function callSummary(call) {
    const input = call.input || {};
    if (call.name === 'Bash') return input.description || String(input.command || '').split('\n')[0];
    if (input.file_path) return input.file_path;
    if (input.pattern) return input.pattern;
    if (input.description) return input.description;
    if (input.prompt) return fmt.truncate(input.prompt, 100);
    const keys = Object.keys(input);
    return keys.length ? fmt.truncate(JSON.stringify(input), 100) : '';
  }

  function inputText(call) {
    const input = call.input || {};
    if (call.name === 'Bash') return input.command || '';
    if (call.name === 'Edit' || call.name === 'Write') {
      const parts = [input.file_path];
      if (input.old_string != null) parts.push('--- old', input.old_string, '+++ new', input.new_string);
      if (input.content != null) parts.push(input.content);
      return parts.filter((p) => p != null).join('\n');
    }
    return JSON.stringify(input, null, 2);
  }

  function nodeChips(run, ids) {
    return (ids || []).map((id) => h('span.tag.link', { text: id, onclick: (event) => { event.stopPropagation(); AM.nodeDrawer.open(run, id); } }));
  }

  function renderCall(run, sid, call, ctx) {
    const result = call.result;
    const status = result ? result.status : 'pending';
    const automil = call.automil;
    const created = automil ? automil.created.map((c) => c.node_id) : [];
    const ids = created.length ? created : (automil ? automil.node_ids : []);
    const chips = ids.length ? nodeChips(run, ids.slice(0, 6)).concat(ids.length > 6 ? [h('span.tag', { text: `+${ids.length - 6}` })] : []) : [];
    const body = h('div.body');
    const details = h('details.call',
      h('summary',
        h('span.result-dot', { class: status, title: status }),
        h('span.tool', call.name),
        h('span.summary-text', { title: callSummary(call) }, callSummary(call)),
        chips.length ? h('span.chips', chips) : null,
      ),
      body,
    );
    let filled = false;
    details.addEventListener('toggle', async () => {
      if (!details.open || filled) return;
      filled = true;
      body.append(h('h5', 'input'), h('pre', inputText(call)));
      if (!result) {
        body.append(h('h5', 'result'), h('div.muted.small', 'still running'));
        return;
      }
      if (result.status === 'missing') {
        body.append(h('h5', 'result'), h('div.muted.small', 'no result was recorded (the call was interrupted or the session ended)'));
        return;
      }
      const head = h('h5', `result (${result.status}` + (result.bytes != null ? `, ${fmt.bytes(result.bytes)}` : '') + (result.images ? `, ${result.images} image(s) omitted` : '') + ')');
      const pre = h('pre', result.preview || '');
      body.append(head, pre);
      if (result.truncated) {
        const button = h('button.quiet', { text: 'Load the full result' });
        button.addEventListener('click', async () => {
          button.disabled = true;
          try {
            const full = await run.source.fullResult(run.id, sid, call.tool_use_id);
            pre.textContent = full.text + (full.truncated ? '\n…[capped at 1 MB]' : '');
            button.remove();
          } catch (err) {
            button.textContent = `Not available (${err.message})`;
          }
        });
        body.append(button);
      }
      if (result.agent_id) {
        body.append(await subagentBlock(run, sid, result.agent_id, ctx));
      }
    });
    return details;
  }

  async function subagentBlock(run, sid, agentId, ctx) {
    const box = h('details.subagent', h('summary', `Subagent ${agentId}`));
    let loaded = false;
    box.addEventListener('toggle', async () => {
      if (!box.open || loaded) return;
      loaded = true;
      try {
        const payload = await run.source.agent(run.id, sid, agentId);
        box.querySelector('summary').textContent = `Subagent: ${payload.description || payload.agent_type || agentId} (${payload.turns.length} turns)`;
        const list = h('div.turns');
        for (const turn of payload.turns) list.append(renderTurn(run, sid, turn, Object.assign({}, ctx, { nested: true })));
        box.append(list);
      } catch (err) {
        box.append(h('div.muted.small', `No transcript stored for this subagent (${err.message}).`));
      }
    });
    return box;
  }

  function renderTurn(run, sid, turn, ctx) {
    const el = h('div.turn', { class: turn.kind, id: ctx.nested ? null : `turn-${turn.index}` });
    const who = { assistant: 'Agent', human: 'Operator', notification: 'Notification', system: 'System' }[turn.kind] || turn.kind;
    const head = h('div.turn-head',
      h('span.who', who),
      h('span.idx', `#${turn.index}`),
      turn.at ? h('span', fmt.time(turn.at)) : null,
      turn.model ? h('span', turn.model) : null,
      turn.usage ? h('span', `${fmt.tokens(turn.usage.output)} out`) : null,
      ctx.nested ? null : h('a', { href: `#/run/${encodeURIComponent(run.key)}/session/${sid}/turn/${turn.index}`, text: 'link' }),
    );
    el.append(head);
    if (turn.kind === 'system') {
      const s = turn.system || {};
      if (s.subtype === 'compact_boundary') el.append(h('div', `Context compacted: ${fmt.tokens(s.pre_tokens)} tokens became ${fmt.tokens(s.post_tokens)}.`));
      else if (s.subtype === 'compact_summary') el.append(h('details', h('summary', 'Compaction summary'), AM.markdown.element(turn.text)));
      else el.append(h('div.small', fmt.truncate(turn.text, 300)));
      return el;
    }
    if (turn.kind === 'notification') {
      const note = turn.notification || {};
      el.append(h('details', h('summary', note.title || 'Task notification'), turn.text ? AM.markdown.element(turn.text) : null));
      return el;
    }
    if (turn.kind === 'human') {
      el.append(AM.markdown.element(turn.text, 'text'));
      return el;
    }
    if (turn.thinking && ctx.filters.thinking) {
      el.append(h('details.thinking-box', h('summary', 'thinking'), h('div.thinking', turn.thinking)));
    }
    if (turn.text) el.append(AM.markdown.element(turn.text, 'text'));
    if (turn.tool_calls && turn.tool_calls.length && ctx.filters.tools) {
      el.append(h('div.calls', turn.tool_calls.map((call) => renderCall(run, sid, call, ctx))));
    }
    return el;
  }

  /* A turn that only calls tools (no text worth reading) folds with its neighbours. */
  function toolOnly(turn) {
    if (turn.kind !== 'assistant') return false;
    const text = (turn.text || '').trim();
    return (turn.tool_calls || []).length > 0 && text.length < 80 && !text.includes('\n');
  }

  function foldBlock(run, sid, group, ctx) {
    const calls = group.flatMap((t) => t.tool_calls || []);
    const byTool = new Map();
    for (const c of calls) byTool.set(c.name, (byTool.get(c.name) || 0) + 1);
    const tools = Array.from(byTool.entries()).sort((a, b) => b[1] - a[1]).map(([n, k]) => `${n} \u00d7${k}`).join(', ');
    const created = [];
    for (const c of calls) for (const cr of (c.automil && c.automil.created) || []) if (!created.includes(cr.node_id)) created.push(cr.node_id);
    const errors = calls.filter((c) => c.result && c.result.status === 'error').length;
    const first = group[0];
    const last = group[group.length - 1];
    const span = first.at && last.at ? `${fmt.time(first.at)} to ${fmt.time(last.at)}` : '';
    const fold = h('details.fold', { id: `turn-${first.index}` },
      h('summary',
        h('span.caret'),
        h('span.count', `${group.length} turns, ${calls.length} tool calls`),
        h('span.tools', tools),
        span ? h('span.span', span) : null,
        errors ? h('span.tools', `${errors} failed`) : null,
        created.length ? h('span.chips', nodeChips(run, created.slice(0, 8)), created.length > 8 ? h('span.tag', { text: `+${created.length - 8}` }) : null) : null,
      ),
    );
    let filled = false;
    fold.addEventListener('toggle', () => {
      if (!fold.open || filled) return;
      filled = true;
      fold.append(h('div.turns', group.map((t) => renderTurn(run, sid, t, ctx))));
    });
    return fold;
  }

  function passes(turn, filters) {
    if (turn.kind === 'human') return filters.prompts;
    if (turn.kind === 'notification') return filters.notifications;
    if (turn.kind === 'system') return filters.system;
    if (filters.automilOnly) return turn.tool_calls.some((c) => c.automil);
    return true;
  }

  function sessionItem(run, session, active) {
    const duration = fmt.between(session.first_at, session.last_at);
    return h('a.session-item', { class: active ? 'active' : '', href: `#/run/${encodeURIComponent(run.key)}/session/${session.session_id}` },
      h('div', h('span.sid', fmt.shortId(session.session_id)), session.live ? h('span.tag', { text: 'live', style: { marginLeft: '6px' } }) : null),
      h('div.meta', `${fmt.when(session.first_at || session.opened_at)}${duration ? ', ' + fmt.duration(duration) : ''}`),
      h('div.meta', `${session.n_turns} turns, ${session.n_tool_calls} tool calls, ${session.n_automil_commands} automil commands`),
      session.usage ? h('div.meta', `${fmt.tokens(session.usage.output)} output tokens, ${fmt.tokens(session.usage.cache_read + session.usage.input)} read`) : null,
      session.source === 'missing' ? h('div.meta', { style: { color: 'var(--coral)' } }, 'transcript not stored') : null,
    );
  }

  AM.views.agent = async function (root, ctx) {
    const { run } = ctx;
    const source = run.source;
    const sessions = await source.sessions(run.id);
    const page = h('div.page.wide');
    root.append(page);
    if (!sessions.sessions.length) {
      page.append(h('div.page-title', h('h1', 'Agent')), h('div.empty', 'No agent session is recorded for this run.' + (sessions.journal_error ? ` The activity journal could not be read: ${sessions.journal_error}.` : '')));
      return;
    }
    const wanted = ctx.route.params.session;
    const session = sessions.sessions.find((s) => s.session_id === wanted) || sessions.sessions[0];
    const sid = session.session_id;
    const filters = { prompts: true, notifications: true, system: true, tools: true, thinking: true, fold: true, automilOnly: false };
    const turnsEl = h('div.turns');
    const loadMore = h('div.turn-more');
    const listEl = h('div.session-list', sessions.sessions.map((s) => sessionItem(run, s, s.session_id === sid)));
    const filterEl = h('div.filters',
      ...[['fold', 'fold runs of tool calls'], ['prompts', 'operator prompts'], ['notifications', 'notifications'], ['system', 'system lines'], ['tools', 'tool calls'], ['thinking', 'thinking'], ['automilOnly', 'only turns with automil commands']].map(([key, label]) => {
        const box = h('input', { type: 'checkbox', checked: filters[key] || null });
        box.addEventListener('change', () => { filters[key] = box.checked; redraw(); });
        return h('label', box, label);
      }),
    );
    listEl.append(filterEl);
    page.append(h('div.agent', listEl, h('div', h('div.session-head',
      h('h3', `Session ${fmt.shortId(sid)}`),
      h('span.muted.small.session-cwd', session.cwd ? `in ${session.cwd}` : ''),
      h('span.label', { style: { marginLeft: 'auto' } }, session.ended_by ? `ended, ${session.ended_by}` : session.live ? 'live' : 'recorded'),
    ), turnsEl, loadMore)));

    let turns = [];
    let openTurn = null;
    let loadedChunks = 0;
    let nChunks = session.n_chunks || 1;
    const drawn = new Set();
    function redraw() {
      AM.clear(turnsEl);
      drawn.clear();
      appendTurns(turns);
      if (openTurn) appendOpen(openTurn);
    }
    function appendTurns(list) {
      const ctx = { filters, run };
      let group = [];
      const flush = () => {
        if (!group.length) return;
        if (group.length >= 2) turnsEl.append(foldBlock(run, sid, group, ctx));
        else turnsEl.append(renderTurn(run, sid, group[0], ctx));
        group = [];
      };
      for (const turn of list) {
        if (drawn.has(turn.index) || !passes(turn, filters)) continue;
        drawn.add(turn.index);
        if (filters.fold && !filters.automilOnly && toolOnly(turn) && !(turn.tool_calls || []).some((c) => c.automil && c.automil.created.length)) {
          group.push(turn);
          continue;
        }
        flush();
        turnsEl.append(renderTurn(run, sid, turn, ctx));
      }
      flush();
    }
    let openEl = null;
    function appendOpen(turn) {
      if (openEl) openEl.remove();
      openEl = renderTurn(run, sid, turn, { filters, run });
      openEl.classList.add('open-turn');
      turnsEl.append(openEl);
    }
    async function loadChunk(k) {
      const payload = await source.chunk(run.id, sid, k);
      turns = turns.concat(payload.turns.filter((t) => !turns.some((u) => u.index === t.index)));
      if (payload.open_turn) openTurn = payload.open_turn;
      loadedChunks = Math.max(loadedChunks, k + 1);
      nChunks = Math.max(nChunks, Math.ceil(payload.n_turns / CHUNK) || 1);
      appendTurns(payload.turns);
      if (payload.open_turn) appendOpen(payload.open_turn);
      updateMore();
    }
    function updateMore() {
      AM.clear(loadMore);
      if (loadedChunks < nChunks) {
        const button = h('button', { text: `Load turns ${loadedChunks * CHUNK}–${Math.min((loadedChunks + 1) * CHUNK, nChunks * CHUNK) - 1}` });
        button.addEventListener('click', () => loadChunk(loadedChunks));
        loadMore.append(button);
      } else {
        loadMore.append(h('span.muted.small', session.live ? 'Following the session live.' : `End of the session, ${turns.length} turns.`));
      }
    }
    const target = ctx.route.params.turn;
    const firstChunk = Number.isFinite(target) ? Math.floor(target / CHUNK) : 0;
    for (let k = 0; k <= firstChunk; k += 1) await loadChunk(k);
    if (Number.isFinite(target)) {
      const el = document.getElementById(`turn-${target}`);
      if (el) { el.classList.add('highlight'); el.scrollIntoView({ block: 'start' }); }
    }
    const sentinel = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting) && loadedChunks < nChunks) loadChunk(loadedChunks);
    });
    sentinel.observe(loadMore);

    const unsubscribe = source.subscribe(async (frame) => {
      if (frame.session_id !== sid) return;
      if (frame.type === 'transcript_delta') {
        if (frame.from_turn !== turns.length) {
          const missing = Math.floor(turns.length / CHUNK);
          for (let k = missing; k < Math.ceil(frame.n_turns / CHUNK); k += 1) await loadChunk(k);
        } else {
          turns = turns.concat(frame.turns);
          appendTurns(frame.turns);
          for (const patch of frame.patches || []) {
            const turn = turns.find((t) => t.index === patch.turn);
            if (turn) turn.tool_calls[patch.call] = Object.assign({}, turn.tool_calls[patch.call], { result: patch.result });
          }
          if (frame.patches && frame.patches.length) redraw();
        }
        openTurn = frame.open_turn;
        if (openTurn) appendOpen(openTurn); else if (openEl) { openEl.remove(); openEl = null; }
        nChunks = Math.max(nChunks, frame.n_chunks || 1);
        loadedChunks = nChunks;
        updateMore();
      } else if (frame.type === 'transcript_invalidate') {
        turns = []; openTurn = null; loadedChunks = 0; AM.clear(turnsEl); drawn.clear();
        await loadChunk(0);
      }
    });
    return () => { sentinel.disconnect(); unsubscribe(); };
  };
})(window.AM);
