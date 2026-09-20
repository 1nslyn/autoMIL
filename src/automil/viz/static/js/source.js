/* A record source: the bundled snapshot, a live server, or the inlined single-file record. */
(function (AM) {
  'use strict';

  class RecordSource {
    /* kind: 'snapshot' | 'live' | 'inline'; base: origin or relative root without trailing slash. */
    constructor(kind, base, inline) {
      this.kind = kind;
      this.base = base;
      this.inline = inline || null;
      this.cache = new Map();
      this.listeners = new Set();
      this.events = null;
      this.connection = kind === 'live' ? 'offline' : 'recorded';
      this.prefix = kind === 'inline' ? '' : `${base}/record/`;
    }

    async get(rel) {
      if (this.cache.has(rel)) return this.cache.get(rel);
      const promise = this.fetch(rel);
      this.cache.set(rel, promise);
      try {
        return await promise;
      } catch (err) {
        this.cache.delete(rel);
        throw err;
      }
    }

    async fetch(rel) {
      if (this.kind === 'inline') {
        if (!(rel in this.inline)) throw new Error(`${rel} is not in this file`);
        return this.inline[rel];
      }
      const options = { cache: 'no-store' };
      if (this.kind === 'live' && /^https?:/.test(this.base) && !this.base.startsWith(location.origin)) {
        options.mode = 'cors';
        try { options.targetAddressSpace = 'loopback'; } catch (e) { /* older browsers */ }
      }
      const response = await fetch(this.prefix + rel, options);
      if (!response.ok) throw new Error(`${response.status} for ${rel}`);
      return response.json();
    }

    invalidate(prefix) {
      for (const key of Array.from(this.cache.keys())) {
        if (!prefix || key.startsWith(prefix)) this.cache.delete(key);
      }
    }

    index() { return this.get('index.json'); }
    graph(run) { return this.get(`runs/${run}/graph.json`); }
    timeline(run) { return this.get(`runs/${run}/timeline.json`); }
    sessions(run) { return this.get(`runs/${run}/sessions.json`); }
    links(run) { return this.get(`runs/${run}/agent_links.json`); }
    notes(run) { return this.get(`runs/${run}/notes.json`); }
    certified(run) { return this.get(`runs/${run}/certified.json`).catch(() => null); }
    node(run, id) { return this.get(`runs/${run}/nodes/${id}.json`); }
    file(run, id, path) { return this.get(`runs/${run}/nodes/${id}/files/${path}.json`); }
    chunk(run, sid, k) { return this.get(`runs/${run}/sessions/${sid}/turns/${k}.json`); }
    fullResult(run, sid, tool) { return this.get(`runs/${run}/sessions/${sid}/results/${tool}.json`); }
    agent(run, sid, agent) { return this.get(`runs/${run}/sessions/${sid}/agents/${agent}.json`); }

    /* Live only: one EventSource shared by every view; listeners get parsed frames. */
    subscribe(listener) {
      this.listeners.add(listener);
      if (this.kind === 'live' && !this.events) this.connect();
      return () => this.listeners.delete(listener);
    }

    connect() {
      const url = `${this.base}/events`;
      let source;
      try {
        source = new EventSource(url);
      } catch (err) {
        this.connection = 'offline';
        this.emit({ type: 'connection', state: 'offline' });
        return;
      }
      this.events = source;
      source.onopen = () => {
        this.connection = 'live';
        this.emit({ type: 'connection', state: 'live' });
      };
      source.onmessage = (event) => {
        let frame;
        try { frame = JSON.parse(event.data); } catch (err) { return; }
        this.applyFrame(frame);
        this.emit(frame);
      };
      source.onerror = () => {
        this.connection = 'offline';
        this.emit({ type: 'connection', state: 'offline' });
      };
    }

    applyFrame(frame) {
      const run = frame.run_id;
      if (frame.type === 'graph_update') {
        this.invalidate('index.json');
        this.invalidate('runs/');
        if (frame.full_graph && run) this.cache.set(`runs/${run}/graph.json`, Promise.resolve(Object.assign({ run_id: run }, frame.full_graph)));
      } else if (frame.type === 'sessions_update') {
        this.invalidate('index.json');
        if (run) {
          this.invalidate(`runs/${run}/sessions.json`);
          this.invalidate(`runs/${run}/timeline.json`);
          this.invalidate(`runs/${run}/agent_links.json`);
          if (frame.sessions) this.cache.set(`runs/${run}/sessions.json`, Promise.resolve(frame.sessions));
        }
      } else if (frame.type === 'transcript_delta' || frame.type === 'transcript_invalidate') {
        if (run) {
          this.invalidate(`runs/${run}/sessions/${frame.session_id}/`);
          this.invalidate(`runs/${run}/sessions.json`);
          this.invalidate(`runs/${run}/agent_links.json`);
          this.invalidate(`runs/${run}/timeline.json`);
          this.invalidate(`runs/${run}/nodes/`);
        }
      }
    }

    emit(frame) {
      for (const fn of this.listeners) {
        try { fn(frame); } catch (err) { console.error('listener failed', err); }
      }
    }

    close() {
      if (this.events) this.events.close();
      this.events = null;
    }
  }

  /* The record this page was published with: inline map, else ./record next to the page. */
  async function openBundled() {
    const inline = document.getElementById('automil-record');
    if (inline) {
      let map;
      try { map = JSON.parse(inline.textContent); } catch (err) { map = null; }
      if (map && map['index.json']) return new RecordSource('inline', '', map);
    }
    const probe = new RecordSource('snapshot', '.');
    const index = await probe.index();
    if (index.mode === 'live') {
      const live = new RecordSource('live', location.origin + location.pathname.replace(/\/[^/]*$/, '').replace(/\/$/, ''));
      live.cache.set('index.json', Promise.resolve(index));
      return live;
    }
    return probe;
  }

  function openLocal(origin) {
    const base = String(origin || '').replace(/\/+$/, '');
    if (!/^https?:\/\//.test(base)) throw new Error('enter the server address as http://host:port');
    return new RecordSource('live', base);
  }

  AM.RecordSource = RecordSource;
  AM.openBundled = openBundled;
  AM.openLocal = openLocal;
})(window.AM);
