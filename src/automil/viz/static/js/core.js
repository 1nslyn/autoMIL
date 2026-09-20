/* autoMIL frontend core: DOM helpers, formatting, state, router, tooltip, theme. */
window.AM = (function () {
  'use strict';

  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
  }

  /* h('div.cls', {attrs}, children...) builds an element; strings become text. */
  function h(spec, attrs, ...children) {
    const [tag, ...classes] = spec.split('.');
    const el = tag === 'svg'
      ? document.createElementNS('http://www.w3.org/2000/svg', 'svg')
      : document.createElement(tag || 'div');
    if (classes.length) el.setAttribute('class', classes.join(' '));
    if (attrs && typeof attrs === 'object' && !(attrs instanceof Node) && !Array.isArray(attrs)) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value == null || value === false) continue;
        if (key === 'class') el.setAttribute('class', ((el.getAttribute('class') || '') + ' ' + value).trim());
        else if (key === 'text') el.textContent = value;
        else if (key === 'html') el.innerHTML = value;
        else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
        else if (key === 'dataset') Object.assign(el.dataset, value);
        else if (key === 'style' && typeof value === 'object') Object.assign(el.style, value);
        else el.setAttribute(key, value === true ? '' : value);
      }
    } else if (attrs != null) {
      children.unshift(attrs);
    }
    append(el, children);
    return el;
  }
  function append(el, children) {
    for (const child of children.flat(Infinity)) {
      if (child == null || child === false) continue;
      el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }
  function clear(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
    return el;
  }

  /* ---- formatting ---- */
  const fmt = {
    num(value, digits = 4) {
      if (value == null || Number.isNaN(Number(value))) return '–';
      return Number(value).toFixed(digits);
    },
    delta(value, digits = 4) {
      if (value == null || Number.isNaN(Number(value))) return '–';
      const n = Number(value);
      return (n > 0 ? '+' : '') + n.toFixed(digits);
    },
    int(value) {
      if (value == null) return '–';
      return Number(value).toLocaleString();
    },
    tokens(value) {
      if (value == null) return '–';
      const n = Number(value);
      if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
      if (n >= 1e3) return Math.round(n / 1e3) + 'k';
      return String(n);
    },
    when(iso, opts) {
      if (!iso) return '–';
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return String(iso);
      const options = Object.assign({ year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }, opts || {});
      return date.toLocaleString(undefined, options);
    },
    time(iso) {
      if (!iso) return '–';
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return String(iso);
      return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    },
    duration(seconds) {
      if (seconds == null || Number.isNaN(Number(seconds))) return '–';
      const s = Math.max(0, Math.round(Number(seconds)));
      if (s < 60) return s + ' s';
      const m = Math.round(s / 60);
      if (m < 60) return m + ' min';
      const hours = Math.floor(m / 60);
      const rest = m % 60;
      return rest ? `${hours} h ${rest} min` : `${hours} h`;
    },
    between(a, b) {
      if (!a || !b) return null;
      return (new Date(b) - new Date(a)) / 1000;
    },
    bytes(n) {
      if (n == null) return '–';
      if (n < 1024) return n + ' B';
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
      return (n / 1024 / 1024).toFixed(1) + ' MB';
    },
    shortId(sid) {
      return sid ? String(sid).slice(0, 8) : '–';
    },
    truncate(text, n) {
      const s = String(text == null ? '' : text);
      return s.length > n ? s.slice(0, n - 1) + '…' : s;
    },
  };

  const STATUS_LABEL = {
    keep: 'kept', discard: 'discarded', pending: 'proposed', running: 'running', crash: 'crashed',
    oom: 'out of memory', timeout: 'timed out', cancelled: 'cancelled', partial: 'partial',
    candidate: 'gate candidate', registered: 'registered',
  };
  function statusLabel(status) {
    return STATUS_LABEL[status] || status || 'unknown';
  }
  function statusClass(status) {
    return 'status-' + (status || 'discard');
  }
  const STATUS_COLOR_VAR = {
    keep: '--status-keep', candidate: '--status-gate', registered: '--status-gate', discard: '--status-discard',
    pending: '--status-pending', running: '--status-running', crash: '--status-crash', oom: '--status-crash',
    timeout: '--status-crash', partial: '--status-partial', cancelled: '--status-cancelled',
  };
  function statusColor(status) {
    const name = STATUS_COLOR_VAR[status] || '--status-discard';
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ---- graph helpers ---- */
  function lineage(nodes, id) {
    const path = [];
    const seen = new Set();
    let current = id;
    while (current && nodes[current] && !seen.has(current)) {
      seen.add(current);
      path.unshift(nodes[current]);
      current = nodes[current].parent_id;
    }
    return path;
  }
  function executedInOrder(graph) {
    return Object.values(graph.nodes)
      .filter((n) => n.type === 'executed')
      .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || '') || a.id.localeCompare(b.id));
  }

  /* ---- state ---- */
  const state = {
    index: null,
    sources: {},
    run: null,
    selection: null,
    theme: null,
  };
  const listeners = new Set();
  function setState(patch) {
    Object.assign(state, patch);
    for (const fn of listeners) fn(state);
  }
  function onState(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  /* ---- router (hash) ---- */
  function parseRoute(hash) {
    const raw = (hash || '#/').replace(/^#/, '');
    const [pathPart, queryPart] = raw.split('?');
    const parts = pathPart.split('/').filter(Boolean);
    const query = Object.fromEntries(new URLSearchParams(queryPart || ''));
    if (parts.length === 0) return { name: 'home', params: {}, query };
    if (parts[0] === 'runs') return { name: 'runs', params: {}, query };
    if (parts[0] === 'remote') return { name: 'remote', params: {}, query };
    if (parts[0] === 'run' && parts[1]) {
      const run = decodeURIComponent(parts[1]);
      const view = parts[2] || 'tree';
      if (view === 'node' && parts[3]) return { name: 'tree', params: { run, node: parts[3] }, query };
      if (view === 'session' && parts[3]) {
        return { name: 'agent', params: { run, session: parts[3], turn: parts[4] === 'turn' ? Number(parts[5]) : null }, query };
      }
      if (['tree', 'timeline', 'agent', 'nodes', 'notes'].includes(view)) return { name: view, params: { run }, query };
    }
    return { name: 'home', params: {}, query };
  }
  function runHref(run, view, extra) {
    return `#/run/${encodeURIComponent(run)}/${view}${extra || ''}`;
  }
  function navigate(hash) {
    if (location.hash === hash) window.dispatchEvent(new HashChangeEvent('hashchange'));
    else location.hash = hash;
  }

  /* ---- tooltip ---- */
  const tooltip = {
    el: null,
    show(html, x, y) {
      if (!this.el) this.el = document.getElementById('tooltip');
      this.el.innerHTML = html;
      this.el.style.display = 'block';
      const rect = this.el.getBoundingClientRect();
      const left = Math.min(x + 14, window.innerWidth - rect.width - 8);
      const top = Math.min(y + 14, window.innerHeight - rect.height - 8);
      this.el.style.left = left + 'px';
      this.el.style.top = top + 'px';
    },
    hide() {
      if (this.el) this.el.style.display = 'none';
    },
  };

  /* ---- theme ---- */
  function initTheme() {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => window.dispatchEvent(new CustomEvent('am:theme')));
    let saved = null;
    try { saved = localStorage.getItem('am.theme'); } catch (e) { saved = null; }
    if (saved === 'light' || saved === 'dark') document.documentElement.dataset.theme = saved;
    const button = document.getElementById('theme-toggle');
    if (button) {
      button.addEventListener('click', () => {
        const dark = document.documentElement.dataset.theme === 'dark' ||
          (!document.documentElement.dataset.theme && window.matchMedia('(prefers-color-scheme: dark)').matches);
        const next = dark ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        try { localStorage.setItem('am.theme', next); } catch (e) { /* private mode */ }
        window.dispatchEvent(new CustomEvent('am:theme'));
      });
    }
  }

  function storage(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch (e) { return null; }
    return value;
  }

  return {
    esc, h, append, clear, fmt, statusLabel, statusClass, statusColor, cssVar, lineage, executedInOrder,
    state, setState, onState, parseRoute, runHref, navigate, tooltip, initTheme, storage,
    views: {},
  };
})();
