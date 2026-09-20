/* Markdown to sanitised HTML. Transcript text is untrusted content: it always goes through DOMPurify. */
(function (AM) {
  'use strict';

  const hasMarked = typeof window.marked !== 'undefined';
  const hasPurify = typeof window.DOMPurify !== 'undefined';
  if (hasMarked) {
    window.marked.setOptions({ gfm: true, breaks: false, mangle: false, headerIds: false });
  }

  function render(text) {
    const source = String(text == null ? '' : text);
    let html;
    if (hasMarked) {
      try { html = window.marked.parse(source); } catch (err) { html = '<pre>' + AM.esc(source) + '</pre>'; }
    } else {
      html = '<pre>' + AM.esc(source) + '</pre>';
    }
    if (hasPurify) {
      return window.DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: ['style', 'form', 'input', 'button', 'iframe', 'object', 'embed'],
        FORBID_ATTR: ['style', 'onerror', 'onload'],
      });
    }
    return '<pre>' + AM.esc(source) + '</pre>';
  }

  /* A DOM node with the rendered markdown; links open in a new tab. */
  function element(text, className) {
    const el = AM.h('div.prose' + (className ? '.' + className : ''));
    el.innerHTML = render(text);
    for (const link of el.querySelectorAll('a[href]')) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener');
    }
    return el;
  }

  AM.markdown = { render, element };
})(window.AM);
