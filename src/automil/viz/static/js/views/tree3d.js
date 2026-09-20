/* The tree in three dimensions: a top-down layout on the page's paper, no glow, the kept lineage in teal. */
(function (AM) {
  'use strict';
  const { esc, fmt } = AM;
  const KEEP_CLASS = new Set(['keep', 'candidate', 'registered']);

  function available() {
    return typeof ForceGraph3D !== 'undefined' && typeof THREE !== 'undefined';
  }

  function graphData(graph) {
    const nodes = graph.nodes || {};
    const spine = new Set(AM.lineage(nodes, graph.meta && graph.meta.best_node_id).map((n) => n.id));
    const list = Object.values(nodes)
      .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || '') || a.id.localeCompare(b.id))
      .map((n) => ({ id: n.id, status: n.status, description: n.description, primary_value: n.primary_value, type: n.type, spine: spine.has(n.id), best: n.id === (graph.meta && graph.meta.best_node_id) }));
    const links = [];
    for (const n of list) {
      const parent = nodes[n.id].parent_id;
      if (parent && nodes[parent]) links.push({ source: parent, target: n.id, spine: spine.has(parent) && spine.has(n.id) });
    }
    return { nodes: list, links };
  }

  function colorOf(status) {
    return new THREE.Color(AM.statusColor(status) || '#999');
  }

  function sphere(node, opts) {
    const group = new THREE.Group();
    const radius = opts.compact ? (node.best ? 5.5 : node.spine ? 4.5 : 3.8) : node.best ? 8 : node.spine ? 6.5 : 5.2;
    const material = new THREE.MeshLambertMaterial({ color: colorOf(node.status) });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 24, 16), material);
    if (node.status === 'discard' || node.status === 'cancelled') {
      material.color = new THREE.Color(AM.cssVar('--status-discard'));
      const rim = new THREE.Mesh(new THREE.SphereGeometry(radius + 0.5, 24, 16), new THREE.MeshBasicMaterial({ color: new THREE.Color(AM.cssVar('--status-discard-stroke')), transparent: true, opacity: 0.5, side: THREE.BackSide }));
      group.add(rim);
    } else {
      const rim = new THREE.Mesh(new THREE.SphereGeometry(radius + 0.5, 24, 16), new THREE.MeshBasicMaterial({ color: new THREE.Color(AM.cssVar('--ink')), transparent: true, opacity: 0.55, side: THREE.BackSide }));
      group.add(rim);
    }
    group.add(mesh);
    if (!opts.compact && (node.spine || node.status === 'running' || node.status === 'pending') && typeof SpriteText !== 'undefined') {
      const label = new SpriteText(node.id.replace('node_', ''), 4.2, AM.cssVar('--ink'));
      label.fontFace = 'JetBrains Mono, Menlo, monospace';
      label.backgroundColor = false;
      label.position.y = radius + 5;
      group.add(label);
    }
    return group;
  }

  /* Draws into container; returns {update(selected), fit(), focus(id), destroy(), setTheme()}. */
  function renderTree3D(container, graph, options) {
    const opts = Object.assign({ selected: null, onSelect: null, compact: false, rotate: false }, options || {});
    const data = graphData(graph);
    const paper = AM.cssVar('--paper');
    let selectedId = opts.selected;
    const fg = ForceGraph3D({ controlType: opts.compact ? 'orbit' : 'trackball' })(container)
      .width(container.clientWidth || 600)
      .height(container.clientHeight || 400)
      .backgroundColor(paper)
      .showNavInfo(false)
      .enableNodeDrag(false)
      .enableNavigationControls(true)
      .dagMode('td')
      .dagLevelDistance(opts.compact ? 36 : 54)
      .nodeThreeObject((node) => sphere(node, opts))
      .nodeThreeObjectExtend(false)
      .linkColor((link) => (link.spine ? AM.cssVar('--ink') : '#b5b5b5'))
      .linkWidth((link) => (link.spine ? (opts.compact ? 1.4 : 2.4) : opts.compact ? 0.6 : 1.0))
      .linkOpacity(1)
      .linkDirectionalParticles(0)
      .d3Force('charge', d3.forceManyBody().strength(opts.compact ? -40 : -70))
      .d3Force('link', d3.forceLink().id((n) => n.id).distance(opts.compact ? 16 : 26).strength(0.9))
      .d3AlphaDecay(0.045)
      .cooldownTicks(170)
      .warmupTicks(40);
    if (!opts.compact) {
      fg.nodeLabel((node) => `<div class="gl-tooltip"><span class="mono">${esc(node.id)}</span> ${esc(AM.statusLabel(node.status))}<br>${esc(fmt.truncate(node.description, 90))}${node.primary_value != null && node.type === 'executed' ? '<br>' + esc(fmt.num(node.primary_value)) : ''}</div>`)
        .onNodeHover((node) => { container.style.cursor = node ? 'pointer' : 'default'; })
        .onNodeClick((node) => { if (node && opts.onSelect) opts.onSelect(node.id); });
    }
    fg.graphData(data);
    /* start near the layout so the first frames already show a tree, not a speck */
    fg.cameraPosition({ x: 0, y: 40, z: opts.compact ? 230 : 320 }, { x: 0, y: 0, z: 0 }, 0);
    if (opts.compact) {
      /* the hero only turns by itself: no wheel, drag or pan, so the page keeps scrolling */
      const controls = fg.controls();
      controls.enableZoom = false;
      controls.enableRotate = false;
      controls.enablePan = false;
    }
    let settled = false;
    function centroid() {
      return {
        x: d3.mean(data.nodes, (n) => n.x || 0) || 0,
        y: d3.mean(data.nodes, (n) => n.y || 0) || 0,
        z: d3.mean(data.nodes, (n) => n.z || 0) || 0,
      };
    }
    function tilt(ms) {
      /* an elevated, slightly turned viewpoint, so the depth of the layout shows */
      const c = centroid();
      const cam = fg.camera().position;
      const d = Math.hypot(cam.x - c.x, cam.y - c.y, cam.z - c.z) * 0.95;
      fg.cameraPosition({ x: c.x + d * Math.sin(0.55) * 0.9, y: c.y + d * 0.42, z: c.z + d * Math.cos(0.55) * 0.9 }, c, ms);
    }
    let onSettled = null;
    fg.onEngineStop(() => {
      if (settled) return;
      settled = true;
      fg.zoomToFit(400, opts.compact ? 6 : 24);
      setTimeout(() => {
        tilt(700);
        if (onSettled) setTimeout(onSettled, 800);
      }, 450);
    });

    /* selection ring */
    let ring = null;
    function update(id) {
      selectedId = id;
      if (ring) { fg.scene().remove(ring); ring = null; }
      const node = data.nodes.find((n) => n.id === id);
      if (!node || node.x == null) return;
      ring = new THREE.Mesh(new THREE.TorusGeometry(8, 0.7, 12, 48), new THREE.MeshBasicMaterial({ color: new THREE.Color(AM.cssVar('--accent-ink')) }));
      ring.position.set(node.x, node.y, node.z);
      ring.lookAt(fg.camera().position);
      fg.scene().add(ring);
    }
    setTimeout(() => update(selectedId), 900);

    /* slow rotation for the hero */
    let frame = null;
    let angle = 0;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (opts.rotate && !reduce) {
      const start = () => {
        const centre = centroid();
        const cam = fg.camera().position;
        const distance = Math.hypot(cam.x - centre.x, cam.z - centre.z) * 0.9 || 200;
        const height = cam.y;
        angle = Math.atan2(cam.x - centre.x, cam.z - centre.z);
        const spin = () => {
          angle += 0.003;
          fg.cameraPosition({ x: centre.x + distance * Math.sin(angle), y: height, z: centre.z + distance * Math.cos(angle) }, centre);
          frame = requestAnimationFrame(spin);
        };
        frame = requestAnimationFrame(spin);
      };
      onSettled = start;
    }

    const onResize = () => fg.width(container.clientWidth || 600).height(container.clientHeight || 400);
    window.addEventListener('resize', onResize);

    return {
      update,
      fit() { fg.zoomToFit(400, 24); setTimeout(() => tilt(600), 450); },
      focus(id) {
        const node = data.nodes.find((n) => n.id === id);
        if (!node || node.x == null) return;
        const distance = 120;
        const norm = Math.hypot(node.x, node.y, node.z) || 1;
        const ratio = 1 + distance / norm;
        fg.cameraPosition({ x: node.x * ratio, y: node.y * ratio, z: node.z * ratio }, node, 600);
      },
      setTheme() { fg.backgroundColor(AM.cssVar('--paper')); },
      destroy() {
        if (frame) cancelAnimationFrame(frame);
        window.removeEventListener('resize', onResize);
        try { fg._destructor && fg._destructor(); } catch (err) { /* older builds */ }
        AM.clear(container);
      },
    };
  }

  AM.tree3d = { available, renderTree3D };
})(window.AM);
