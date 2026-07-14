/* autoMIL Experiment Graph Dashboard - 3D Force Graph Edition */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
var graphData = null;
var selectedNode = null;
var activeTab = 'tree';
var graph3d = null;
var lastTreeFingerprint = null;

// Status color map
var STATUS_COLORS = {
    keep: '#34d399',
    discard: '#475569',
    pending: '#f59e0b',
    running: '#38bdf8',
    crash: '#ef4444',
    oom: '#ef4444',
    timeout: '#ef4444',
    cancelled: '#64748b'
};

var STATUS_GLOW = {
    keep: 'rgba(52, 211, 153, 0.4)',
    discard: 'rgba(71, 85, 105, 0.2)',
    pending: 'rgba(245, 158, 11, 0.4)',
    running: 'rgba(56, 189, 248, 0.55)',
    crash: 'rgba(239, 68, 68, 0.3)',
    oom: 'rgba(239, 68, 68, 0.3)',
    timeout: 'rgba(239, 68, 68, 0.3)',
    cancelled: 'rgba(100, 116, 139, 0.2)'
};

// ---------------------------------------------------------------------------
// SSE Client
// ---------------------------------------------------------------------------
var reconnectDelay = 1000;

function connectSSE() {
    var source = new EventSource('/events');

    source.onopen = function () {
        reconnectDelay = 1000;
        setConnectionStatus(true);
    };

    source.onmessage = function (evt) {
        try {
            var msg = JSON.parse(evt.data);
            if (msg.type === 'graph_update' && msg.full_graph) {
                graphData = msg.full_graph;
                renderActiveView();
                updateStats();
            }
        } catch (e) {
            console.error('SSE parse error:', e);
        }
    };

    source.onerror = function () {
        source.close();
        setConnectionStatus(false);
        var delay = Math.min(reconnectDelay, 30000);
        reconnectDelay = Math.min(reconnectDelay * 2, 30000);
        setTimeout(connectSSE, delay);
    };
}

function setConnectionStatus(connected) {
    var dot = document.querySelector('#connection-status .status-dot');
    var text = document.querySelector('#connection-status .status-text');
    dot.className = 'status-dot ' + (connected ? 'connected' : 'disconnected');
    text.textContent = connected ? 'Live' : 'Offline';
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
function updateStats() {
    if (!graphData || !graphData.meta) return;
    document.getElementById('stat-executed').textContent = graphData.meta.total_executed || 0;
    document.getElementById('stat-proposed').textContent = graphData.meta.total_proposed || 0;
    document.getElementById('stat-best').textContent = (graphData.meta.best_composite || 0).toFixed(3);
}

// ---------------------------------------------------------------------------
// Tab Switching
// ---------------------------------------------------------------------------
function initTabs() {
    var tabs = document.querySelectorAll('.tab');
    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            switchTab(tab.dataset.tab);
        });
    });

    var hash = window.location.hash.replace('#', '');
    if (['tree', 'leaderboard', 'potential', 'techniques'].indexOf(hash) !== -1) {
        switchTab(hash);
    }
}

function switchTab(name) {
    activeTab = name;
    window.location.hash = '#' + name;

    document.querySelectorAll('.tab').forEach(function (t) {
        t.classList.toggle('active', t.dataset.tab === name);
    });
    document.querySelectorAll('.view').forEach(function (v) {
        v.classList.toggle('active', v.id === 'view-' + name);
    });

    renderActiveView();
}

function renderActiveView() {
    if (!graphData) return;
    switch (activeTab) {
        case 'tree': renderTree3D(); break;
        case 'leaderboard': renderLeaderboard(); break;
        case 'potential': renderPotential(); break;
        case 'techniques': renderTechniques(); break;
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function getNodes() {
    if (!graphData || !graphData.nodes) return [];
    return Object.values(graphData.nodes);
}

function getNodeById(id) {
    if (!graphData || !graphData.nodes) return null;
    return graphData.nodes[id] || null;
}

function getLineage(nodeId) {
    var path = [];
    var current = nodeId;
    var visited = new Set();
    while (current && !visited.has(current)) {
        visited.add(current);
        var node = getNodeById(current);
        if (!node) break;
        path.push(node);
        current = node.parent_id;
    }
    path.reverse();
    return path;
}

function getKeepSpineIds() {
    if (!graphData || !graphData.meta || !graphData.meta.best_node_id) return new Set();
    var ids = new Set();
    var lineage = getLineage(graphData.meta.best_node_id);
    lineage.forEach(function (n) { ids.add(n.id); });
    return ids;
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.substring(0, len) + '...' : str;
}

function formatDelta(val) {
    if (val === undefined || val === null) return '-';
    var sign = val > 0 ? '+' : '';
    return sign + val.toFixed(4);
}

function statusBadge(status) {
    var cls = 'badge badge-' + (status || 'discard');
    return '<span class="' + cls + '">' + (status || '-') + '</span>';
}

function tierBadge(tier) {
    var labels = { 1: 'Tier 1', 2: 'Tier 2', 3: 'Tier 3' };
    var cls = 'badge badge-tier-' + (tier || 2);
    return '<span class="' + cls + '">' + (labels[tier] || 'Tier ' + tier) + '</span>';
}

// ---------------------------------------------------------------------------
// Node Selection & Detail Panel
// ---------------------------------------------------------------------------
function selectNode(nodeId) {
    selectedNode = nodeId;
    renderDetailPanel();
    // Highlight in leaderboard
    document.querySelectorAll('#leaderboard-table tbody tr').forEach(function (tr) {
        tr.classList.toggle('selected', tr.dataset.nodeId === nodeId);
    });
}

function deselectNode() {
    selectedNode = null;
    var panel = document.getElementById('detail-panel');
    panel.classList.remove('open');
    document.querySelectorAll('#leaderboard-table tbody tr').forEach(function (tr) {
        tr.classList.remove('selected');
    });
    var lineageEl = document.getElementById('leaderboard-lineage');
    if (lineageEl) lineageEl.innerHTML = '';
}

function renderDetailPanel() {
    var node = getNodeById(selectedNode);
    if (!node) {
        deselectNode();
        return;
    }

    var panel = document.getElementById('detail-panel');
    panel.classList.add('open');

    document.getElementById('detail-id').textContent = node.id;
    document.getElementById('detail-status').innerHTML = statusBadge(node.status);
    document.getElementById('detail-composite').textContent =
        node.composite !== undefined ? node.composite.toFixed(4) : '-';
    document.getElementById('detail-type').textContent = node.type || '-';

    // Metrics
    var metricsHtml = '';
    // val-firewall: the dashboard shows only validation metrics (the selection
    // signal). Test is sealed in certify.json and never surfaced during search.
    var metricFields = [
        ['val_auc', 'Val AUC'], ['val_bacc', 'Val BACC'],
        ['val_c_index', 'Val C-index']
    ];
    metricFields.forEach(function (pair) {
        // D-200 / DEC-04: metrics live under node.metrics post-Phase-8; defensive fallback for legacy graph.json returns {}.
        var val = (node.metrics || {})[pair[0]];
        metricsHtml += '<div class="metric-item">' +
            '<div class="metric-label">' + pair[1] + '</div>' +
            '<div class="metric-value">' + (val !== undefined ? val.toFixed(4) : '-') + '</div>' +
            '</div>';
    });
    document.getElementById('detail-metrics').innerHTML = metricsHtml;

    // Deltas
    var deltasHtml = '';
    var pd = node.parent_delta;
    var gd = node.global_delta;
    deltasHtml += '<div class="metric-item"><div class="metric-label">Parent &Delta;</div>' +
        '<div class="metric-value ' + (pd > 0 ? 'positive' : pd < 0 ? 'negative' : '') + '">' +
        formatDelta(pd) + '</div></div>';
    deltasHtml += '<div class="metric-item"><div class="metric-label">Global &Delta;</div>' +
        '<div class="metric-value ' + (gd > 0 ? 'positive' : gd < 0 ? 'negative' : '') + '">' +
        formatDelta(gd) + '</div></div>';
    document.getElementById('detail-deltas').innerHTML = deltasHtml;

    // Techniques
    var techHtml = '';
    (node.techniques || []).forEach(function (t) {
        techHtml += '<span class="tag">' + t + '</span>';
    });
    document.getElementById('detail-techniques').innerHTML = techHtml || '<span style="color:#475569">None</span>';

    // Description
    document.getElementById('detail-description').textContent = node.description || '-';

    // Lineage
    renderLineageBar('detail-lineage', node.id);

    // Info
    var info = '';
    var infoFields = [
        ['Children', node.child_count],
        ['Potential', node.potential !== undefined ? node.potential.toFixed(4) : '-'],
        ['Created', node.created_at ? node.created_at.replace('T', ' ').substring(0, 19) : '-'],
        ['VRAM (GB)', node.vram_gb !== undefined ? node.vram_gb.toFixed(1) : '-'],
        ['Time (min)', node.elapsed_min !== undefined ? node.elapsed_min.toFixed(1) : '-'],
        ['Commit', node.commit || '-']
    ];
    infoFields.forEach(function (pair) {
        info += '<span class="label">' + pair[0] + '</span><span class="value">' + pair[1] + '</span>';
    });
    document.getElementById('detail-info').innerHTML = info;
}

function renderLineageBar(containerId, nodeId) {
    var container = document.getElementById(containerId);
    var lineage = getLineage(nodeId);
    var html = '';
    lineage.forEach(function (n, i) {
        if (i > 0) html += '<span class="lineage-sep">&#9656;</span>';
        var cls = 'lineage-node' + (n.id === nodeId ? ' current' : '');
        html += '<span class="' + cls + '" data-node-id="' + n.id + '">' +
            truncate(n.description || n.id, 20) + '</span>';
    });
    container.innerHTML = html;
    container.querySelectorAll('.lineage-node').forEach(function (el) {
        el.addEventListener('click', function () {
            selectNode(el.dataset.nodeId);
        });
    });
}

// ---------------------------------------------------------------------------
// 3D Tree View
// ---------------------------------------------------------------------------
function buildGraphData(nodes) {
    var keepSpine = getKeepSpineIds();
    var nodeMap = {};
    nodes.forEach(function (n) { nodeMap[n.id] = n; });

    var graphNodes = [];
    var graphLinks = [];

    nodes.forEach(function (n) {
        graphNodes.push({
            id: n.id,
            status: n.status || 'discard',
            composite: n.composite || 0,
            description: n.description || n.id,
            type: n.type || 'executed',
            isSpine: keepSpine.has(n.id),
            nodeData: n
        });

        if (n.parent_id && nodeMap[n.parent_id]) {
            var isSpineLink = keepSpine.has(n.id) && keepSpine.has(n.parent_id);
            graphLinks.push({
                source: n.parent_id,
                target: n.id,
                isSpine: isSpineLink
            });
        }
    });

    return { nodes: graphNodes, links: graphLinks };
}

function computeTreeFingerprint(gData) {
    // Cheap stable signature of just the structural fields the 3D view binds to.
    // Sorted by id so order changes don't trigger a reheat.
    var ids = gData.nodes.map(function (n) { return n.id; }).sort();
    var parts = [];
    var byId = {};
    gData.nodes.forEach(function (n) { byId[n.id] = n; });
    ids.forEach(function (id) {
        var n = byId[id];
        parts.push(id + ':' + n.status + ':' + (n.composite || 0).toFixed(4) + ':' + (n.isSpine ? '1' : '0'));
    });
    var linkSig = gData.links.map(function (l) {
        var s = typeof l.source === 'object' ? l.source.id : l.source;
        var t = typeof l.target === 'object' ? l.target.id : l.target;
        return s + '>' + t + (l.isSpine ? '*' : '');
    }).sort().join('|');
    return parts.join(';') + '||' + linkSig;
}

function renderTree3D() {
    var allNodes = getNodes();
    var emptyEl = document.getElementById('tree-empty');
    var container = document.getElementById('tree-3d');

    if (typeof ForceGraph3D === 'undefined') {
        emptyEl.textContent = '3D library failed to load. Check your internet connection.';
        emptyEl.classList.add('visible');
        console.error('ForceGraph3D not loaded');
        return;
    }

    if (allNodes.length === 0) {
        emptyEl.classList.add('visible');
        if (graph3d) {
            graph3d.graphData({ nodes: [], links: [] });
        }
        return;
    }
    emptyEl.classList.remove('visible');

    var gData = buildGraphData(allNodes);

    try {
        if (!graph3d) {
            graph3d = ForceGraph3D()(container)
                .width(container.clientWidth || window.innerWidth)
                .height(container.clientHeight || window.innerHeight - 56)
                .backgroundColor('#0a0e1a')
                .showNavInfo(false)
                .nodeRelSize(4)
                .nodeVal(function (node) {
                    return Math.max(1, (node.composite || 0) * 10 + 1);
                })
                .nodeColor(function (node) {
                    return STATUS_COLORS[node.status] || '#475569';
                })
                .nodeOpacity(0.9)
                .nodeThreeObject(function (node) {
                    var group = new THREE.Group();

                    // Sphere — running/pending nodes lack composite, give them a visible floor
                    var baseRadius = 2 + (node.composite || 0) * 8;
                    if (node.status === 'running' || node.status === 'pending') {
                        baseRadius = Math.max(baseRadius, 4);
                    }
                    var radius = Math.max(2, Math.min(8, baseRadius));
                    var color = STATUS_COLORS[node.status] || '#475569';
                    var emissiveIntensity = node.isSpine ? 0.4 : 0.15;
                    if (node.status === 'running') emissiveIntensity = 0.55;
                    var material = new THREE.MeshPhongMaterial({
                        color: color,
                        transparent: true,
                        opacity: node.status === 'discard' ? 0.5 : 0.9,
                        emissive: color,
                        emissiveIntensity: emissiveIntensity
                    });
                    var geometry = new THREE.SphereGeometry(radius, 16, 12);
                    var sphere = new THREE.Mesh(geometry, material);
                    group.add(sphere);

                    // Glow for keep/running/pending nodes (running gets a stronger halo)
                    if (node.status === 'keep' || node.status === 'running' || node.status === 'pending') {
                        var glowScale = node.status === 'running' ? 1.9 : 1.6;
                        var glowOpacity = node.status === 'running' ? 0.18 : 0.08;
                        var glowGeometry = new THREE.SphereGeometry(radius * glowScale, 16, 12);
                        var glowMaterial = new THREE.MeshBasicMaterial({
                            color: color,
                            transparent: true,
                            opacity: glowOpacity
                        });
                        var glow = new THREE.Mesh(glowGeometry, glowMaterial);
                        group.add(glow);
                    }

                    // Label for keep nodes
                    if (node.status === 'keep' && typeof SpriteText !== 'undefined') {
                        var sprite = new SpriteText(truncate(node.description, 18));
                        sprite.color = '#94a3b8';
                        sprite.textHeight = 2.5;
                        sprite.position.y = radius + 4;
                        sprite.backgroundColor = false;
                        group.add(sprite);
                    }

                    return group;
                })
                .linkWidth(function (link) {
                    return link.isSpine ? 2.5 : 0.5;
                })
                .linkColor(function (link) {
                    return link.isSpine ? 'rgba(52, 211, 153, 0.6)' : 'rgba(71, 85, 105, 0.3)';
                })
                .linkOpacity(1)
                .linkDirectionalParticles(function (link) {
                    return link.isSpine ? 2 : 0;
                })
                .linkDirectionalParticleWidth(1.5)
                .linkDirectionalParticleColor(function () {
                    return '#34d399';
                })
                .linkDirectionalParticleSpeed(0.005)
                .onNodeHover(function (node) {
                    container.style.cursor = node ? 'pointer' : 'default';
                })
                .onNodeClick(function (node) {
                    if (node) {
                        selectNode(node.id);
                        // Focus camera on clicked node
                        var distance = 80;
                        var norm = Math.hypot(node.x || 0, node.y || 0, node.z || 0) || 1;
                        var distRatio = 1 + distance / norm;
                        graph3d.cameraPosition(
                            { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
                            node,
                            1000
                        );
                    }
                })
                .nodeLabel(function (node) {
                    var score = node.composite ? node.composite.toFixed(4) : '-';
                    var statusColor = STATUS_COLORS[node.status] || '#475569';
                    return '<div class="graph-tooltip">' +
                        '<div class="tt-title">' + truncate(node.description, 40) + '</div>' +
                        '<div class="tt-score" style="color:' + statusColor + '">' + score + '</div>' +
                        '<div class="tt-status" style="color:' + statusColor + '">' + (node.status || '-') + '</div>' +
                        '</div>';
                })
                .d3Force('charge', d3.forceManyBody().strength(-120))
                .d3Force('link', d3.forceLink().id(function (node) {
                    return node.id;
                }).distance(30).strength(0.7))
                // Freeze the layout after initial settle. Default cooldownTicks
                // is Infinity, so nodes keep drifting (what looks like a
                // "fold/unfold" pulse). Stop the simulation after ~200 ticks.
                .cooldownTicks(200);

            // Auto-rotate disabled — it adds continuous perceived motion on
            // top of an already-busy scene. Use mouse drag to rotate manually.

            // Handle resize
            window.addEventListener('resize', function () {
                if (activeTab === 'tree' && graph3d) {
                    graph3d.width(container.clientWidth || window.innerWidth);
                    graph3d.height(container.clientHeight || window.innerHeight - 56);
                }
            });

            graph3d.graphData(gData);
            lastTreeFingerprint = computeTreeFingerprint(gData);

            // Fit to view once on initial render; subsequent updates preserve camera
            setTimeout(function () {
                if (graph3d) {
                    graph3d.zoomToFit(800, 50);
                }
            }, 500);
        } else {
            // Skip graphData() if nothing structural changed — calling it
            // triggers d3ForceLayout.alpha(1) inside the lib, reheating the
            // simulation and visibly resetting the layout.
            var fp = computeTreeFingerprint(gData);
            if (fp !== lastTreeFingerprint) {
                graph3d.graphData(gData);
                lastTreeFingerprint = fp;
            }
        }
    } catch (err) {
        var message = 'Tree render failed. Check the browser console for details.';
        if (err && err.message && err.message.indexOf('WebGL context') !== -1) {
            message = 'Tree render failed: WebGL is unavailable in this browser or environment.';
        }
        emptyEl.textContent = message;
        emptyEl.classList.add('visible');
        console.error('Tree render failed:', err);
    }
}

// ---------------------------------------------------------------------------
// Leaderboard View
// ---------------------------------------------------------------------------
function renderLeaderboard() {
    var nodes = getNodes().filter(function (n) { return n.type === 'executed'; });
    nodes.sort(function (a, b) { return (b.composite || 0) - (a.composite || 0); });
    nodes = nodes.slice(0, 20);

    var tbody = document.querySelector('#leaderboard-table tbody');
    var emptyEl = document.getElementById('leaderboard-empty');

    if (nodes.length === 0) {
        tbody.innerHTML = '';
        emptyEl.classList.add('visible');
        return;
    }
    emptyEl.classList.remove('visible');

    var html = '';
    nodes.forEach(function (node, i) {
        var selected = node.id === selectedNode ? ' selected' : '';
        html += '<tr class="' + selected + '" data-node-id="' + node.id + '">' +
            '<td>' + (i + 1) + '</td>' +
            '<td>' + truncate(node.description || '-', 40) + '</td>' +
            '<td>' + (node.composite || 0).toFixed(4) + '</td>' +
            '<td>' + formatDelta(node.parent_delta) + '</td>' +
            '<td>' + formatDelta(node.global_delta) + '</td>' +
            '<td>' + statusBadge(node.status) + '</td>' +
            '</tr>';
    });
    tbody.innerHTML = html;

    tbody.querySelectorAll('tr').forEach(function (tr) {
        tr.addEventListener('click', function () {
            selectNode(tr.dataset.nodeId);
            renderLineageBar('leaderboard-lineage', tr.dataset.nodeId);
        });
    });

    if (selectedNode) {
        renderLineageBar('leaderboard-lineage', selectedNode);
    }
}

// ---------------------------------------------------------------------------
// Branch Potential View
// ---------------------------------------------------------------------------
function renderPotential() {
    var nodes = getNodes().filter(function (n) {
        return n.type === 'proposed' && n.status === 'pending';
    });
    nodes.sort(function (a, b) { return (b.potential || 0) - (a.potential || 0); });

    var tbody = document.querySelector('#potential-table tbody');
    var emptyEl = document.getElementById('potential-empty');

    if (nodes.length === 0) {
        tbody.innerHTML = '';
        emptyEl.classList.add('visible');
        return;
    }
    emptyEl.classList.remove('visible');

    var html = '';
    nodes.forEach(function (node, i) {
        var parentNode = getNodeById(node.parent_id);
        var parentLabel = parentNode ? truncate(parentNode.description || parentNode.id, 20) : '-';
        var techs = (node.techniques || []).join(', ');
        html += '<tr data-node-id="' + node.id + '">' +
            '<td>' + (i + 1) + '</td>' +
            '<td>' + truncate(node.description || '-', 40) + '</td>' +
            '<td>' + parentLabel + '</td>' +
            '<td>' + techs + '</td>' +
            '<td>' + (node.potential || 0).toFixed(4) + '</td>' +
            '<td>' + tierBadge(node.tier) + '</td>' +
            '</tr>';
    });
    tbody.innerHTML = html;

    tbody.querySelectorAll('tr').forEach(function (tr) {
        tr.addEventListener('click', function () {
            selectNode(tr.dataset.nodeId);
        });
    });
}

// ---------------------------------------------------------------------------
// Technique Effectiveness View
// ---------------------------------------------------------------------------
function renderTechniques() {
    var stats = graphData ? graphData.technique_stats || {} : {};
    var keys = Object.keys(stats);
    var emptyEl = document.getElementById('techniques-empty');

    if (keys.length === 0) {
        document.querySelector('#techniques-table tbody').innerHTML = '';
        emptyEl.classList.add('visible');
        return;
    }
    emptyEl.classList.remove('visible');

    keys.sort(function (a, b) {
        return (stats[b].times_tried || 0) - (stats[a].times_tried || 0);
    });

    var html = '';
    keys.forEach(function (tech) {
        var s = stats[tech];
        var bestDelta = s.best_parent_delta;
        if (bestDelta === -Infinity || bestDelta === undefined) bestDelta = 0;
        var effective = bestDelta > 0;
        html += '<tr>' +
            '<td>' + tech + '</td>' +
            '<td>' + (s.times_tried || 0) + '</td>' +
            '<td>' + formatDelta(bestDelta) + '</td>' +
            '<td>' + formatDelta(s.avg_parent_delta) + '</td>' +
            '<td><span class="' + (effective ? 'verdict-effective' : 'verdict-ineffective') + '">' +
            (effective ? 'Effective' : 'Ineffective') + '</span></td>' +
            '</tr>';
    });
    document.querySelector('#techniques-table tbody').innerHTML = html;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
    initTabs();
    connectSSE();

    document.getElementById('detail-close').addEventListener('click', function () {
        deselectNode();
    });

    document.getElementById('reset-view').addEventListener('click', function () {
        if (graph3d) {
            graph3d.zoomToFit(800, 50);
        }
    });
});
