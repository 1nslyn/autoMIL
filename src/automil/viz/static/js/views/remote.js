/* Remote access: reaching a dashboard that runs on another machine. */
(function (AM) {
  'use strict';
  const { h } = AM;

  AM.views.remote = async function (root) {
    const page = h('div.page.narrow',
      h('div.page-title', h('h1', 'Remote access'), h('p', 'The dashboard runs where the project runs. It listens on the loopback interface of that host only, so a browser elsewhere reaches it through an SSH port forward.')),
      h('section.section',
        h('h2', 'On the host'),
        h('pre', 'cd <your project>\nuv run automil viz start'),
        h('p', 'The command prints the address and, when it sees that you are logged in over SSH, the exact forward to run on your own machine.'),
      ),
      h('section.section',
        h('h2', 'From your own machine'),
        h('p', 'One host, reachable directly:'),
        h('pre', 'ssh -N -L 8420:127.0.0.1:8420 user@gpu-host\n# then open http://localhost:8420'),
        h('p', 'A compute node behind a login node (a cluster with a scheduler):'),
        h('pre', 'ssh -N -L 8420:<compute-node>:8420 user@login-node\n# the dashboard must bind that node\'s address: viz.host in automil/config.yaml, or --host'),
        h('p', 'A jump host in between:'),
        h('pre', 'ssh -N -J user@jump -L 8420:127.0.0.1:8420 user@gpu-host'),
        h('p', 'VS Code and similar editors forward the port on their own when they see the address in the terminal.'),
      ),
      h('section.section',
        h('h2', 'Two ways to view it'),
        h('p', 'Open http://localhost:8420: the host serves this same site with your run as its only run.'),
        h('p', 'Or stay on this site and connect it to the forwarded port from the ', h('a', { href: '#/runs', text: 'runs page' }), '. The page reads http://localhost:8420 directly; the host allows that only for this site and for pages served from localhost. Chrome asks once for permission to reach the local network. Safari does not allow a public page to read a local address; use the first way there.'),
      ),
      h('section.section',
        h('h2', 'Sharing a run without a tunnel'),
        h('pre', 'uv run automil viz export --out ./site --single-file ./run.html'),
        h('p', 'The directory is the complete site with the run recorded inside it and can be served by any static file server. The single file opens from disk.'),
      ),
      h('section.section',
        h('h2', 'Ports'),
        h('p', 'The dashboard uses 8420 by default (viz.port in automil/config.yaml, or --port). Two projects on one host need two ports and two forwards.'),
      ),
    );
    root.append(page);
  };
})(window.AM);
