"""What to print when the dashboard starts on a remote host.

Most projects run on a GPU host reached over SSH. The dashboard binds the
loopback interface there, so the browser on the user's own machine needs a
port forward. The hint names the exact command.
"""
from __future__ import annotations

import getpass
import os
import socket
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class HostInfo:
    user: str
    host: str
    over_ssh: bool

    def tunnel_command(self, port: int, *, local_port: int | None = None) -> str:
        local = local_port or port
        return f"ssh -N -L {local}:127.0.0.1:{port} {self.user}@{self.host}"


def host_info(env: Mapping[str, str] | None = None) -> HostInfo:
    environment = os.environ if env is None else env
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = environment.get("USER") or "user"
    host = socket.getfqdn() or socket.gethostname() or "host"
    over_ssh = bool(environment.get("SSH_CONNECTION") or environment.get("SSH_TTY") or environment.get("SSH_CLIENT"))
    return HostInfo(user=user, host=host, over_ssh=over_ssh)


def start_banner(host: str, port: int, *, info: HostInfo | None = None) -> str:
    """The lines ``automil viz start`` prints once the server is bound."""
    info = info or host_info()
    lines = [f"Dashboard: http://{host}:{port}"]
    if info.over_ssh:
        lines.append("From your own machine, forward the port and open http://localhost:%d" % port)
        lines.append(f"  {info.tunnel_command(port)}")
        lines.append("  (or open https://automil.org and connect it to http://localhost:%d)" % port)
    lines.append(f"Another server on this port? Start with --port {port + 1}")
    return "\n".join(lines)
