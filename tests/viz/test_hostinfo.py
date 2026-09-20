"""viz.hostinfo: the port-forward hint printed on a remote host."""
from __future__ import annotations

from automil.viz.hostinfo import HostInfo, host_info, start_banner


def test_host_info_detects_ssh_from_the_environment():
    assert host_info(env={"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}).over_ssh is True
    assert host_info(env={}).over_ssh is False
    info = host_info(env={"SSH_TTY": "/dev/pts/1"})
    assert info.user and info.host


def test_banner_names_the_exact_tunnel_command_only_over_ssh():
    remote = HostInfo(user="yinshuol", host="login3.fir.alliancecan.ca", over_ssh=True)
    banner = start_banner("127.0.0.1", 8420, info=remote)
    assert "Dashboard: http://127.0.0.1:8420" in banner
    assert "ssh -N -L 8420:127.0.0.1:8420 yinshuol@login3.fir.alliancecan.ca" in banner
    assert "--port 8421" in banner
    local = HostInfo(user="leo", host="laptop", over_ssh=False)
    assert "ssh -N -L" not in start_banner("127.0.0.1", 8420, info=local)
    assert remote.tunnel_command(8420, local_port=9000) == "ssh -N -L 9000:127.0.0.1:8420 yinshuol@login3.fir.alliancecan.ca"
