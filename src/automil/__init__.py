"""autoMIL: Autonomous agent-driven MIL model improvement."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("automil")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "0.0.0+unknown"
