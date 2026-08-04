"""Coverage for PurityValidator (REG-03 / D-30 purity)."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module bodies for purity tests.
# ---------------------------------------------------------------------------

CLEAN_MODULE = '''
"""Clean variant module."""
from __future__ import annotations
from automil.registry import register, VariantSpec, ModelVariant

CONST = "ok"
PI = 3.14
FLAGS = (True, False)


@register(VariantSpec(
    name="clean", kind="model", parent="p",
    base_commit="abc1234", composite=0.5, node_id="node_0001",
    created_at="2026-05-02T10:00:00Z",
))
class Clean(ModelVariant):
    def forward(self, features, coords=None):
        # Function-body I/O is allowed.
        with open("/tmp/x", "w") as f:
            f.write("ok")
        return None
'''

OPEN_AT_MODULE_LEVEL = '''
"""BAD: top-level open()."""
data = open("/etc/passwd").read()  # line 3
'''

PATH_WRITE_AT_MODULE_LEVEL = '''
from pathlib import Path
Path("/tmp/x").write_text("y")  # line 3
'''

REQUESTS_AT_MODULE_LEVEL = '''
import requests
r = requests.get("http://example.com")  # line 3
'''

URLLIB_AT_MODULE_LEVEL = '''
import urllib.request
urllib.request.urlopen("http://example.com")  # line 3
'''

SOCKET_AT_MODULE_LEVEL = '''
import socket
s = socket.socket()  # line 3
'''

SUBPROCESS_AT_MODULE_LEVEL = '''
import subprocess
subprocess.run(["echo", "x"])  # line 3
'''

MUTABLE_LIST = '''
STATE = []  # line 2 — mutable module-level global
'''

MUTABLE_DICT = '''
CACHE = {}  # line 2
'''

PRINT_AT_MODULE_LEVEL = '''
print("loading...")  # line 2
'''

OS_SYSTEM = '''
import os
os.system("rm -rf /tmp/x")  # line 3
'''

OS_ENVIRON_SET = '''
import os
os.environ["X"] = "y"  # line 3
'''

IF_MAIN_BLOCK = '''
"""Library, not script."""
if __name__ == "__main__":  # line 3
    pass
'''

UNIMPORTABLE_PKG = '''
"""Module that imports a nonexistent package — purity should still pass
because purity does NOT actually import the module."""
import nonexistent_pkg
CONST = "ok"
'''


def _write_module(tmp_path: Path, body: str, name: str = "x.py") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_clean_module_passes(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    path = _write_module(tmp_path, CLEAN_MODULE)
    PurityValidator().check(path)  # no exception


def test_immutable_constants_at_module_level_ok(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    body = '''
CONST = "x"
PI = 3.14
FLAGS = (True, False)
TUP = (1, 2, 3)
NONE = None
'''
    path = _write_module(tmp_path, body)
    PurityValidator().check(path)


def test_function_body_io_ok(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    path = _write_module(tmp_path, CLEAN_MODULE)  # has open() inside method
    PurityValidator().check(path)


def test_undecorated_helper_class_is_allowed(tmp_path):
    from automil.registry.validators.purity import PurityValidator

    body = CLEAN_MODULE.replace(
        "@register(VariantSpec(",
        "class OptimizerWrapper:\n"
        "    def step(self):\n"
        "        return None\n\n\n"
        "@register(VariantSpec(",
    )
    PurityValidator().check(_write_module(tmp_path, body))


@pytest.mark.parametrize("class_state", ["STATE = []", "STATE = {}", "STATE = set()"])
def test_mutable_class_state_is_rejected(tmp_path, class_state):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    body = CLEAN_MODULE.replace(
        "class Clean(ModelVariant):",
        f"class Clean(ModelVariant):\n    {class_state}",
    )
    with pytest.raises(ValidationError, match="class attributes must be immutable"):
        PurityValidator().check(_write_module(tmp_path, body))


@pytest.mark.parametrize("scope_statement", ["global COUNT", "nonlocal COUNT"])
def test_method_scope_mutation_is_rejected(tmp_path, scope_statement):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    body = CLEAN_MODULE.replace(
        "        # Function-body I/O is allowed.",
        f"        {scope_statement}\n        # Function-body I/O is allowed.",
    )
    with pytest.raises(ValidationError, match="global or enclosing-scope state"):
        PurityValidator().check(_write_module(tmp_path, body))


@pytest.mark.parametrize("body", [
    CLEAN_MODULE.replace('name="clean"', 'name=open("/tmp/purity", "w")'),
    CLEAN_MODULE.replace(
        "def forward(self, features, coords=None):",
        'def forward(self, features, coords=open("/tmp/purity", "w")):',
    ),
    CLEAN_MODULE.replace(
        "class Clean(ModelVariant):",
        'class Clean(ModelVariant):\n    MARK = open("/tmp/purity", "w")',
    ),
])
def test_definition_time_calls_are_rejected(tmp_path, body):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    path = _write_module(tmp_path, body)
    with pytest.raises(ValidationError, match="definition-time"):
        PurityValidator().check(path)


# ---------------------------------------------------------------------------
# Top-level I/O rejections
# ---------------------------------------------------------------------------

def test_open_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, OPEN_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError) as exc_info:
        PurityValidator().check(path)
    err = exc_info.value
    assert err.validator_name == "purity"
    assert "open" in err.reason.lower()
    # Line number reported.
    assert err.line == 3


def test_path_write_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, PATH_WRITE_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError, match=r"write|filesystem|I/O"):
        PurityValidator().check(path)


def test_requests_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, REQUESTS_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError, match=r"requests|network"):
        PurityValidator().check(path)


def test_urllib_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, URLLIB_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError, match=r"urllib|network"):
        PurityValidator().check(path)


def test_socket_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, SOCKET_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError, match=r"socket|network"):
        PurityValidator().check(path)


def test_subprocess_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, SUBPROCESS_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError, match=r"subprocess|process"):
        PurityValidator().check(path)


def test_print_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, PRINT_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError, match=r"print"):
        PurityValidator().check(path)


def test_os_system_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, OS_SYSTEM)
    with pytest.raises(ValidationError, match=r"os\.|system"):
        PurityValidator().check(path)


def test_os_environ_set_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, OS_ENVIRON_SET)
    with pytest.raises(ValidationError, match=r"os\.environ|environ|env"):
        PurityValidator().check(path)


# ---------------------------------------------------------------------------
# Mutable globals
# ---------------------------------------------------------------------------

def test_mutable_list_global_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, MUTABLE_LIST)
    with pytest.raises(ValidationError, match=r"mutable|list"):
        PurityValidator().check(path)


def test_mutable_dict_global_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, MUTABLE_DICT)
    with pytest.raises(ValidationError, match=r"mutable|dict"):
        PurityValidator().check(path)


# ---------------------------------------------------------------------------
# Script smells
# ---------------------------------------------------------------------------

def test_if_main_block_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, IF_MAIN_BLOCK)
    with pytest.raises(ValidationError, match=r"__main__|library|script"):
        PurityValidator().check(path)


# ---------------------------------------------------------------------------
# AST-only invariant (NEVER imports the module)
# ---------------------------------------------------------------------------

def test_unimportable_package_does_not_crash_purity(tmp_path):
    """D-30: purity is pure AST — never imports. A module that imports a
    nonexistent package would crash the interface validator at import time,
    but purity must succeed (its job is structural, not behavioural)."""
    from automil.registry.validators.purity import PurityValidator
    path = _write_module(tmp_path, UNIMPORTABLE_PKG)
    PurityValidator().check(path)  # no exception


# ---------------------------------------------------------------------------
# Error format
# ---------------------------------------------------------------------------

def test_validation_error_format(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, OPEN_AT_MODULE_LEVEL)
    try:
        PurityValidator().check(path)
    except ValidationError as e:
        s = str(e)
        assert "purity" in s
        assert str(path) in s
        assert "open" in s.lower()


def test_line_number_reported_on_failure(tmp_path):
    """ValidationError.line should match the AST line of the offending node."""
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, OPEN_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError) as exc_info:
        PurityValidator().check(path)
    assert exc_info.value.line is not None
    assert exc_info.value.line >= 1
