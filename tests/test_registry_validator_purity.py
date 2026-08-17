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
    base_commit="abc1234", primary_value=0.5, node_id="node_0001",
    created_at="2026-05-02T10:00:00Z",
))
class Clean(ModelVariant):
    def forward(self, features, coords=None):
        # Function-body I/O is allowed.
        with open("/tmp/x", "w") as f:
            f.write("ok")
        return None
'''

POLICY_MODULE = CLEAN_MODULE.replace("ModelVariant", "PolicyVariant").replace(
    'kind="model"', 'kind="policy"',
)

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


def test_free_mode_model_keeps_documented_import_and_class_config_api(tmp_path):
    from automil.registry.validators.purity import PurityValidator

    body = CLEAN_MODULE.replace(
        "from automil.registry import",
        "import torch\nfrom automil.registry import",
    ).replace(
        "class Clean(ModelVariant):",
        'class Clean(ModelVariant):\n    CLAM_ARGS = {"dropout": 0.25}',
    )
    PurityValidator().check(_write_module(tmp_path, body))


def test_free_mode_policy_keeps_general_variant_import_api(tmp_path):
    from automil.registry.validators.purity import PurityValidator

    body = POLICY_MODULE.replace(
        "from automil.registry import",
        "import torch\nfrom automil.registry import",
    ).replace(
        "class Clean(PolicyVariant):",
        'class Clean(PolicyVariant):\n    OPTIONS = {"momentum": 0.9}',
    )
    PurityValidator().check(_write_module(tmp_path, body))


@pytest.mark.parametrize("class_state", ["STATE = []", "STATE = {}", "STATE = set()"])
def test_mutable_class_state_is_rejected(tmp_path, class_state):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    body = POLICY_MODULE.replace(
        "class Clean(PolicyVariant):",
        f"class Clean(PolicyVariant):\n    {class_state}",
    )
    with pytest.raises(ValidationError, match="class attributes must be immutable"):
        PurityValidator(strict_policy=True).check(_write_module(tmp_path, body))


@pytest.mark.parametrize("scope_statement", ["global COUNT", "nonlocal COUNT"])
def test_method_scope_mutation_is_rejected(tmp_path, scope_statement):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    body = POLICY_MODULE.replace(
        "        # Function-body I/O is allowed.",
        f"        {scope_statement}\n        # Function-body I/O is allowed.",
    )
    with pytest.raises(ValidationError, match="global or enclosing-scope state"):
        PurityValidator(strict_policy=True).check(_write_module(tmp_path, body))


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
        PurityValidator(strict_policy=True).check(path)


# ---------------------------------------------------------------------------
# Top-level I/O rejections
# ---------------------------------------------------------------------------

def test_open_at_module_level_rejected(tmp_path):
    from automil.registry.validators.purity import PurityValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, OPEN_AT_MODULE_LEVEL)
    with pytest.raises(ValidationError) as exc_info:
        PurityValidator(strict_policy=True).check(path)
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
# AST-only invariant and import allowlist
# ---------------------------------------------------------------------------

def test_untrusted_import_is_rejected_without_importing_it(tmp_path):
    """D-30: the AST validator rejects the import without executing it."""
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    path = _write_module(
        tmp_path,
        POLICY_MODULE.replace(
            "from automil.registry import", "import nonexistent_pkg\nfrom automil.registry import",
        ),
    )
    with pytest.raises(ValidationError, match="untrusted top-level import"):
        PurityValidator(strict_policy=True).check(path)


def test_import_alias_cannot_hide_top_level_side_effect(tmp_path):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    path = _write_module(
        tmp_path,
        POLICY_MODULE.replace(
            "from automil.registry import", "from os import system\nfrom automil.registry import",
        ),
    )
    with pytest.raises(ValidationError, match="untrusted top-level import"):
        PurityValidator(strict_policy=True).check(path)


@pytest.mark.parametrize(
    "body",
    [
        "if True:\n    open('/tmp/unsafe', 'w')\n",
        "VALUE = make_value()\n",
        "harmless_looking_call()\n",
    ],
)
def test_all_import_time_execution_is_rejected(tmp_path, body):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    with pytest.raises(ValidationError, match="top-level|module attributes"):
        PurityValidator(strict_policy=True).check(_write_module(tmp_path, body))


@pytest.mark.parametrize(
    "annotation_site",
    [
        "LEAK: open('/tmp/leak', 'w') = 1\n",
        "class Helper:\n    LEAK: open('/tmp/leak', 'w') = 1\n",
    ],
)
def test_annotation_calls_are_rejected(tmp_path, annotation_site):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    body = POLICY_MODULE.replace(
        "from __future__ import annotations\n", "",
    ).replace(
        "from automil.registry import", annotation_site + "from automil.registry import",
    )
    with pytest.raises(ValidationError, match="annotation contains a call"):
        PurityValidator().check(_write_module(tmp_path, body))


def test_postponed_annotation_calls_are_not_executed(tmp_path):
    from automil.registry.validators.purity import PurityValidator

    body = POLICY_MODULE.replace(
        "from automil.registry import",
        "LEAK: open('/tmp/not-executed', 'w') = 1\nfrom automil.registry import",
    )
    PurityValidator(strict_policy=True).check(_write_module(tmp_path, body))


@pytest.mark.parametrize(
    "write",
    [
        "type(self).COUNT += 1",
        "Clean.COUNT = 2",
        "helper.calls += 1",
        "alias = helper\n        alias.calls += 1",
    ],
)
def test_policy_shared_class_or_helper_writes_are_rejected(tmp_path, write):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    body = POLICY_MODULE.replace(
        "CONST = \"ok\"",
        "CONST = \"ok\"\ndef helper():\n    return None\n",
    ).replace(
        "        # Function-body I/O is allowed.",
        f"        {write}\n        # Function-body I/O is allowed.",
    )
    with pytest.raises(ValidationError, match="shared class/helper state"):
        PurityValidator(strict_policy=True).check(_write_module(tmp_path, body))


@pytest.mark.parametrize(
    "assignment",
    [
        "PolicyVariant.LEAK = 1",
        "PolicyVariant.LEAK: int = 1",
        "PolicyVariant.LEAK += 1",
    ],
)
def test_strict_policy_rejects_module_scope_attribute_assignment(
    tmp_path, assignment,
):
    from automil.registry.errors import ValidationError
    from automil.registry.validators.purity import PurityValidator

    body = POLICY_MODULE.replace(
        "CONST = \"ok\"", f"CONST = \"ok\"\n{assignment}",
    )
    with pytest.raises(ValidationError, match="attributes|constants cannot be mutated"):
        PurityValidator(strict_policy=True).check(_write_module(tmp_path, body))


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
