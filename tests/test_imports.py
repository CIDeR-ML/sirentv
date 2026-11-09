"""Every module compiles and imports (catches syntax errors and broken imports)."""
import compileall
import importlib
import pkgutil
from pathlib import Path

import pytest

import sirentv

MODULES = sorted(m.name for m in pkgutil.walk_packages(sirentv.__path__, "sirentv."))


def test_compiles():
    root = Path(sirentv.__file__).resolve().parents[1]
    assert compileall.compile_dir(root / "sirentv", quiet=1)
    assert compileall.compile_dir(root / "tests", quiet=1)


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)
