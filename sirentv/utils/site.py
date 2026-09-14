"""
Site-dependent paths for notebooks and ad-hoc scripts, resolved at import time.

Import this instead of hardcoding /sdf or /global/cfs paths, and the same
notebook runs unchanged at either site:

    from sirentv.utils import site
    site.summary()                      # print everything that was resolved
    site.LUT_N50                        # compressed 50-component PCA LUT
    site.GRAD_CACHE                     # cached gradient targets
    site.RUN_DIR                        # where training writes logs/checkpoints

REPO_DIR is derived from this file's own location rather than a per-site constant, so a
checkout anywhere is found correctly. Every value can be overridden by an environment
variable (SIRENTV_DATA_DIR, SIRENTV_RUN_DIR, ...), which is also how you point a notebook
at a one-off copy of the data without editing it.

Nothing here touches the filesystem beyond existence checks, and nothing raises on a
missing path -- `missing()` reports what is absent so a notebook can say so plainly
instead of failing several cells later with a confusing error.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------- site
if os.environ.get("NERSC_HOST"):
    SITE = "nersc"
elif Path("/sdf").is_dir():
    SITE = "s3df"
else:
    SITE = "unknown"


def _env(name: str, default):
    value = os.environ.get(name)
    return Path(value) if value else default


# sirentv/sirentv/utils/site.py -> parents[2] is the repository root (the directory
# holding setup.py, config/, notebooks/).
REPO_DIR = _env("SIRENTV_REPO_DIR", Path(__file__).resolve().parents[2])
CFG_DIR = _env("SIRENTV_CFG_DIR", REPO_DIR / "config")
GRAD_CACHE_DIR = _env("SIRENTV_GRAD_CACHE_DIR", REPO_DIR / "grad_frob_cache")

# ---------------------------------------------------------------- data files
# On NERSC both LUTs were consolidated into one CIDER-ML/data directory during the
# migration. On s3df they live in two different shared locations, so the file paths are
# given individually rather than derived from a single DATA_DIR.
if SITE == "nersc":
    DATA_DIR = _env("SIRENTV_DATA_DIR", REPO_DIR.parent / "data")
    LUT_N50 = DATA_DIR / "compressed_plib_b05_quantile_log_prod_lite_n50.h5"
    LUT_QUANTILE = DATA_DIR / "full_wvfm_plib_b05_quantile_prod_lite_fixed.h5"
    RUN_DIR = _env("SIRENTV_RUN_DIR", Path("/pscratch/sd/j/junjiex/SIREN/logs"))
    ANALYSIS_DIR = _env("SIRENTV_ANALYSIS_DIR", Path("/pscratch/sd/j/junjiex/SIREN/analysis"))
else:
    DATA_DIR = _env("SIRENTV_DATA_DIR", Path("/sdf/data/neutrino"))
    LUT_N50 = DATA_DIR / "youngsam" / "compressed_plib_b05_quantile_log_prod_lite_n50.h5"
    LUT_QUANTILE = (
        DATA_DIR / "pubdata" / "lut" / "optical" / "consolidated"
        / "full_wvfm_plib_b05_quantile_prod_lite_fixed.h5"
    )
    RUN_DIR = _env("SIRENTV_RUN_DIR", DATA_DIR / "junjie" / "SIREN" / "cider-ml" / "logs")
    ANALYSIS_DIR = _env(
        "SIRENTV_ANALYSIS_DIR",
        Path("/sdf/home/j/junjie/sdf-data/SIREN/cider-ml/analysis/sirentv"),
    )

# ------------------------------------------------------- import search paths
# sirentv imports slar from siren-lartpc and photonlib from its own checkout, both
# siblings of REPO_DIR. Separately, a few third-party packages the container image does
# not carry (hist, wandb, scikit-learn) live in a user package directory that is put on
# PYTHONPATH rather than installed into the read-only image.
WORKSPACE = REPO_DIR.parent
SIBLING_DIRS = [WORKSPACE / "siren-lartpc", WORKSPACE / "photonlib"]
if SITE == "nersc":
    PACKAGE_DIR = _env(
        "SIRENTV_PACKAGE_DIR",
        Path("/global/cfs/cdirs/m5238/users/junjiex/python_packages/py3.10"),
    )
else:
    PACKAGE_DIR = _env("SIRENTV_PACKAGE_DIR", Path(""))


def search_paths():
    """Every directory that has to be importable, in priority order.

    Only paths that exist are returned, so a site without a user package directory
    simply gets a shorter list rather than a broken PYTHONPATH.
    """
    candidates = [REPO_DIR, *SIBLING_DIRS, PACKAGE_DIR]
    seen, out = set(), []
    for c in candidates:
        s = str(c)
        if s and s not in seen and Path(c).is_dir():
            seen.add(s)
            out.append(s)
    return out


def pythonpath():
    """search_paths() as a ':'-joined string, for a subprocess or a shell command.

    Notebook cells that shell out to `python3 -m sirentv.train` need this rather than
    just REPO_DIR: without the siblings the child process dies on `import slar`, and
    without PACKAGE_DIR on `import hist`. Note that the larcv2 image declares its own
    PYTHONPATH in its ENV, so this has to be applied INSIDE the container (a value
    exported outside shifter is overridden and silently has no effect).
    """
    return ":".join(search_paths())


GRAD_CACHE = _env("SIRENTV_GRAD_CACHE", GRAD_CACHE_DIR / "dualpca_n50_grad_frob.h5")
GRAD_CACHE_QUANTILE = _env(
    "SIRENTV_GRAD_CACHE_QUANTILE", GRAD_CACHE_DIR / "quantile_grad_frob.h5"
)
PLOT_DIR = _env("SIRENTV_PLOT_DIR", ANALYSIS_DIR / "plots")

_NAMED = {
    "REPO_DIR": REPO_DIR,
    "CFG_DIR": CFG_DIR,
    "DATA_DIR": DATA_DIR,
    "LUT_N50": LUT_N50,
    "LUT_QUANTILE": LUT_QUANTILE,
    "GRAD_CACHE_DIR": GRAD_CACHE_DIR,
    "GRAD_CACHE": GRAD_CACHE,
    "GRAD_CACHE_QUANTILE": GRAD_CACHE_QUANTILE,
    "RUN_DIR": RUN_DIR,
    "ANALYSIS_DIR": ANALYSIS_DIR,
    "PLOT_DIR": PLOT_DIR,
}


def missing():
    """Names whose path does not exist. Empty means everything resolved."""
    return {k: str(v) for k, v in _NAMED.items() if not Path(v).exists()}


def summary(check: bool = True):
    """Print the resolved paths, marking any that do not exist."""
    print(f"site: {SITE}")
    width = max(len(k) for k in _NAMED)
    for key, value in _NAMED.items():
        mark = ""
        if check:
            mark = "" if Path(value).exists() else "   <-- MISSING"
        print(f"  {key:<{width}} {value}{mark}")


def ensure_importable():
    """Put every search path on sys.path, for a notebook outside an installed package.

    Returns REPO_DIR, which is what the notebooks bind to DEFAULT_SIRENTV_SRC.
    """
    import sys

    for p in reversed(search_paths()):
        if p not in sys.path:
            sys.path.insert(0, p)
    return str(REPO_DIR)
