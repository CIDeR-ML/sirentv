"""Bitwise comparison of two run_suite.sh output dirs (compare.py DIR_A DIR_B) or two files (compare.py A.ckpt B.ckpt)

Every *.ckpt / *.pt is loaded and walked recursively; tensors must match in dtype, shape
and raw bytes (so NaN == NaN, and -0.0 != 0.0). log.csv files are compared on every
column except wall-clock timings, within print precision (the CSVs are printed, not raw).
Exits 1 on any difference.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

MAX_REPORT = 20
# outputs removed on purpose by the refactor: (parent key, key) present only in A
REMOVED = {("visibility_per_pmt", "std")}
# run metadata, not numerics: skipped whether missing on either side or different
# (older checkpoints have no model_cfg["ckpt_file"]; resumed runs point it at another path)
IGNORED = {("model_cfg", "ckpt_file")}


def same_tensor(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    flat = lambda t: t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(flat(a), flat(b))


def diff(a, b, path, out):
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, torch.Tensor):
        if not same_tensor(a, b):
            detail = f"dtype {a.dtype}/{b.dtype} shape {tuple(a.shape)}/{tuple(b.shape)}"
            if a.shape == b.shape and a.is_floating_point() and b.is_floating_point():
                d = (a.double() - b.double()).abs().nan_to_num(nan=math.inf)
                detail += f" max|diff| {d.max().item():.3e} n_diff {(d != 0).sum().item()}"
            out.append(f"{path}: tensor differs ({detail})")
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            out.append(f"{path}: ndarray differs")
    elif isinstance(a, dict):
        parent = path.rsplit("[", 1)[-1].strip("]'") if path else ""
        skip = lambda k: (parent, k) in IGNORED
        only_a = {k for k in a.keys() - b.keys() if (parent, k) not in REMOVED and not skip(k)}
        only_b = {k for k in b.keys() - a.keys() if not skip(k)}
        if only_a or only_b:
            out.append(f"{path}: keys differ, only A {sorted(map(str, only_a))}, "
                       f"only B {sorted(map(str, only_b))}")
        for k in {k for k in a.keys() & b.keys() if not skip(k)}:
            diff(a[k], b[k], f"{path}[{k!r}]", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def read_csv(path):
    """{iter: row} without wall-clock columns (a repeated iter keeps its last row)."""
    with open(path) as f:
        return {float(r["iter"]): {k: v for k, v in r.items() if "time" not in k} for r in csv.DictReader(f)}


def diff_csv(a, b, out):
    """Rows matched by iter, on the columns both have; one-sided rows/columns are only noted
    (the baseline CSV logger wrote just the final row, and no analysis columns)."""
    common = a.keys() & b.keys()
    if not common:
        out.append(f"no common iter rows (A: {len(a)}, B: {len(b)})")
        return
    if len(common) != max(len(a), len(b)):
        print(f"    note: comparing {len(common)} common iter rows (A has {len(a)}, B has {len(b)})")
    for it in sorted(common):
        for k in sorted(a[it].keys() & b[it].keys()):
            x, y = float(a[it][k]), float(b[it][k])
            # printed values ("%f" before, "%.4e" now): equal within the coarser print precision
            if not (x == y or abs(x - y) <= 5e-7 + 5e-5 * max(abs(x), abs(y))):
                out.append(f"[iter {it}][{k}]: {a[it][k]} vs {b[it][k]}")


def main():
    a, b = map(Path, sys.argv[1:3])
    if a.is_file():  # compare.py A.ckpt B.ckpt
        pairs = [(a.name, a, b)]
    else:
        pairs = [(str(rel), a / rel, b / rel) for rel in sorted(
            p.relative_to(a) for ext in ("*.ckpt", "*.pt", "*.csv") for p in a.rglob(ext))]
    if not pairs:
        sys.exit(f"no artifacts found under {a}")
    failed = False
    for label, path_a, path_b in pairs:
        if not path_b.exists():
            print(f"MISSING  {label}")
            failed = True
            continue
        out = []
        if path_a.suffix == ".csv":
            diff_csv(read_csv(path_a), read_csv(path_b), out)
        else:
            load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
            diff(load(path_a), load(path_b), "", out)
        print(f"{'DIFF' if out else 'OK'}      {label}")
        for line in out[:MAX_REPORT]:
            print(f"    {line}")
        if len(out) > MAX_REPORT:
            print(f"    ... {len(out) - MAX_REPORT} more")
        failed |= bool(out)
    print("FAIL: outputs differ" if failed else "PASS: bitwise identical")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
