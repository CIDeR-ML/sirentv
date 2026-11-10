"""Write a copy of a YAML config with dotted-key overrides: make_cfg.py BASE OUT key.path=value ..."""
import sys

import yaml

base, out, *overrides = sys.argv[1:]
with open(base) as f:
    cfg = yaml.safe_load(f)
for item in overrides:
    key, value = item.split("=", 1)
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)
with open(out, "w") as f:
    yaml.safe_dump(cfg, f)
