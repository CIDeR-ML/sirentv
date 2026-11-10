"""Run a script as `python SCRIPT ARGS...` would, with torch's deterministic algorithms on."""
import os
import runpy
import sys

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(script)
runpy.run_path(script, run_name="__main__")
