"""Create the tiny waveform photon library used by the portable example."""

from pathlib import Path
import argparse

import h5py
import numpy as np


def create_tiny_photonlib(path: str | Path) -> Path:
    path = Path(path)
    shape = np.array([2, 2, 2], dtype=np.int64)
    n_voxels = int(np.prod(shape))
    n_pmts = 2
    n_ticks = 8

    ticks = np.arange(n_ticks, dtype=np.float32)
    visibility = np.empty((n_voxels, n_pmts, n_ticks), dtype=np.float32)
    for voxel in range(n_voxels):
        for pmt in range(n_pmts):
            center = 1.5 + ((voxel + pmt) % 4)
            amplitude = 0.02 * (1 + voxel + pmt)
            visibility[voxel, pmt] = amplitude * np.exp(
                -0.5 * ((ticks - center) / 1.2) ** 2
            )

    with h5py.File(path, "w") as output:
        output.create_dataset("numvox", data=shape)
        output.create_dataset("min", data=np.array([-1.0, -1.0, -1.0]))
        output.create_dataset("max", data=np.array([1.0, 1.0, 1.0]))
        output.create_dataset("vis", data=visibility)

    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="tiny_photonlib.h5")
    args = parser.parse_args()
    print(create_tiny_photonlib(args.output))


if __name__ == "__main__":
    main()
