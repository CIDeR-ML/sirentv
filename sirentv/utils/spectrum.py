"""
Spatial-frequency (kx, ky, kz) power spectra of the truth fields, the gradient
targets, and a model's predictions of them.

Extracted from ex-junjie.ipynb's frequency-domain cells so it can be run as an
evaluation item instead of only interactively. Deliberately self-contained: it
imports nothing from sirentv.eval, so it works against either eval implementation
and can be wired into whichever one you settle on with a few lines.

Why this diagnostic exists: a SIREN layer sin(w0*(Wx+b)) has a characteristic
frequency scale, and the question driving the omega_0 / depth ablations is whether
the network's representable bandwidth covers the bandwidth the target field
actually needs. Comparing truth and prediction spectra at matched k answers that
directly: prediction far BELOW truth at some k means that frequency is not being
reproduced (over-smoothed); far ABOVE means the model is injecting power the data
does not have (ringing).

Conventions preserved from the notebook, each load-bearing:

* **Invalid voxels are inpainted, never constant-filled.** Filling with any
  constant creates a discontinuity at every invalid voxel. Those are scattered
  through the volume interior, so a constant fill acts like impulse noise -- which
  has a flat power spectrum, i.e. exactly the false high-frequency content this
  diagnostic would otherwise be measuring. NaN/-inf are strictly worse than 0: a
  DFT sums over every input point, so one non-finite value poisons every output
  bin, not just nearby ones.
* **x is not Hann-tapered; y and z are.** The PMTs sit on one x wall, so the
  near-PMT signal of interest lives near one edge of that axis specifically, and a
  symmetric taper would suppress precisely the region under study.
* **The FFT is unnormalized and DC is excluded.** Absolute scale is therefore not
  a calibrated PSD and is NOT comparable across different quantities (v vs t0 vs
  grad_coeffs) or across grids of different size. Truth and prediction of the SAME
  quantity do use an identical grid and FFT convention, so they ARE directly
  comparable at each k -- which is the only comparison this module makes.
* **The prediction is masked with the TRUTH's validity mask.** Otherwise the
  model's arbitrary output at invalid (voxel, PMT) locations would inject structure
  with no counterpart on the truth side.
"""

from __future__ import annotations

import numpy as np


class GridIndex:
    """Maps the LUT's flat voxel list onto its regular (nx, ny, nz) lattice.

    The LUT stores voxels as a flat (N, 3) position array; every spectrum here
    needs them back on the lattice, along with the true physical spacing so that
    |k| comes out in cycles/mm rather than cycles/sample.
    """

    def __init__(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        self.x_unique = np.unique(pos[:, 0])
        self.y_unique = np.unique(pos[:, 1])
        self.z_unique = np.unique(pos[:, 2])
        self.nx = len(self.x_unique)
        self.ny = len(self.y_unique)
        self.nz = len(self.z_unique)
        self.ix = np.searchsorted(self.x_unique, pos[:, 0])
        self.iy = np.searchsorted(self.y_unique, pos[:, 1])
        self.iz = np.searchsorted(self.z_unique, pos[:, 2])

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def spacings(self):
        """(dx, dy, dz) in the LUT's own length units (mm)."""
        return (
            float(self.x_unique[1] - self.x_unique[0]),
            float(self.y_unique[1] - self.y_unique[0]),
            float(self.z_unique[1] - self.z_unique[0]),
        )

    def to_grid(self, values, valid=None, fill_invalid=0.0, inpaint=True):
        """Scatter a (N,) or (N, C) array -- already sliced to ONE PMT -- onto the
        lattice, giving (nx, ny, nz) or (nx, ny, nz, C).

        `valid` is always 1D: validity depends on (voxel, PMT) only, never on which
        channel, so it needs no expansion against a channel axis.

        inpaint=True overwrites invalid voxels with their nearest valid neighbour's
        value (see the module docstring for why a constant fill is not an option).
        fill_invalid applies only when inpaint=False or no mask is given.
        """
        values = np.asarray(values)
        shape = self.shape if values.ndim == 1 else (*self.shape, values.shape[1])
        grid = np.full(shape, fill_invalid, dtype=np.float64)

        if valid is None:
            grid[self.ix, self.iy, self.iz] = values
            return grid

        valid = np.asarray(valid, dtype=bool)
        grid[self.ix[valid], self.iy[valid], self.iz[valid]] = values[valid]

        if inpaint:
            from scipy.ndimage import distance_transform_edt

            valid_grid = np.zeros(self.shape, dtype=bool)
            valid_grid[self.ix[valid], self.iy[valid], self.iz[valid]] = True
            # distance_transform_edt finds the nearest ZERO of its input, so pass
            # ~valid_grid (True where invalid) to get nearest-VALID indices back.
            nearest = distance_transform_edt(
                ~valid_grid, return_distances=False, return_indices=True
            )
            grid = grid[tuple(nearest)]
        return grid


def power_spectrum_3d(grid, hann=True):
    """|FFT(grid)|^2 over the first three axes only.

    A trailing channel axis (from to_grid's (N, C) path) is never transformed and
    never implicitly reduced -- select or sum over it yourself afterwards, since
    what the right reduction is depends on the comparison being made.

    hann: taper y and z but not x (see module docstring).
    """
    grid = np.asarray(grid, dtype=np.float64)
    nx, ny, nz = grid.shape[:3]
    if hann:
        window = (
            np.ones(nx)[:, None, None]
            * np.hanning(ny)[None, :, None]
            * np.hanning(nz)[None, None, :]
        )
        if grid.ndim == 4:
            window = window[..., None]
        grid = grid * window
    return np.abs(np.fft.fftn(grid, axes=(0, 1, 2))) ** 2


def axis_marginal_spectrum(power_3d, axis, spacing):
    """Marginalize a 3D power spectrum onto one axis by averaging over the other two.

    Unlike radial_average, this keeps direction: x (the PMT-facing axis, left
    un-tapered) can behave quite differently from y and z, which a radial average
    would blend away. Two-sided -- for real input the spectrum is symmetric under
    k -> -k, and plotting both sides is itself a check that the symmetry holds.
    DC is dropped.
    """
    others = tuple(a for a in (0, 1, 2) if a != axis)
    marginal = np.asarray(power_3d).mean(axis=others)
    n = marginal.shape[0]
    k = np.fft.fftshift(np.fft.fftfreq(n, d=spacing))
    marginal = np.fft.fftshift(marginal)
    keep = np.arange(n) != int(np.argmin(np.abs(k)))
    return k[keep], marginal[keep]


def radial_average(power_3d, spacings, n_bins=50):
    """Bin a 3D power spectrum into n_bins isotropic |k| shells, DC excluded.

    Returns (k_centers, mean_power). Isotropic, so it hides the x-vs-y/z asymmetry
    that axis_marginal_spectrum is there to expose -- use it for an overall
    bandwidth summary, not for the near-PMT question.
    """
    power_3d = np.asarray(power_3d)
    nx, ny, nz = power_3d.shape[:3]
    dx, dy, dz = spacings
    KX, KY, KZ = np.meshgrid(
        np.fft.fftfreq(nx, d=dx),
        np.fft.fftfreq(ny, d=dy),
        np.fft.fftfreq(nz, d=dz),
        indexing="ij",
    )
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    nonzero = k_mag > 0  # exactly the DC term, not merely "small" frequencies
    k_flat = k_mag[nonzero]
    power_flat = power_3d[nonzero]

    edges = np.linspace(0.0, float(k_flat.max()), n_bins + 1)
    idx = np.clip(np.digitize(k_flat, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=power_flat, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.clip(counts, 1, None)


def field_spectra(grid_index, fields, valid, hann=True, n_bins=50):
    """Per-axis marginals plus a radial average for each named field.

    fields: {name: (N,) or (N, C) array already sliced to one PMT}
    valid:  (N,) boolean truth-side validity mask, applied to every field

    Returns {name: {"kx": (k, p), "ky": ..., "kz": ..., "radial": (k, p)}}, with a
    channel axis (if any) summed over so each entry is one curve per axis. All
    arrays are plain numpy, so the result serializes with np.savez.
    """
    spacings = grid_index.spacings
    out = {}
    for name, values in fields.items():
        power = power_spectrum_3d(
            grid_index.to_grid(values, valid=valid), hann=hann
        )
        if power.ndim == 4:
            power = power.sum(axis=-1)
        entry = {}
        for axis, label in enumerate(("kx", "ky", "kz")):
            entry[label] = axis_marginal_spectrum(power, axis, spacings[axis])
        entry["radial"] = radial_average(power, spacings, n_bins=n_bins)
        out[name] = entry
    return out


def flatten_for_npz(spectra, prefix=""):
    """Flatten field_spectra's nested dict into {key: array} for np.savez."""
    flat = {}
    for name, entry in spectra.items():
        for label, (k, p) in entry.items():
            flat[f"{prefix}{name}/{label}/k"] = np.asarray(k)
            flat[f"{prefix}{name}/{label}/power"] = np.asarray(p)
    return flat
