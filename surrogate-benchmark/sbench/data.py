"""Data generation: run the solver, save fields on the common target grid.

The target grid is the 10-cells-per-diameter grid (220 x 41). The
reference solution is computed at 20 cells per diameter and averaged
2 x 2 onto the target grid. The coarse-solver baselines and the surrogates
give their output on the same target grid, so all errors are computed on
the same cells.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

from .lbm import LBM, Geometry

D_REF = 20          # reference resolution, cells per diameter
D_TARGET = 10       # grid of the surrogate and of all error metrics
T_END = 80.0        # simulated time, units of D/U
DT_SAVE = 0.25      # time between saved frames, units of D/U
DT_COEF = 0.05      # time between saved force coefficients

# Reynolds numbers. Test values sit between the training values.
RE_TRAIN = [float(r) for r in np.linspace(70, 150, 21)]          # step 4
RE_TEST = [72.0, 86.0, 102.0, 118.0, 134.0, 146.0]
RE_OOD = [50.0, 60.0, 165.0, 180.0, 200.0]
SPLITS = {"train": RE_TRAIN, "test": RE_TEST, "ood": RE_OOD}

TARGET = Geometry(D_TARGET)


def to_target(fields, d):
    """Map fields (B,3,ny,nx) from a grid with d cells per diameter to the
    target grid. Cells that are solid on the target grid are set to 0."""
    if d == D_TARGET:
        out = fields
    elif d == 2 * D_TARGET:
        out = F.avg_pool2d(fields, 2)
    else:
        # Bilinear sampling at the target cell centres, in physical
        # coordinates (units of D). The grids can differ a little in height
        # because ny is rounded to an integer.
        ny, nx = fields.shape[-2:]
        ys = (torch.arange(TARGET.ny) + 0.5) / D_TARGET * d / ny
        xs = (torch.arange(TARGET.nx) + 0.5) / D_TARGET * d / nx
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        # grid_sample with align_corners=False maps -1/1 to the outer edges.
        grid = torch.stack([2 * xx - 1, 2 * yy - 1], -1)[None].to(fields.dtype)
        grid = grid.expand(fields.shape[0], -1, -1, -1)
        out = F.grid_sample(fields, grid, mode="bilinear", align_corners=False)
    return out * TARGET.fluid.to(fields.dtype)


def run(d, re, t_end=T_END, dt_save=DT_SAVE, threads=None, u_lat=0.05):
    """Run a batch of cases. Returns a dict with
    frames (B, T, 3, 41, 220) float32 on the target grid,
    coef (B, Tc, 2) drag and lift coefficients,
    t_frames, t_coef, wall-clock seconds, lattice steps."""
    if threads:
        torch.set_num_threads(threads)
    s = LBM(d, re, u_lat=u_lat)
    k_save = round(dt_save / s.dt_phys)
    k_coef = round(DT_COEF / s.dt_phys)
    assert k_save % k_coef == 0
    n_frames = round(t_end / dt_save) + 1
    frames, coef, tf, tc = [], [], [], []
    t0 = time.perf_counter()
    frames.append(to_target(s.fields(), d))
    tf.append(0.0)
    for _ in range(n_frames - 1):
        for _ in range(k_save // k_coef):
            s.step(k_coef)
            coef.append(s.coeffs().clone())
            tc.append(s.t * s.dt_phys)
        frames.append(to_target(s.fields(), d))
        tf.append(s.t * s.dt_phys)
    wall = time.perf_counter() - t0
    return dict(
        frames=torch.stack(frames, 1).numpy().astype(np.float32),
        coef=torch.stack(coef, 1).numpy(),
        t_frames=np.array(tf), t_coef=np.array(tc),
        wall=wall, steps=s.t, d=d, re=np.array(re), nodes=s.g.nx * s.g.ny)
