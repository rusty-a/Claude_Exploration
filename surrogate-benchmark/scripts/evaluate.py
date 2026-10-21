"""Accuracy of the surrogates and of the coarse solvers against the
reference solution (20 cells per diameter), on the test and OOD sets.

Every method starts from the fluid at rest (t = 0) and runs to t = 80 D/U.
Pointwise errors grow when the vortex street gets out of phase, also for a
good method, so the main metrics do not depend on phase:

- mean-field error: relative L2 error of the time-averaged (u, v, p) field
- RMS-field error:  relative L2 error of the field of temporal std
- Strouhal number and amplitude of v at a probe in the wake

All averages are over the window t = 60 .. 80 D/U. At Re 70 the lift
amplitude is within 4% of its final value by t = 60 (checked on the data).

python scripts/evaluate.py
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbench.data import SPLITS, D_REF, TARGET
from sbench.dataset import load_case, Normalizer, Stepper
from sbench.models import build

WINDOW = (60.0, 80.0)
PROBE = (20, 50)          # (iy, ix) on the target grid: x = 5 D, y = 2.05 D
SOLVER_D = [8, 10, 12, 16]

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--ckpt", default="checkpoints")
p.add_argument("--models", default="unet,fno")
p.add_argument("--out", default="results")
p.add_argument("--threads", type=int, default=4)
a = p.parse_args()
torch.set_num_threads(a.threads)
fluid = TARGET.fluid.numpy()


def strouhal(sig, t):
    """Frequency from upward zero crossings, with linear interpolation."""
    x = sig - sig.mean()
    i = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(i) < 3:
        return float("nan")
    tc = t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
    return float(1.0 / np.diff(tc).mean())


def rel(a_, b_):
    return float(np.sqrt(((a_ - b_) ** 2)[..., fluid].sum() / (b_ ** 2)[..., fluid].sum()))


def metrics(pred, true, t):
    w = (t >= WINDOW[0]) & (t <= WINDOW[1])
    pw, tw = pred[w], true[w]
    pt = np.sqrt(((pred - true) ** 2)[..., fluid].sum((1, 2)) / (true ** 2)[..., fluid].sum((1, 2)))
    sp = pw[:, 1, PROBE[0], PROBE[1]]
    st = tw[:, 1, PROBE[0], PROBE[1]]
    return dict(
        mean_err=rel(pw.mean(0), tw.mean(0)),
        rms_err=rel(pw.std(0), tw.std(0)),
        st=strouhal(sp, t[w]), st_true=strouhal(st, t[w]),
        amp=float(sp.std()), amp_true=float(st.std()),
        err_t=pt.tolist(),
        finite=bool(np.isfinite(pred).all()))


def load_model(name):
    ck = torch.load(f"{a.ckpt}/{name}.pt", weights_only=False)
    net = build(ck["model"])
    net.load_state_dict(ck["net"])
    m = Stepper(net, Normalizer(state=ck["norm"]))
    m.eval()
    return m


models = {n: load_model(n) for n in a.models.split(",") if os.path.exists(f"{a.ckpt}/{n}.pt")}
results = {}
for split in ("test", "ood"):
    for re in SPLITS[split]:
        ref = load_case(f"{a.data}/d{D_REF}_re{re:g}.npz")
        true, t = ref["frames"], ref["t_frames"]
        row = {}
        for d in SOLVER_D:
            path = f"{a.data}/d{d}_re{re:g}.npz"
            if os.path.exists(path):
                row[f"solver_d{d}"] = metrics(load_case(path)["frames"], true, t)
        for name, m in models.items():
            x0 = torch.tensor(true[None, 0])
            with torch.no_grad():
                pred = m.rollout(x0, torch.tensor([re]), len(t) - 1)[0].numpy()
            pred = np.nan_to_num(pred, nan=1e3, posinf=1e3, neginf=-1e3)
            row[name] = metrics(pred, true, t)
            if re in (118.0, 180.0):
                os.makedirs(a.out, exist_ok=True)
                np.save(f"{a.out}/rollout_{name}_re{re:g}.npy", pred[::4].astype(np.float16))
        results[f"{split}:{re:g}"] = row
        print(split, re, {k: (round(v["mean_err"], 3), round(v["rms_err"], 3), round(v["st"], 3))
                          for k, v in row.items()}, "St_true", round(row[next(iter(row))]["st_true"], 3),
              flush=True)

os.makedirs(a.out, exist_ok=True)
json.dump(dict(window=WINDOW, probe=PROBE, results=results,
               t=load_case(f"{a.data}/d{D_REF}_re{SPLITS['test'][0]:g}.npz")["t_frames"].tolist()),
          open(f"{a.out}/accuracy.json", "w"))
