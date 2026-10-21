"""Solver check against the DFG benchmark 2D-2 (Schaefer and Turek 1996), Re = 100.

Reference intervals: Cd_max 3.22 .. 3.24, Cl_max 0.99 .. 1.01, St 0.295 .. 0.305.

python scripts/validate_dfg.py --d 20 --t-end 80
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbench.lbm import LBM

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=20)
p.add_argument("--t-end", type=float, default=80.0)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", default="results")
a = p.parse_args()
torch.set_num_threads(a.threads)

s = LBM(a.d, [100.0])
every = round(0.05 / s.dt_phys)
n = round(a.t_end / s.dt_phys) // every
cd, cl, t = [], [], []
t0 = time.perf_counter()
for _ in range(n):
    s.step(every)
    c = s.coeffs()[0]
    cd.append(c[0].item()); cl.append(c[1].item()); t.append(s.t * s.dt_phys)
wall = time.perf_counter() - t0
cd, cl, t = map(np.array, (cd, cl, t))
w = t >= 0.75 * a.t_end                      # last quarter: periodic state
x = cl[w] - cl[w].mean()
i = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
tc = t[w][i] - x[i] * (t[w][i + 1] - t[w][i]) / (x[i + 1] - x[i])
res = dict(d=a.d, nx=s.g.nx, ny=s.g.ny, tau=float(s.tau_p[0]), wall_s=wall,
           cd_max=float(cd[w].max()), cd_mean=float(cd[w].mean()),
           cl_max=float(cl[w].max()), st=float(1 / np.diff(tc).mean()))
print(json.dumps(res, indent=1))
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/dfg_d{a.d}.json", "w"), indent=1)
