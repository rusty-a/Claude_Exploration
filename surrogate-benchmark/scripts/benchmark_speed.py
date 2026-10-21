"""Wall-clock cost of solvers and surrogates on the same CPU, same threads.

Cost unit: seconds of wall-clock time per case per D/U of simulated time.
Batch 1 is one query at a time. Batch 8 runs 8 Reynolds numbers together,
which is how both a solver and a network would be used for a parameter
sweep. Run this on an idle machine.

python scripts/benchmark_speed.py
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbench.lbm import LBM
from sbench.data import DT_SAVE, TARGET
from sbench.dataset import Normalizer, Stepper
from sbench.models import build

p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, default=4)
p.add_argument("--ckpt", default="checkpoints")
p.add_argument("--out", default="results")
a = p.parse_args()
torch.set_num_threads(a.threads)
res = {"threads": a.threads, "cpu": os.cpu_count(), "torch": torch.__version__}


def best_of(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


for d in [8, 10, 12, 16, 20]:
    for b in [1, 8]:
        s = LBM(d, [100.0] * b)
        s.step(round(1.0 / s.dt_phys))                   # warm-up
        k = round(1.0 / s.dt_phys)                       # steps per D/U
        sec = best_of(lambda: s.step(k))
        res[f"solver_d{d}_b{b}"] = sec / b
        print(f"solver d={d:2d} batch {b}: {sec / b * 1e3:8.1f} ms per case per D/U "
              f"({k} steps, {s.g.nx}x{s.g.ny})", flush=True)

for name in ["unet", "fno"]:
    path = f"{a.ckpt}/{name}.pt"
    if not os.path.exists(path):
        continue
    ck = torch.load(path, weights_only=False)
    net = build(ck["model"])
    net.load_state_dict(ck["net"])
    m = Stepper(net, Normalizer(state=ck["norm"])).eval()
    k = round(1.0 / DT_SAVE)
    for b in [1, 8]:
        x = torch.zeros(b, 3, TARGET.ny, TARGET.nx)
        re = torch.full((b,), 100.0)
        m.rollout(x, re, k)                              # warm-up
        sec = best_of(lambda: m.rollout(x, re, k))
        res[f"{name}_b{b}"] = sec / b
        print(f"{name:5s} batch {b}: {sec / b * 1e3:8.1f} ms per case per D/U", flush=True)

os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/speed.json", "w"), indent=1)
