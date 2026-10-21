"""How many queries before the surrogate pays back its data and training cost?

Offline cost = reference runs for the training set (at the clean batch-8
speed) + training wall-clock time. The comparison is made twice: against
the reference solver, and against the cheapest coarse solver whose test
RMS-field error is not larger than the surrogate's.

python scripts/breakeven.py
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbench.data import RE_TRAIN, T_END

R = "results"
acc = json.load(open(f"{R}/accuracy.json"))["results"]
spd = json.load(open(f"{R}/speed.json"))
test = [k for k in acc if k.startswith("test:")]


def err(m):
    return float(np.mean([acc[k][m]["rms_err"] for k in test]))


models = [m for m in ("unet", "fno") if m in acc[test[0]]]
best = min(models, key=err)
ck = torch.load(f"checkpoints/{best}.pt", weights_only=False)
c_ref = spd["solver_d20_b8"] * T_END
c_surr = spd[f"{best}_b8"] * T_END
c_data = len(RE_TRAIN) * c_ref
c_train = ck["train_seconds"]
solvers = sorted((spd[f"solver_d{d}_b8"] * T_END, f"solver_d{d}") for d in (8, 10, 12, 16)
                 if err(f"solver_d{d}") <= err(best))
out = dict(model=best, err_model=err(best), c_ref=c_ref, c_surr=c_surr, c_data=c_data,
           c_train=c_train, c_offline=c_data + c_train,
           n_vs_ref=(c_data + c_train) / (c_ref - c_surr) if c_ref > c_surr else None)
if not solvers:
    # Every coarse solver is more accurate than the surrogate. Compare
    # with the cheapest one; the surrogate is then dominated in accuracy.
    solvers = [(spd["solver_d8_b8"] * T_END, "solver_d8")]
    out["note"] = "every coarse solver is more accurate than the surrogate; compared with the cheapest"
c_match, match = solvers[0]
out.update(match=match, err_match=err(match), c_match=c_match,
           n_vs_match=(c_data + c_train) / (c_match - c_surr) if c_match > c_surr else None)
nb = [v for v in (out.get("n_vs_ref"), out.get("n_vs_match")) if v]
out["n_max"] = int(1.5 * max(nb)) if nb else 200
print(json.dumps(out, indent=1))
json.dump(out, open(f"{R}/breakeven.json", "w"), indent=1)
