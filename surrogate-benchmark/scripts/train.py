"""Train one surrogate (U-Net or FNO) as an autoregressive time stepper.

Stage 1: one-step training with noise injection on the input state.
Stage 2: unrolled training over K steps (the gradient goes through the
         model's own predictions), at a lower learning rate.

Model selection uses two validation Reynolds numbers taken out of the
training list. The test and out-of-distribution sets are never used here.

python scripts/train.py --model unet --minutes1 30 --minutes2 15
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbench.data import RE_TRAIN, TARGET
from sbench.dataset import load_split, Normalizer, Stepper
from sbench.models import build, n_params

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", default="data")
p.add_argument("--out", default="checkpoints")
p.add_argument("--minutes1", type=float, default=30)
p.add_argument("--minutes2", type=float, default=15)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--noise", type=float, default=0.5, help="input noise, in units of the one-step change std")
p.add_argument("--unroll", type=int, default=4)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
np.random.seed(a.seed)

RE_VAL = [78.0, 126.0]
re_fit = [r for r in RE_TRAIN if r not in RE_VAL]
fit = load_split(re_fit, a.data)
val = load_split(RE_VAL, a.data)
nm = Normalizer(fit)
X = torch.tensor(np.stack([c["frames"] for c in fit]))      # (C, T, 3, ny, nx)
RE = torch.tensor(re_fit, dtype=torch.float32)
XV = torch.tensor(np.stack([c["frames"] for c in val]))
REV = torch.tensor(RE_VAL, dtype=torch.float32)
n_case, n_t = X.shape[:2]
fluid = TARGET.fluid.float()
n_fluid = fluid.sum()

model = Stepper(build(a.model), nm)
print(f"{a.model}: {n_params(model.net)} parameters; {n_case} cases x {n_t} frames", flush=True)
dstd = nm.dstd.view(1, -1, 1, 1)


def masked_mse(pred, target):
    return (((pred - target) ** 2) * fluid).sum() / (n_fluid * pred.shape[0] * pred.shape[1])


@torch.no_grad()
def validate(horizon=40, starts=(40, 120, 200, 260)):
    """Mean relative L2 error of a `horizon`-step rollout from several
    start frames, on the validation cases."""
    model.eval()
    errs = []
    for s0 in starts:
        pred = model.rollout(XV[:, s0], REV, horizon)
        true = XV[:, s0:s0 + horizon + 1]
        num = ((pred - true) ** 2).sum((2, 3, 4)).sqrt()
        den = (true ** 2).sum((2, 3, 4)).sqrt()
        errs.append((num / den)[:, 1:].mean().item())
    model.train()
    return float(np.mean(errs))


def run_stage(minutes, lr, k, log, tag):
    opt = torch.optim.AdamW(model.net.parameters(), lr=lr, weight_decay=1e-5)
    t_end = time.time() + 60 * minutes
    t_start = time.time()
    step, best = 0, (float("inf"), None)
    last_val = time.time()
    while time.time() < t_end:
        ci = torch.randint(0, n_case, (a.batch,))
        ti = torch.randint(0, n_t - k, (a.batch,))
        x = X[ci, ti]
        re = RE[ci]
        x = x + a.noise * dstd * torch.randn_like(x) * fluid
        loss = 0.0
        for j in range(1, k + 1):
            target = X[ci, ti + j]
            if k == 1:
                # One-step: compare the normalized change. The target change
                # is measured from the noisy input, so the model learns to
                # remove the noise ("pull back to the data").
                pred_d = model.delta_norm(x, re)
                loss = masked_mse(pred_d, (target - x) / dstd)
            else:
                x = model(x, re)
                loss = loss + masked_mse(x / dstd, target / dstd) / k
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        # Cosine learning-rate decay over the time budget.
        frac = min(1.0, (time.time() - t_start) / (60 * minutes))
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1 + np.cos(np.pi * frac))
        opt.step()
        step += 1
        if time.time() - last_val > 120 or time.time() >= t_end:
            v = validate()
            last_val = time.time()
            log.append(dict(stage=tag, step=step, minutes=(time.time() - t_start) / 60,
                            loss=float(loss.detach()), val=v))
            print(f"[{tag}] step {step} loss {float(loss.detach()):.4g} val {v:.4f}", flush=True)
            if v < best[0]:
                best = (v, {kk: vv.clone() for kk, vv in model.net.state_dict().items()})
    model.net.load_state_dict(best[1])
    return step, best[0]


log = []
t0 = time.time()
s1, v1 = run_stage(a.minutes1, a.lr, 1, log, "one-step")
s2, v2 = (0, v1)
if a.minutes2 > 0:
    w1 = {k: v.clone() for k, v in model.net.state_dict().items()}
    s2, v2 = run_stage(a.minutes2, a.lr * 0.2, a.unroll, log, f"unroll{a.unroll}")
    if v2 > v1:
        print("unrolled stage did not help; keep the one-step weights", flush=True)
        model.net.load_state_dict(w1)
        v2 = v1
train_s = time.time() - t0
os.makedirs(a.out, exist_ok=True)
torch.save(dict(net=model.net.state_dict(), norm=nm.state_dict(), model=a.model,
                args=vars(a), train_seconds=train_s, val=v2, log=log),
           f"{a.out}/{a.model}.pt")
print(f"done: {train_s/60:.1f} min, val one-step stage {v1:.4f}, final {v2:.4f}", flush=True)
