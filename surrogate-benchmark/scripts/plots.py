"""Make the figures and the summary table from results/*.json.

python scripts/plots.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbench.data import SPLITS, RE_TRAIN, TARGET, D_REF, T_END
from sbench.dataset import load_case

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
R, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)
acc = json.load(open(f"{R}/accuracy.json"))
spd = json.load(open(f"{R}/speed.json"))
res = acc["results"]
t = np.array(acc["t"])
SOLVERS = ["solver_d8", "solver_d10", "solver_d12", "solver_d16"]
MODELS = [m for m in ("unet", "fno") if m in next(iter(res.values()))]
LABEL = {"solver_d8": "LBM 8 cells/D", "solver_d10": "LBM 10 cells/D", "solver_d12": "LBM 12 cells/D",
         "solver_d16": "LBM 16 cells/D", "unet": "U-Net surrogate", "fno": "FNO surrogate"}
COLOR = {"solver_d8": "#9ecae1", "solver_d10": "#6baed6", "solver_d12": "#3182bd",
         "solver_d16": "#08519c", "unet": "#e6550d", "fno": "#31a354"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


GROUPS = {"test": lambda k: k.startswith("test:"),
          "low": lambda k: k.startswith("ood:") and float(k.split(":")[1]) < 70,
          "high": lambda k: k.startswith("ood:") and float(k.split(":")[1]) > 150}


def keys(split):
    if split in GROUPS:
        return [k for k in res if GROUPS[split](k)]
    return [k for k in res if k.startswith(split + ":")]


def avg(split, method, metric):
    return float(np.nanmean([res[k][method][metric] for k in keys(split)]))


def st_err(split, m):
    """Mean |St/St_ref - 1| over cases where the reference sheds vortices
    (probe std > 0.05). At Re = 50 the reference does not shed by t = 80."""
    e = [abs(res[k][m]["st"] / res[k][m]["st_true"] - 1) for k in keys(split)
         if res[k][m]["amp_true"] > 0.05]
    return float(np.nanmean(e)) if e else float("nan")


# ---------- summary table ----------
rows = []
for m in SOLVERS + MODELS:
    r = dict(method=m, cost=spd[f"{m}_b8"])
    for g in GROUPS:
        r.update({f"{g}_mean_err": avg(g, m, "mean_err"), f"{g}_rms_err": avg(g, m, "rms_err"),
                  f"{g}_st_err": st_err(g, m)})
    rows.append(r)
json.dump(rows, open(f"{R}/summary.json", "w"), indent=1)
with open(f"{R}/summary.md", "w") as f:
    f.write("| method | s per query (batch 8) | test: mean-field | test: RMS-field | test: St "
            "| Re 165-200: mean-field | Re 165-200: RMS-field | Re 165-200: St | Re 50-60: mean-field "
            "| Re 50-60: RMS-field |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {LABEL[r['method']]} | {r['cost'] * T_END:.1f} | {r['test_mean_err']:.3f} | "
                f"{r['test_rms_err']:.3f} | {100 * r['test_st_err']:.1f}% | {r['high_mean_err']:.3f} | "
                f"{r['high_rms_err']:.3f} | {100 * r['high_st_err']:.1f}% | {r['low_mean_err']:.3f} | "
                f"{r['low_rms_err']:.3f} |\n")
    f.write(f"| LBM 20 cells/D (reference) | {spd['solver_d20_b8'] * T_END:.1f} | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |\n")
print(open(f"{R}/summary.md").read())


# ---------- 1. snapshots ----------
def vort(f):
    u, v = f[0], f[1]
    h = 1.0 / 10
    w = np.gradient(v, h, axis=1) - np.gradient(u, h, axis=0)
    return np.where(TARGET.fluid.numpy(), w, np.nan)


for re in (118, 180):
    i = int(75 / 0.25)
    true = load_case(f"{DATA}/d{D_REF}_re{re}.npz")["frames"][i]
    panels = [("Reference, LBM 20 cells/D", true)]
    p10 = f"{DATA}/d10_re{re}.npz"
    if os.path.exists(p10):
        panels.append(("LBM 10 cells/D", load_case(p10)["frames"][i]))
    for m in MODELS:
        panels.append((LABEL[m], np.load(f"{R}/rollout_{m}_re{re}.npy")[i // 4].astype(np.float32)))
    fig, ax = plt.subplots(len(panels), 1, figsize=(8, 1.55 * len(panels)), constrained_layout=True)
    for a_, (name, f_) in zip(ax, panels):
        a_.imshow(vort(f_), origin="lower", cmap="RdBu_r", vmin=-8, vmax=8,
                  extent=[0, 22, 0, 4.1], interpolation="nearest")
        a_.set_title(name, fontsize=9, loc="left")
        a_.set_xticks([]); a_.set_yticks([])
    tag = "test, inside training range" if re == 118 else "outside training range"
    fig.suptitle(f"Vorticity at t = 75 D/U, Re = {re} ({tag}); every model started from rest at t = 0",
                 fontsize=10)
    fig.savefig(f"{FIG}/snapshots_re{re}.png", dpi=130)
    plt.close(fig)

# ---------- 2. error vs time ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, constrained_layout=True)
for a_, split in zip(ax, ("test", "ood")):
    for m in ["solver_d10", "solver_d16"] + MODELS:
        e = np.nanmean([res[k][m]["err_t"] for k in keys(split)], 0)
        # Skip t < 2: the fluid starts at rest, so the reference is close to
        # zero and the relative error has no meaning there.
        s_ = t >= 2
        a_.plot(t[s_], e[s_], color=COLOR[m], label=LABEL[m])
    a_.axvspan(*acc["window"], color="0.92", zorder=-1)
    a_.set_xlabel("t (D/U)")
    a_.set_title({"test": "Test Re (inside 70-150)", "ood": "Re outside training range"}[split], fontsize=10)
    a_.set_yscale("log")
ax[0].set_ylabel("pointwise relative L2 error")
ax[0].legend(frameon=False, fontsize=8)
fig.savefig(f"{FIG}/error_vs_time.png", dpi=130)
plt.close(fig)

# ---------- 3. Strouhal number and wake amplitude ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
allk = keys("test") + keys("ood")
re_all = np.array([float(k.split(":")[1]) for k in allk])
o = np.argsort(re_all)
for a_, (q, lab) in zip(ax, (("st", "Strouhal number (probe in wake)"),
                              ("amp", "std of v/U at the wake probe"))):
    ref = [res[allk[i]]["solver_d10"][q + "_true"] for i in o]
    if q == "st":   # no shedding in the reference -> no Strouhal number
        ref = [v if res[allk[i]]["solver_d10"]["amp_true"] > 0.05 else np.nan for v, i in zip(ref, o)]
    a_.plot(re_all[o], ref, "k-o", ms=4, label="Reference")
    for m in ["solver_d10"] + MODELS:
        a_.plot(re_all[o], [res[allk[i]][m][q] for i in o], "o--", color=COLOR[m], ms=4, label=LABEL[m])
    a_.axvspan(min(RE_TRAIN), max(RE_TRAIN), color="0.93", zorder=-1, label="training range")
    a_.set_xlabel("Re"); a_.set_ylabel(lab)
ax[0].legend(frameon=False, fontsize=8)
fig.savefig(f"{FIG}/strouhal.png", dpi=130)
plt.close(fig)

# ---------- 4. cost vs error ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for a_, b in zip(ax, (1, 8)):
    for m in SOLVERS + MODELS:
        c = spd[f"{m}_b{b}"] * T_END
        e = avg("test", m, "rms_err")
        a_.scatter(c, e, color=COLOR[m], s=40, marker="s" if m.startswith("solver") else "o", zorder=3)
        a_.annotate(LABEL[m], (c, e), textcoords="offset points", xytext=(5, 4), fontsize=8)
    a_.axvline(spd[f"solver_d20_b{b}"] * T_END, color="k", ls=":", lw=1)
    a_.text(spd[f"solver_d20_b{b}"] * T_END, a_.get_ylim()[1] * 0.9, " reference solver", fontsize=8)
    a_.set_xscale("log"); a_.set_yscale("log")
    a_.set_xlabel("wall-clock seconds per query (80 D/U), same 4-core CPU")
    a_.set_title(f"batch {b}", fontsize=10)
ax[0].set_ylabel("RMS-field error vs reference (test set)")
fig.savefig(f"{FIG}/cost_vs_error.png", dpi=130)
plt.close(fig)

# ---------- 5. break-even ----------
be = json.load(open(f"{R}/breakeven.json")) if os.path.exists(f"{R}/breakeven.json") else None
if be:
    n = np.arange(0, be["n_max"] + 1)
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    ax.plot(n, n * be["c_ref"] / 3600, "k-", label="reference solver, 20 cells/D")
    ax.plot(n, n * be["c_match"] / 3600, color=COLOR[be["match"]], label=f"{LABEL[be['match']]} (similar error)")
    ax.plot(n, (be["c_offline"] + n * be["c_surr"]) / 3600, color=COLOR[be["model"]],
            label=f"{LABEL[be['model']]} incl. data + training")
    ax.set_xlabel("number of queries (new Re, 80 D/U each)"); ax.set_ylabel("total CPU wall-clock (h)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{FIG}/breakeven.png", dpi=130)
    plt.close(fig)
print("figures written")
