"""Load saved cases and build normalized model inputs."""
import glob
import os
import numpy as np
import torch

from .data import TARGET, D_REF

RE_MID, RE_SCALE = 110.0, 40.0


def load_case(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load_split(res, root, d=D_REF):
    return [load_case(os.path.join(root, f"d{d}_re{r:g}.npz")) for r in res]


def static_channels():
    """Mask, x, y as (3, ny, nx). The mask is 1 in solids."""
    ny, nx = TARGET.ny, TARGET.nx
    y = torch.linspace(-1, 1, ny).view(ny, 1).expand(ny, nx)
    x = torch.linspace(-1, 1, nx).view(1, nx).expand(ny, nx)
    return torch.stack([TARGET.solid.float(), x, y])


class Normalizer:
    """Per-channel mean/std of the state and std of the one-step change,
    from the training cases only (fluid cells only)."""

    def __init__(self, cases=None, state=None):
        if state is not None:
            self.__dict__.update({k: torch.tensor(v) for k, v in state.items()})
            return
        fl = TARGET.fluid.numpy()
        x = np.concatenate([c["frames"][:, :, fl] for c in cases])          # (N,3,cells)
        dx = np.concatenate([np.diff(c["frames"], axis=0)[:, :, fl] for c in cases])
        self.mean = torch.tensor(x.mean((0, 2)), dtype=torch.float32)
        self.std = torch.tensor(x.std((0, 2)), dtype=torch.float32)
        self.dstd = torch.tensor(dx.std((0, 2)), dtype=torch.float32)

    def state_dict(self):
        return {k: getattr(self, k).tolist() for k in ("mean", "std", "dstd")}

    def norm(self, x):
        return (x - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)


class Stepper(torch.nn.Module):
    """Wraps a network as a time stepper in physical units:
    x_{n+1} = x_n + dstd * net([norm(x_n), mask, x, y, Re])."""

    def __init__(self, net, normalizer):
        super().__init__()
        self.net = net
        self.nm = normalizer
        self.register_buffer("static", static_channels())
        self.register_buffer("fluid", TARGET.fluid.float())

    def inputs(self, x, re):
        b = x.shape[0]
        rec = ((re - RE_MID) / RE_SCALE).view(b, 1, 1, 1).expand(b, 1, *x.shape[-2:])
        return torch.cat([self.nm.norm(x), self.static.expand(b, -1, -1, -1), rec], 1)

    def delta_norm(self, x, re):
        return self.net(self.inputs(x, re))

    def forward(self, x, re):
        d = self.delta_norm(x, re) * self.nm.dstd.view(-1, 1, 1)
        return (x + d) * self.fluid

    @torch.no_grad()
    def rollout(self, x0, re, n):
        out = [x0]
        x = x0
        for _ in range(n):
            x = self(x, re)
            out.append(x)
        return torch.stack(out, 1)
