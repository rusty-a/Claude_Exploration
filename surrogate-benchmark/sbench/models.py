"""Two surrogate architectures that map the flow state at time t to t + dt.

Input channels:  u, v, p (normalized), cylinder/wall mask, Re (normalized),
                 x and y coordinates.
Output channels: the change of u, v, p over one surrogate step (normalized).

Both models predict the change (a residual), not the new state. For a time
stepper, the change is small and has a better scale than the state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN, N_OUT = 7, 3


def _pad_to(x, m):
    """Pad H and W at the high end to a multiple of m. Returns (x, (h, w))."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return F.pad(x, (0, pw, 0, ph)), (h, w)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(4, cout)
        self.n2 = nn.GroupNorm(4, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + self.skip(x))


class UNet(nn.Module):
    """A small U-Net: 3 down-sampling levels, local convolutions only."""

    def __init__(self, width=16, levels=3):
        super().__init__()
        ch = [width * 2 ** i for i in range(levels + 1)]
        self.inp = Block(N_IN, ch[0])
        self.down = nn.ModuleList(Block(ch[i], ch[i + 1]) for i in range(levels))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(levels)))
        self.dec = nn.ModuleList(Block(2 * ch[i], ch[i]) for i in reversed(range(levels)))
        self.out = nn.Conv2d(ch[0], N_OUT, 1)
        self.levels = levels

    def forward(self, x):
        x, (h, w) = _pad_to(x, 2 ** self.levels)
        skips = []
        z = self.inp(x)
        for blk in self.down:
            skips.append(z)
            z = blk(F.avg_pool2d(z, 2))
        for up, dec in zip(self.up, self.dec):
            z = dec(torch.cat([up(z), skips.pop()], 1))
        return self.out(z)[..., :h, :w]


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        b, _, h, w = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = self.m1, self.m2
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1)
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2)
        return torch.fft.irfft2(out, s=(h, w))


class FNO2d(nn.Module):
    """Fourier neural operator (Li et al. 2021) with zero padding, because
    the channel is not periodic."""

    def __init__(self, width=24, m1=12, m2=20, layers=4, pad=8):
        super().__init__()
        self.pad = pad
        self.lift = nn.Conv2d(N_IN, width, 1)
        self.spec = nn.ModuleList(SpectralConv2d(width, width, m1, m2) for _ in range(layers))
        self.lin = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, N_OUT, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        z = F.pad(self.lift(x), (0, self.pad, 0, self.pad))
        for i, (s, l) in enumerate(zip(self.spec, self.lin)):
            z = s(z) + l(z)
            if i < len(self.spec) - 1:
                z = F.gelu(z)
        return self.proj(z[..., :h, :w])


def build(name):
    return {"unet": UNet, "fno": FNO2d}[name]()


def n_params(m):
    return sum(p.numel() for p in m.parameters())
