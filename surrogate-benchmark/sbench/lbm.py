"""D2Q9 lattice Boltzmann solver for 2D flow past a cylinder in a channel.

Geometry: the DFG benchmark of Schaefer and Turek (1996), case 2D-2.
Channel height H = 0.41, length L = 2.2, cylinder diameter D = 0.1,
cylinder centre at (0.2, 0.2). The cylinder is 0.005 below the channel
centre line, so vortex shedding starts without a manual perturbation.

Inlet: parabolic profile, mean velocity U. Walls: no slip. Outlet: open.
Re = U * D / nu.

Method:
- Regularized BGK collision (Latt and Chopard 2006). Option "trt" gives a
  two-relaxation-time collision; it was unstable at 10 cells per diameter
  for Re >= 150, so it is not the default.
- Full-way bounce-back on a staircase mask for walls and cylinder.
- Inlet (prescribed velocity) and outlet (density 1) by non-equilibrium
  extrapolation (Guo et al. 2002). The inlet velocity ramps up from zero.
- Forces on the cylinder by momentum exchange.

All operations are torch tensor operations, so the solver can run a batch
of cases (one Reynolds number per case) at the same time, on CPU or GPU.
This is important for a fair speed comparison with a neural surrogate,
which also runs in batches.
"""
import math
import torch

# D2Q9 velocity set. Order: rest, 4 axis, 4 diagonal.
C = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# Benchmark geometry in physical units.
H_PHYS, L_PHYS, D_PHYS, XC_PHYS, YC_PHYS = 0.41, 2.2, 0.1, 0.2, 0.2


class Geometry:
    """Grid and masks for a resolution of `d` cells per cylinder diameter."""

    def __init__(self, d: int):
        self.d = d
        self.nx = round(L_PHYS / D_PHYS * d)
        self.ny = round(H_PHYS / D_PHYS * d)
        # Cell centres in units of D, origin at the lower-left corner.
        x = (torch.arange(self.nx) + 0.5) / d
        y = (torch.arange(self.ny) + 0.5) / d
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xc, yc = XC_PHYS / D_PHYS, YC_PHYS / D_PHYS
        self.cyl = (xx - xc) ** 2 + (yy - yc) ** 2 <= 0.25
        wall = torch.zeros(self.ny, self.nx, dtype=torch.bool)
        wall[0, :] = True
        wall[-1, :] = True
        self.wall = wall
        self.solid = self.cyl | wall
        self.fluid = ~self.solid
        # Parabolic inlet profile with mean 1, zero at the wall nodes.
        h = self.ny - 2
        s = (torch.arange(self.ny, dtype=torch.float64) - 0.5) / h
        prof = 6.0 * s * (1.0 - s)
        prof[0] = 0.0
        prof[-1] = 0.0
        self.inlet_profile = prof
        # Links for momentum exchange: population i arrives at a cylinder
        # node from a fluid node.
        link = []
        for i in range(9):
            cx, cy = C[i].tolist()
            src_fluid = torch.roll(self.fluid, shifts=(cy, cx), dims=(0, 1))
            link.append(self.cyl & src_fluid)
        self.link = torch.stack(link)


class LBM:
    """Batched TRT lattice Boltzmann solver.

    Parameters
    ----------
    d : cells per diameter.
    re : 1D tensor or list of Reynolds numbers, one per batch case.
    u_lat : mean inlet velocity in lattice units (sets the Mach number).
    """

    def __init__(self, d, re, u_lat=0.05, collision="reg", dtype=torch.float32, device="cpu"):
        self.g = Geometry(d)
        self.dtype, self.device = dtype, device
        re = torch.as_tensor(re, dtype=torch.float64).reshape(-1)
        self.re = re
        self.b = len(re)
        self.u_lat = u_lat
        nu = u_lat * d / re
        tau_p = 3.0 * nu + 0.5
        tau_m = 0.25 / (tau_p - 0.5) + 0.5
        self.tau_p = tau_p
        self.om_p = (1.0 / tau_p).to(dtype).to(device).view(-1, 1, 1, 1)
        self.om_m = (1.0 / tau_m).to(dtype).to(device).view(-1, 1, 1, 1)
        self.c = C.to(dtype).to(device)
        self.w = W.to(dtype).to(device).view(1, 9, 1, 1)
        self.collision = collision
        cx, cy = C[:, 0].double(), C[:, 1].double()
        self.cc = torch.stack([cx * cx, cx * cy, cy * cy], 1).to(dtype).to(device)
        cs2 = 1.0 / 3.0
        hq = torch.stack([cx * cx - cs2, 2 * cx * cy, cy * cy - cs2], 1) * (W.double() * 4.5).view(9, 1)
        self.hq = hq.to(dtype).to(device)
        self.solid = self.g.solid.to(device)
        self.fluid = self.g.fluid.to(device)
        self.link = self.g.link.to(dtype).to(device)
        g = self.g
        u_in = torch.zeros(2, g.ny, dtype=torch.float64)
        u_in[0] = u_lat * g.inlet_profile
        self.u_in = u_in.to(dtype).to(device)
        # One lattice time step in units of D / U.
        self.dt_phys = u_lat / d
        self.reset()

    # ---- equilibrium ----
    def feq(self, rho, u):
        """rho: (B,1,ny,nx), u: (B,2,ny,nx) -> (B,9,ny,nx)."""
        cu = torch.einsum("qk,bkyx->bqyx", self.c, u)
        usq = (u * u).sum(1, keepdim=True)
        return self.w * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)

    def ramp(self, t):
        """Smooth start of the inlet velocity over T_RAMP units of D/U.
        A fast start excites a standing pressure wave between inlet and
        outlet (period about 7.6 D/U here), which decays only slowly."""
        s = min(1.0, t * self.dt_phys / self.T_RAMP)
        return math.sin(0.5 * math.pi * s) ** 2

    T_RAMP = 15.0

    def reset(self):
        """Fluid at rest. The inlet velocity ramps up from zero."""
        g = self.g
        rho = torch.ones(self.b, 1, g.ny, g.nx, dtype=self.dtype, device=self.device)
        u = torch.zeros(self.b, 2, g.ny, g.nx, dtype=self.dtype, device=self.device)
        self.f = self.feq(rho, u)
        self.t = 0
        self.force = torch.zeros(self.b, 2, dtype=self.dtype, device=self.device)

    def macro(self):
        rho = self.f.sum(1, keepdim=True)
        u = torch.einsum("qk,bqyx->bkyx", self.c, self.f) / rho
        u = u * self.fluid
        return rho, u

    def step(self, n=1):
        f = self.f
        for _ in range(n):
            rho = f.sum(1, keepdim=True)
            u = torch.einsum("qk,bqyx->bkyx", self.c, f) / rho
            fe = self.feq(rho, u)
            if self.collision == "trt":
                fo = f[:, OPP]
                feo = fe[:, OPP]
                fs, fa = 0.5 * (f + fo), 0.5 * (f - fo)
                es, ea = 0.5 * (fe + feo), 0.5 * (fe - feo)
                fpost = f - self.om_p * (fs - es) - self.om_m * (fa - ea)
            else:
                # Regularized BGK (Latt and Chopard 2006): keep only the
                # second-order (stress) part of the non-equilibrium.
                # pi = sum_q fneq_q (c_qx c_qx, c_qx c_qy, c_qy c_qy)
                pi = torch.einsum("qn,bqyx->bnyx", self.cc, f - fe)
                # fneq_q = w_q / (2 cs^4) * Q_q : pi, Q = c c - cs^2 I
                fneq = torch.einsum("qn,bnyx->bqyx", self.hq, pi)
                fpost = fe + (1.0 - self.om_p) * fneq
            # No collision inside solids: keep the bounced populations.
            fpost = torch.where(self.solid, f, fpost)
            # Streaming.
            fs_ = torch.empty_like(f)
            for i in range(9):
                cx, cy = int(C[i, 0]), int(C[i, 1])
                fs_[:, i] = torch.roll(fpost[:, i], shifts=(cy, cx), dims=(-2, -1))
            # Momentum exchange on the cylinder: every population that
            # arrives from the fluid is reversed, so it gives 2 f c.
            arr = (fs_ * self.link).sum((-2, -1))
            self.force = 2.0 * arr @ self.c
            # Full-way bounce-back in solids.
            fs_ = torch.where(self.solid, fs_[:, OPP], fs_)
            # Outlet: fixed density rho = 1 (p = 0), velocity and
            # non-equilibrium part from the neighbour column.
            fn = fs_[..., -2:-1]
            rho_n = fn.sum(1, keepdim=True)
            u_n = torch.einsum("qk,bqyx->bkyx", self.c, fn) / rho_n
            new = self.feq(torch.ones_like(rho_n), u_n) + fn - self.feq(rho_n, u_n)
            fs_[..., 1:-1, -1:] = new[..., 1:-1, :]
            # Non-equilibrium extrapolation (Guo et al. 2002): the
            # equilibrium part uses the prescribed velocity, the
            # non-equilibrium part is copied from the neighbour column.
            fn = fs_[..., 1:2]
            rho_n = fn.sum(1, keepdim=True)
            u_n = torch.einsum("qk,bqyx->bkyx", self.c, fn) / rho_n
            ramp = self.ramp(self.t + 1)
            u_in = ramp * self.u_in.view(1, 2, -1, 1).expand(self.b, 2, -1, 1)
            new = self.feq(rho_n, u_in) + fn - self.feq(rho_n, u_n)
            # Only fluid rows; the wall nodes keep their bounce-back values.
            fs_[..., 1:-1, 0:1] = new[..., 1:-1, :]
            f = fs_
            self.t += 1
        self.f = f
        return f

    # ---- outputs ----
    def fields(self):
        """Returns (B,3,ny,nx): u/U, v/U, (p - p_ref)/(rho U^2)."""
        rho, u = self.macro()
        p = (rho - 1.0) / 3.0
        out = torch.cat([u / self.u_lat, p / self.u_lat ** 2], 1)
        return out * self.fluid

    def coeffs(self):
        """Drag and lift coefficients, Cd = 2 Fx / (rho U^2 D)."""
        return 2.0 * self.force / (self.u_lat ** 2 * self.g.d)
