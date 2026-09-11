"""Reusable PyTorch layers, baselines, data utilities, and transport-residual helpers
for the reproducibility scripts accompanying the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F

from groupoid_cnn import (
    D4, R0, R1, R2, R3, ginv, D4Rep, Z2Rep, Types, RectGeometry, Layout,
    StratifiedLayer, BulkBulk, EdgeEdge, CornerCorner, EdgeBulk, BulkEdge,
    CornerBulk, BulkCorner, CornerEdge, EdgeCorner, EdgeEdgeCorner,
    admissible_set, build_transport, eroded_pairs, fiber_matrix,
)

torch.set_default_dtype(torch.float64)

BLOCK_CLASSES = (BulkBulk, EdgeEdge, CornerCorner, EdgeBulk, BulkEdge,
                 CornerBulk, BulkCorner, CornerEdge, EdgeCorner,
                 EdgeEdgeCorner)
SRC_STRATUM = {"bulk<-bulk": "bulk", "edge<-edge (same edge)": "edge",
               "corner<-corner": "corner", "edge<-bulk": "bulk",
               "bulk<-edge": "edge", "corner<-bulk": "bulk",
               "bulk<-corner": "corner", "corner<-edge": "edge",
               "edge<-corner": "corner",
               "edge<-edge (across corner)": "edge"}
TGT_STRATUM = {"bulk<-bulk": "bulk", "edge<-edge (same edge)": "edge",
               "corner<-corner": "corner", "edge<-bulk": "edge",
               "bulk<-edge": "bulk", "corner<-bulk": "corner",
               "bulk<-corner": "bulk", "corner<-edge": "corner",
               "edge<-corner": "edge",
               "edge<-edge (across corner)": "edge"}
CROSS_BLOCKS = {"edge<-bulk", "bulk<-edge", "corner<-bulk", "bulk<-corner",
                "corner<-edge", "edge<-corner", "edge<-edge (across corner)"}


# ----------------------------------------------------------------------------
# Fast torch stratified layer (equivalent to groupoid_cnn.StratifiedLayer)
# ----------------------------------------------------------------------------

class TorchStratifiedLayer(torch.nn.Module):
    """Stratified fields as dicts:
        bulk:   (B, d_rho, H, W)   nonzero only on interior pixels
        edge:   (B, d_eps, n_edge)
        corner: (B, d_delta, 4)
    """

    def __init__(self, geom: RectGeometry, t_in: Types, t_out: Types,
                 r0: int, unconstrained: bool = False,
                 include_cross: bool = True, exclude=None, theta_std: float = 0.3):
        super().__init__()
        self.geom, self.t_in, self.t_out, self.r0 = geom, t_in, t_out, r0
        exclude = frozenset() if exclude is None else frozenset(exclude)
        W, H = geom.W, geom.H
        mask = torch.zeros(H, W)
        for (i, j) in geom.bulk_sites:
            mask[j, i] = 1.0
        self.register_buffer("interior_mask", mask[None, None])
        self.edge_index = {p: n for n, p in enumerate(geom.edge_sites)}
        self.corner_index = {p: n for n, p in enumerate(geom.corner_sites)}

        self.blocks, self.block_data = [], []
        n_par = 0
        self.param_slices = {}
        for cls in BLOCK_CLASSES:
            blk = cls(t_in, t_out, r0)
            if (not include_cross and blk.name in CROSS_BLOCKS) or blk.name in exclude:
                continue
            if unconstrained:
                m = len(blk.slots) * blk.dout * blk.din
                blk.Q = np.eye(m)
                blk.n_params = m
                blk._idx = {k: n for n, k in enumerate(blk.slots)}
            else:
                blk.solve_basis()
            self.blocks.append(blk)
            self.param_slices[blk.name] = slice(n_par, n_par + blk.n_params)
            n_par += blk.n_params
        self.n_params = n_par
        self.theta = torch.nn.Parameter(theta_std * torch.randn(n_par)
                                        / np.sqrt(max(1, len(self.blocks))))

        for blk in self.blocks:
            Q = torch.as_tensor(blk.Q)
            data = {"Q": Q, "n_slots": len(blk.slots),
                    "do": blk.dout, "di": blk.din, "name": blk.name}
            if blk.name == "bulk<-bulk":
                pos = torch.tensor([[b + r0, a + r0] for (a, b) in blk.slots])
                data["wpos"] = pos  # (n_slots, 2): (row=dy+r0, col=dx+r0)
            else:
                taps = blk.taps(geom)
                slot_of = {k: n for n, k in enumerate(blk.slots)}
                Mo = torch.as_tensor(np.stack([t[2] for t in taps]))
                Mi = torch.as_tensor(np.stack([t[4] for t in taps]))
                sidx = torch.tensor([slot_of[t[3]] for t in taps])
                src = torch.tensor([self._site_idx(SRC_STRATUM[blk.name], t[1])
                                    for t in taps])
                tgt = torch.tensor([self._site_idx(TGT_STRATUM[blk.name], t[0])
                                    for t in taps])
                data.update(Mo=Mo, Mi=Mi, sidx=sidx, src=src, tgt=tgt)
            self.block_data.append(data)
        # register non-parameter tensors as buffers so .to(dtype/device)
        # works; store buffer *names* so forward always sees fresh tensors
        for n, d in enumerate(self.block_data):
            for k, v in list(d.items()):
                if torch.is_tensor(v):
                    self.register_buffer(f"_bd{n}_{k}", v)
                    d[k] = f"_bd{n}_{k}"

    def _site_idx(self, stratum, p):
        if stratum == "bulk":
            return p[1] * self.geom.W + p[0]
        if stratum == "edge":
            return self.edge_index[p]
        return self.corner_index[p]

    def forward(self, feats: dict) -> dict:
        geom = self.geom
        B = feats["bulk"].shape[0]
        out = {
            "bulk": torch.zeros(B, self.t_out.rho.dim, geom.H, geom.W,
                                dtype=self.theta.dtype,
                                device=self.theta.device),
            "edge": torch.zeros(B, self.t_out.eps.dim, len(geom.edge_sites),
                                dtype=self.theta.dtype,
                                device=self.theta.device),
            "corner": torch.zeros(B, self.t_out.delta.dim, 4,
                                  dtype=self.theta.dtype,
                                  device=self.theta.device),
        }
        bulk_in = feats["bulk"] * self.interior_mask
        srcs = {"bulk": bulk_in.reshape(B, self.t_in.rho.dim, -1),
                "edge": feats["edge"], "corner": feats["corner"]}
        for blk, data in zip(self.blocks, self.block_data):
            g = (lambda key, d=data: getattr(self, d[key])
                 if isinstance(d[key], str) and d[key].startswith("_bd")
                 else d[key])
            th = self.theta[self.param_slices[blk.name]]
            K = (g("Q") @ th).reshape(data["n_slots"],
                                      data["do"], data["di"])
            if blk.name == "bulk<-bulk":
                w = torch.zeros(data["do"], data["di"],
                                2 * self.r0 + 1, 2 * self.r0 + 1,
                                dtype=th.dtype, device=th.device)
                w[:, :, g("wpos")[:, 0], g("wpos")[:, 1]] = \
                    K.permute(1, 2, 0)
                conv = F.conv2d(bulk_in, w, padding=self.r0)
                out["bulk"] = out["bulk"] + conv * self.interior_mask
            else:
                T = g("Mo") @ K[g("sidx")] @ g("Mi")  # (T, do, di)
                S = srcs[SRC_STRATUM[blk.name]][:, :, g("src")]
                vals = torch.einsum("tod,bdt->bot", T, S)
                tgt_stratum = TGT_STRATUM[blk.name]
                if tgt_stratum == "bulk":
                    flat = out["bulk"].reshape(B, self.t_out.rho.dim, -1)
                    flat.index_add_(2, g("tgt"), vals)
                    out["bulk"] = flat.reshape(B, self.t_out.rho.dim,
                                               geom.H, geom.W)
                else:
                    out[tgt_stratum].index_add_(2, g("tgt"), vals)
        return out


class GEQNet(torch.nn.Module):
    def __init__(self, geom, types_seq, r0, **kw):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TorchStratifiedLayer(geom, a, b, r0, **kw)
            for a, b in zip(types_seq[:-1], types_seq[1:])])

    def forward(self, feats):
        for lay in self.layers:
            feats = lay(feats)
        return feats

    def n_params(self):
        return sum(l.n_params for l in self.layers)


# --- image-format baselines --------------------------------------------------

class PaddedCNN(torch.nn.Module):
    def __init__(self, depth, cin, c, cout, k):
        super().__init__()
        chans = [cin] + [c] * (depth - 1) + [cout]
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(a, b, k, padding=k // 2, bias=False)
            for a, b in zip(chans[:-1], chans[1:])])
        for m in self.convs:
            torch.nn.init.normal_(m.weight, std=0.1)

    def forward(self, x):
        for m in self.convs:
            x = m(x)
        return x

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class SteerConv(torch.nn.Module):
    """D4-steerable convolution with zero padding (free-plane constraint)."""

    def __init__(self, rho_in: D4Rep, rho_out: D4Rep, r0: int):
        super().__init__()
        blk = BulkBulk(Types(rho_in, Z2Rep(1, 0), Z2Rep(1, 0)),
                       Types(rho_out, Z2Rep(1, 0), Z2Rep(1, 0)), r0)
        blk.solve_basis()
        self.n_par, self.r0 = blk.n_params, r0
        self.do, self.di = blk.dout, blk.din
        self.register_buffer("Q", torch.as_tensor(blk.Q))
        self.register_buffer(
            "wpos", torch.tensor([[b + r0, a + r0] for (a, b) in blk.slots]))
        self.n_slots = len(blk.slots)
        self.theta = torch.nn.Parameter(0.3 * torch.randn(self.n_par))

    def forward(self, x):
        K = (self.Q @ self.theta).reshape(self.n_slots, self.do, self.di)
        w = torch.zeros(self.do, self.di, 2 * self.r0 + 1, 2 * self.r0 + 1,
                        dtype=x.dtype, device=x.device)
        w[:, :, self.wpos[:, 0], self.wpos[:, 1]] = K.permute(1, 2, 0)
        return F.conv2d(x, w, padding=self.r0)


class SteerNet(torch.nn.Module):
    def __init__(self, reps_seq, r0):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [SteerConv(a, b, r0) for a, b in zip(reps_seq[:-1], reps_seq[1:])])

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def n_params(self):
        return sum(m.n_par for m in self.layers)


# ----------------------------------------------------------------------------
# Data: discrete Poisson-Dirichlet problem
# ----------------------------------------------------------------------------

def poisson_solver(geom: RectGeometry):
    W, H = geom.W, geom.H
    interior = [(i, j) for j in range(1, H - 1) for i in range(1, W - 1)]
    idx = {p: n for n, p in enumerate(interior)}
    n = len(interior)
    A = sp.lil_matrix((n, n))
    for p, k in idx.items():
        i, j = p
        A[k, k] = 4.0
        for q in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if q in idx:
                A[k, idx[q]] = -1.0
    A = A.tocsc()
    lu = spla.splu(A)

    def solve(f_grid, g_grid):
        rhs = np.array([f_grid[j, i] for (i, j) in interior])
        for p, k in idx.items():
            i, j = p
            for q in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if q not in idx:
                    rhs[k] += g_grid[q[1], q[0]]
        u = lu.solve(rhs)
        out = np.zeros((H, W))
        for p, k in idx.items():
            out[p[1], p[0]] = u[k]
        return out

    return solve, interior


def make_dataset(geom, n_samples, rng, solve):
    W, H = geom.W, geom.H
    n_edge = len(geom.edge_sites)
    per = 2 * (W + H) - 4
    F_, G_, U_ = [], [], []
    for _ in range(n_samples):
        f = ndi.gaussian_filter(rng.standard_normal((H, W)), 1.5,
                                mode="constant")
        f *= 6.0 / max(1e-9, f.std())
        t = ndi.gaussian_filter1d(rng.standard_normal(per), 3.0,
                                  mode="wrap")
        t *= 1.0 / max(1e-9, t.std())
        g = np.zeros((H, W))
        k = 0
        for i in range(W):
            g[0, i] = t[k]; k += 1
        for j in range(1, H):
            g[j, W - 1] = t[k]; k += 1
        for i in range(W - 2, -1, -1):
            g[H - 1, i] = t[k]; k += 1
        for j in range(H - 2, 0, -1):
            g[j, 0] = t[k]; k += 1
        u = solve(f, g)
        F_.append(f); G_.append(g); U_.append(u)
    F_, G_, U_ = map(np.stack, (F_, G_, U_))
    scale = U_.std()
    U_ = U_ / scale

    mask_int = np.zeros((H, W)); mask_int[1:-1, 1:-1] = 1.0
    mask_bd = 1.0 - mask_int
    img = np.stack([F_ * mask_int, G_ * mask_bd], axis=1)  # (B,2,H,W)

    feats = {
        "bulk": torch.as_tensor(F_[:, None] * mask_int),
        "edge": torch.as_tensor(np.stack(
            [[G_[b, p[1], p[0]] for p in geom.edge_sites]
             for b in range(n_samples)])[:, None, :]),
        "corner": torch.as_tensor(np.stack(
            [[G_[b, p[1], p[0]] for p in geom.corner_sites]
             for b in range(n_samples)])[:, None, :]),
    }
    target = torch.as_tensor(U_[:, None] * mask_int)
    return feats, torch.as_tensor(img), target, torch.as_tensor(mask_int)


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------

def rel_mse(pred, target, mask):
    num = (((pred - target) * mask) ** 2).sum()
    den = ((target * mask) ** 2).sum()
    return (num / den).item()

def take(feats, sl):
    return {k: v[sl] for k, v in feats.items()}


def _cast(x, dt):
    if isinstance(x, dict):
        return {k: v.to(dt) for k, v in x.items()}
    return x.to(dt)


def train_model(model, inputs, target, mask, test_inputs, test_target,
                steps=1200, lr=1e-2, stratified=False, batch=128, seed=0):
    dt = torch.float32
    model = model.to(dt)
    inputs, target = _cast(inputs, dt), target.to(dt)
    test_inputs, test_target = _cast(test_inputs, dt), test_target.to(dt)
    mask = mask.to(dt)
    n = target.shape[0]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = np.inf
    for s in range(steps):
        if batch < n:
            sel = torch.randperm(n, generator=gen)[:batch]
        else:
            sel = slice(None)
        opt.zero_grad()
        pred = model(take(inputs, sel) if stratified else inputs[sel])
        if stratified:
            pred = pred["bulk"]
        loss = (((pred - target[sel]) * mask) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if (s + 1) % 100 == 0 or s == steps - 1:
            with torch.no_grad():
                tp = model(test_inputs)
                if stratified:
                    tp = tp["bulk"]
                best = min(best, rel_mse(tp, test_target, mask))
    return best


# ----------------------------------------------------------------------------
# Equivariance certificates (empirical, end-to-end)
# ----------------------------------------------------------------------------

def flat_to_dict(v, geom, types):
    lay = Layout(geom, types)
    W, H = geom.W, geom.H
    bulk = np.zeros((types.rho.dim, H, W))
    edge = np.zeros((types.eps.dim, len(geom.edge_sites)))
    corner = np.zeros((types.delta.dim, 4))
    for p in geom.bulk_sites:
        bulk[:, p[1], p[0]] = v[lay.sl(p)]
    for n, p in enumerate(geom.edge_sites):
        edge[:, n] = v[lay.sl(p)]
    for n, p in enumerate(geom.corner_sites):
        corner[:, n] = v[lay.sl(p)]
    return {k: torch.as_tensor(x[None]) for k, x in
            [("bulk", bulk), ("edge", edge), ("corner", corner)]}


def dict_to_flat(d, geom, types):
    lay = Layout(geom, types)
    v = np.zeros(lay.total)
    bulk = d["bulk"][0].detach().numpy()
    edge = d["edge"][0].detach().numpy()
    corner = d["corner"][0].detach().numpy()
    for p in geom.bulk_sites:
        v[lay.sl(p)] = bulk[:, p[1], p[0]]
    for n, p in enumerate(geom.edge_sites):
        v[lay.sl(p)] = edge[:, n]
    for n, p in enumerate(geom.corner_sites):
        v[lay.sl(p)] = corner[:, n]
    return v


def residual_stratified(net, geom, t_in, t_out, A, t, pairs, rng, n_probe=4):
    if not pairs:
        return 0.0
    Lam_i, PU_i, _ = build_transport(geom, t_in, A, pairs)
    Lam_o, PU_o, PUp_o = build_transport(geom, t_out, A, pairs)
    worst = 0.0
    scale = 0.0
    for _ in range(n_probe):
        psi = rng.standard_normal(Lam_i.shape[1])
        with torch.no_grad():
            lhs = PUp_o @ dict_to_flat(
                net(flat_to_dict(Lam_i @ psi, geom, t_in)), geom, t_out)
            rhs = Lam_o @ (PU_o @ dict_to_flat(
                net(flat_to_dict(PU_i @ psi, geom, t_in)), geom, t_out))
        worst = max(worst, np.abs(lhs - rhs).max())
        scale = max(scale, np.abs(rhs).max())
    return worst / max(scale, 1e-12)


def residual_image(net, geom, A, t, pairs, rng, Rin, Rout, n_probe=4):
    """Certificate for image-format models. Rin/Rout: matrices by which the
    linear part A of the bisection acts on input/output channels (identity
    for scalar A1 channels, rho_h(A) for typed steerable channels)."""
    if not pairs:
        return 0.0
    W, H = geom.W, geom.H
    cin = Rin.shape[0]
    src = [p for (p, q) in pairs]
    worst, scale = 0.0, 0.0
    for _ in range(n_probe):
        psi = rng.standard_normal((1, cin, H, W))
        psi_U = np.zeros_like(psi)
        for p in src:
            psi_U[0, :, p[1], p[0]] = psi[0, :, p[1], p[0]]
        Lpsi = np.zeros_like(psi)
        for (p, q) in pairs:
            Lpsi[0, :, q[1], q[0]] = Rin @ psi[0, :, p[1], p[0]]
        with torch.no_grad():
            o1 = net(torch.as_tensor(Lpsi)).numpy()
            o2 = net(torch.as_tensor(psi_U)).numpy()
        lhs = np.zeros_like(o1)
        rhs = np.zeros_like(o1)
        for (p, q) in pairs:
            lhs[0, :, q[1], q[0]] = o1[0, :, q[1], q[0]]
            rhs[0, :, q[1], q[0]] = Rout @ o2[0, :, p[1], p[0]]
        worst = max(worst, np.abs(lhs - rhs).max())
        scale = max(scale, np.abs(rhs).max())
    return worst / max(scale, 1e-12)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

