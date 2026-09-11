"""Definitive boundary-value benchmark for Part V (Section 15).

Protocol (agreed):
  * two target operators: five-point and nine-point Poisson--Dirichlet inverses
  * per (task, seed): fresh train (400) / validation (200) / test (400) sets
  * target normalization from the n training samples actually used
  * learning rate selected on validation; early stopping on validation;
    test evaluated once, at the best-validation checkpoint
  * 10 seeds; mean and std reported
  * ablation grid: GEQ, GEQ-noEB, GEQ-noC, GEQ-noXC, GEQ-noX, UNC, STEER,
    STEER-w (width-matched), CNN-4 (parameter-matched), CNN-6, CNN-12
  * nonlinear round on the nine-point task: GEQ-nl (gated equivariant
    nonlinearities + admissible biases), CNN-6-nl (GELU)
  * transport residuals of trained operators (seed 0, n=400):
    balanced-eroded partial motions (rho_4 = 2 r0) and global D2
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from groupoid_cnn import (
    D4, R0, R1, R2, MV, D4Rep, Z2Rep, Types, RectGeometry,
    admissible_set, eroded_pairs,
)
from experiment_core import (
    TorchStratifiedLayer, GEQNet, PaddedCNN, SteerNet,
    residual_stratified, residual_image, take, rel_mse,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

W, H, R0_RAD, DEPTH = 16, 12, 2, 4
GEOM = RectGeometry(W, H)
RHO_H = D4Rep({"A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 1})
T_IN = Types(D4Rep({"A1": 1}), Z2Rep(1, 0), Z2Rep(1, 0))
T_H = Types(RHO_H, Z2Rep(2, 1), Z2Rep(1, 1))
T_OUT = Types(D4Rep({"A1": 1}), Z2Rep(1, 0), Z2Rep(1, 0))
TYPES_SEQ = [T_IN] + [T_H] * (DEPTH - 1) + [T_OUT]

SIZES = [25, 100, 400]
N_SEEDS = 10
LR_GRID = [3e-2, 1e-2, 3e-3]
STEPS_STRAT, STEPS_IMG, EVAL_EVERY = 300, 600, 50
PATIENCE_STRAT, PATIENCE_IMG = 4, 6
TASKS = ["five", "nine"]
ALL_TASKS = ["five", "nine", "cornerx"]
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
STATE_DIR = CHECKPOINT_DIR / "states"
CKPT = CHECKPOINT_DIR / "benchmark_ckpt.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

MH = np.array([[1, 0], [0, -1]])


# ---------------------------------------------------------------- solvers ---
def five_point_solver(geom):
    Wg, Hg = geom.W, geom.H
    interior = [(i, j) for j in range(1, Hg - 1) for i in range(1, Wg - 1)]
    idx = {p: n for n, p in enumerate(interior)}
    A = sp.lil_matrix((len(interior), len(interior)))
    for p, k in idx.items():
        i, j = p
        A[k, k] = 4.0
        for q in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if q in idx:
                A[k, idx[q]] = -1.0
    lu = spla.splu(A.tocsc())

    def solve(f, g):
        rhs = np.array([f[j, i] for (i, j) in interior])
        for p, k in idx.items():
            i, j = p
            for q in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if q not in idx:
                    rhs[k] += g[q[1], q[0]]
        u = lu.solve(rhs)
        out = np.zeros((Hg, Wg))
        for p, k in idx.items():
            out[p[1], p[0]] = u[k]
        return out
    return solve


def nine_point_solver(geom):
    """Mehrstellen nine-point Laplacian: 20 u_p - 4*cross - diag = 6 f_p.
    Interior equations adjacent to a corner read the corner Dirichlet value
    through the diagonal neighbour, so the inverse is corner-sensitive."""
    Wg, Hg = geom.W, geom.H
    interior = [(i, j) for j in range(1, Hg - 1) for i in range(1, Wg - 1)]
    idx = {p: n for n, p in enumerate(interior)}
    A = sp.lil_matrix((len(interior), len(interior)))
    cross = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for p, k in idx.items():
        i, j = p
        A[k, k] = 20.0
        for (di, dj) in cross:
            q = (i + di, j + dj)
            if q in idx:
                A[k, idx[q]] = -4.0
        for (di, dj) in diag:
            q = (i + di, j + dj)
            if q in idx:
                A[k, idx[q]] = -1.0
    lu = spla.splu(A.tocsc())

    def solve(f, g):
        rhs = np.array([6.0 * f[j, i] for (i, j) in interior])
        for p, k in idx.items():
            i, j = p
            for (di, dj) in cross:
                q = (i + di, j + dj)
                if q not in idx:
                    rhs[k] += 4.0 * g[q[1], q[0]]
            for (di, dj) in diag:
                q = (i + di, j + dj)
                if q not in idx:
                    rhs[k] += g[q[1], q[0]]
        u = lu.solve(rhs)
        out = np.zeros((Hg, Wg))
        for p, k in idx.items():
            out[p[1], p[0]] = u[k]
        return out
    return solve


SOLVERS = {"five": five_point_solver, "nine": nine_point_solver,
           "cornerx": nine_point_solver}


# ------------------------------------------------------------------- data ---
def raw_fields(geom, n, rng, solve, corner_only=False):
    Wg, Hg = geom.W, geom.H
    per = 2 * (Wg + Hg) - 4
    F_, G_, U_ = [], [], []
    for _ in range(n):
        if corner_only:
            f = np.zeros((Hg, Wg))
            g = np.zeros((Hg, Wg))
            for (i, j) in [(0, 0), (Wg - 1, 0), (0, Hg - 1),
                           (Wg - 1, Hg - 1)]:
                g[j, i] = rng.standard_normal()
            F_.append(f); G_.append(g); U_.append(solve(f, g))
            continue
        f = ndi.gaussian_filter(rng.standard_normal((Hg, Wg)), 1.5,
                                mode="constant")
        f *= 6.0 / max(1e-9, f.std())
        t = ndi.gaussian_filter1d(rng.standard_normal(per), 3.0, mode="wrap")
        t *= 1.0 / max(1e-9, t.std())
        g = np.zeros((Hg, Wg))
        k = 0
        for i in range(Wg):
            g[0, i] = t[k]; k += 1
        for j in range(1, Hg):
            g[j, Wg - 1] = t[k]; k += 1
        for i in range(Wg - 2, -1, -1):
            g[Hg - 1, i] = t[k]; k += 1
        for j in range(Hg - 2, 0, -1):
            g[j, 0] = t[k]; k += 1
        F_.append(f); G_.append(g); U_.append(solve(f, g))
    return map(np.stack, (F_, G_, U_))


def pack(F_, G_, U_, geom, scale):
    Hg, Wg = geom.H, geom.W
    mask_int = np.zeros((Hg, Wg)); mask_int[1:-1, 1:-1] = 1.0
    Un = U_ / scale
    n = F_.shape[0]
    feats = {
        "bulk": torch.as_tensor(F_[:, None] * mask_int),
        "edge": torch.as_tensor(np.stack(
            [[G_[b, p[1], p[0]] for p in geom.edge_sites]
             for b in range(n)])[:, None, :]),
        "corner": torch.as_tensor(np.stack(
            [[G_[b, p[1], p[0]] for p in geom.corner_sites]
             for b in range(n)])[:, None, :]),
    }
    img = torch.as_tensor(
        np.stack([F_ * mask_int, G_ * (1.0 - mask_int)], axis=1))
    target = torch.as_tensor(Un[:, None] * mask_int)
    return feats, img, target, torch.as_tensor(mask_int)


def make_splits(task, seed):
    solve = SOLVERS[task](GEOM)
    rng = np.random.default_rng(
        [{"five": 1, "nine": 2, "cornerx": 3}[task], 1000 + seed])
    co = task == "cornerx"
    tr = raw_fields(GEOM, 400, rng, solve, corner_only=co)
    va = raw_fields(GEOM, 200, rng, solve, corner_only=co)
    te = raw_fields(GEOM, 400, rng, solve, corner_only=co)
    return tuple(tr), tuple(va), tuple(te)


# -------------------------------------------------- equivariant nonlinearity
class EquivNonlin(torch.nn.Module):
    """Pointwise equivariant nonlinearity for a stratified type (Sec. 11.2):
    GELU on invariant channels, tanh on sign channels, norm gating on E."""

    def __init__(self, types: Types):
        super().__init__()
        self.types = types
        mults, i, self.plan = types.rho.mults, 0, []
        for name in ["A1", "A2", "B1", "B2", "E"]:
            for _ in range(mults.get(name, 0)):
                d = 2 if name == "E" else 1
                self.plan.append((name, i, i + d)); i += d

    @staticmethod
    def _norm_gate(v):                      # v: (B, 2, ...)
        n = torch.sqrt((v ** 2).sum(dim=1, keepdim=True) + 1e-12)
        return v * torch.tanh(n) / n

    def forward(self, feats):
        out = {}
        b = feats["bulk"]
        parts = []
        for (name, i, j) in self.plan:
            x = b[:, i:j]
            if name == "A1":
                parts.append(torch.nn.functional.gelu(x))
            elif name == "E":
                parts.append(self._norm_gate(x))
            else:                            # A2, B1, B2: odd characters
                parts.append(torch.tanh(x))
        out["bulk"] = torch.cat(parts, dim=1)
        for key, rep in [("edge", self.types.eps), ("corner", self.types.delta)]:
            x = feats[key]
            ev = torch.nn.functional.gelu(x[:, :rep.m_plus])
            od = torch.tanh(x[:, rep.m_plus:])
            out[key] = torch.cat([ev, od], dim=1)
        return out


class EquivBias(torch.nn.Module):
    """Admissible bias (Sec. 10.3): constants on invariant channels only."""

    def __init__(self, types: Types, geom: RectGeometry):
        super().__init__()
        self.types = types
        n_a1 = types.rho.mults.get("A1", 0)
        self.c_b = torch.nn.Parameter(torch.zeros(n_a1))
        self.c_e = torch.nn.Parameter(torch.zeros(types.eps.m_plus))
        self.c_c = torch.nn.Parameter(torch.zeros(types.delta.m_plus))
        mask = torch.zeros(geom.H, geom.W)
        for (i, j) in geom.bulk_sites:
            mask[j, i] = 1.0
        self.register_buffer("mask", mask[None, None])

    def forward(self, feats):
        out = dict(feats)
        n_a1 = self.c_b.shape[0]
        if n_a1:
            b = feats["bulk"].clone()
            b[:, :n_a1] = b[:, :n_a1] + self.c_b[None, :, None, None] * self.mask
            out["bulk"] = b
        for key, c in [("edge", self.c_e), ("corner", self.c_c)]:
            if c.shape[0]:
                x = feats[key].clone()
                x[:, :c.shape[0]] = x[:, :c.shape[0]] + c[None, :, None]
                out[key] = x
        return out


class GEQNonlinNet(torch.nn.Module):
    def __init__(self, geom, types_seq, r0):
        super().__init__()
        self.linear = torch.nn.ModuleList(
            [TorchStratifiedLayer(geom, a, b, r0)
             for a, b in zip(types_seq[:-1], types_seq[1:])])
        self.bias = torch.nn.ModuleList(
            [EquivBias(t, geom) for t in types_seq[1:-1]])
        self.act = torch.nn.ModuleList(
            [EquivNonlin(t) for t in types_seq[1:-1]])

    def forward(self, feats):
        for k, lay in enumerate(self.linear):
            feats = lay(feats)
            if k < len(self.act):
                feats = self.act[k](self.bias[k](feats))
        return feats

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class PaddedCNNNL(PaddedCNN):
    def forward(self, x):
        for k, m in enumerate(self.convs):
            x = m(x)
            if k < len(self.convs) - 1:
                x = torch.nn.functional.gelu(x)
        return x


# ----------------------------------------------------------------- models ---
EB = frozenset({"edge<-bulk", "bulk<-edge"})
CX = frozenset({"corner<-bulk", "bulk<-corner", "corner<-edge",
                "edge<-corner"})
XC = frozenset({"edge<-edge (across corner)"})


def build_model(name):
    if name == "GEQ":
        return GEQNet(GEOM, TYPES_SEQ, R0_RAD), "strat"
    if name == "GEQ-noEB":
        return GEQNet(GEOM, TYPES_SEQ, R0_RAD, exclude=EB), "strat"
    if name == "GEQ-noC":
        return GEQNet(GEOM, TYPES_SEQ, R0_RAD, exclude=CX), "strat"
    if name == "GEQ-noXC":
        return GEQNet(GEOM, TYPES_SEQ, R0_RAD, exclude=XC), "strat"
    if name == "GEQ-noX":
        return GEQNet(GEOM, TYPES_SEQ, R0_RAD, include_cross=False), "strat"
    if name == "UNC":
        return GEQNet(GEOM, TYPES_SEQ, R0_RAD, unconstrained=True), "strat"
    if name == "STEER":
        return SteerNet([D4Rep({"A1": 2})] + [RHO_H] * (DEPTH - 1)
                        + [D4Rep({"A1": 1})], R0_RAD), "img"
    if name == "STEER-w":
        rho2 = D4Rep({"A1": 2, "A2": 2, "B1": 2, "B2": 2, "E": 2})
        return SteerNet([D4Rep({"A1": 2})] + [rho2] * (DEPTH - 1)
                        + [D4Rep({"A1": 1})], R0_RAD), "img"
    if name == "CNN-4":
        return PaddedCNN(DEPTH, 2, 4, 1, 2 * R0_RAD + 1), "img"
    if name == "CNN-6":
        return PaddedCNN(DEPTH, 2, 6, 1, 2 * R0_RAD + 1), "img"
    if name == "CNN-12":
        return PaddedCNN(DEPTH, 2, 12, 1, 2 * R0_RAD + 1), "img"
    if name == "GEQ-nl":
        return GEQNonlinNet(GEOM, TYPES_SEQ, R0_RAD), "strat"
    if name == "CNN-6-nl":
        return PaddedCNNNL(DEPTH, 2, 6, 1, 2 * R0_RAD + 1), "img"
    raise KeyError(name)


LINEAR_MODELS = ["GEQ", "GEQ-noEB", "GEQ-noC", "GEQ-noXC", "GEQ-noX",
                 "UNC", "STEER", "STEER-w", "CNN-4", "CNN-6", "CNN-12"]
NONLIN_MODELS = ["GEQ-nl", "CNN-6-nl"]


# --------------------------------------------------------------- training ---
def _cast(x, dt):
    if isinstance(x, dict):
        return {k: v.to(dt) for k, v in x.items()}
    return x.to(dt)


def train_eval(name, task, n, seed, lr, save_state=False, data=None):
    """Train one model; select checkpoint on validation; test once."""
    (Ftr, Gtr, Utr), (Fva, Gva, Uva), (Fte, Gte, Ute) = \
        data if data is not None else make_splits(task, seed)
    scale = Utr[:n, 1:-1, 1:-1].std()          # train-only normalization
    trf, tri, tru, mask = pack(Ftr[:n], Gtr[:n], Utr[:n], GEOM, scale)
    vaf, vai, vau, _ = pack(Fva, Gva, Uva, GEOM, scale)
    tef, tei, teu, _ = pack(Fte, Gte, Ute, GEOM, scale)

    torch.manual_seed(10 * seed + 3)
    model, kind = build_model(name)
    strat = kind == "strat"
    STEPS = STEPS_STRAT if strat else STEPS_IMG
    PATIENCE = PATIENCE_STRAT if strat else PATIENCE_IMG
    dt = torch.float32
    model = model.to(dt)
    tr_in = _cast(trf if strat else tri, dt); tru = tru.to(dt)
    va_in = _cast(vaf if strat else vai, dt); vau = vau.to(dt)
    te_in = _cast(tef if strat else tei, dt); teu = teu.to(dt)
    mask = mask.to(dt)

    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    best_val, best_state, since = np.inf, None, 0
    batch = min(64, n)
    for s in range(STEPS):
        sel = torch.randperm(n, generator=gen)[:batch] if batch < n \
            else slice(None)
        opt.zero_grad()
        pred = model(take(tr_in, sel) if strat else tr_in[sel])
        if strat:
            pred = pred["bulk"]
        loss = (((pred - tru[sel]) * mask) ** 2).mean()
        loss.backward()
        opt.step(); sched.step()
        if (s + 1) % EVAL_EVERY == 0 or s == STEPS - 1:
            with torch.no_grad():
                vp = model(va_in)
                v = rel_mse(vp["bulk"] if strat else vp, vau, mask)
            if v < best_val - 1e-9:
                best_val, since = v, 0
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
            else:
                since += 1
                if since >= PATIENCE:
                    break
    model.load_state_dict(best_state)
    with torch.no_grad():
        tp = model(te_in)
        test = rel_mse(tp["bulk"] if strat else tp, teu, mask)
    if save_state:
        torch.save(best_state,
                   os.path.join(STATE_DIR, f"{task}_{name}.pt"))
    return best_val, test


# ------------------------------------------------------------------ stages --
def load_ck():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {}


def save_ck(ck):
    json.dump(ck, open(CKPT, "w"))


# LR is selected per model on validation (seed 100, n=100). The three
# GEQ block ablations reuse GEQ's selected rate (same architecture family).
LRSEL_MODELS = ["GEQ", "GEQ-noX", "UNC", "STEER", "STEER-w",
                "CNN-4", "CNN-6", "CNN-12"]
LR_FAMILY = {"GEQ-noEB": "GEQ", "GEQ-noC": "GEQ", "GEQ-noXC": "GEQ"}


def stage_lrsel(budget=None):
    t_start = time.time()
    ck = load_ck()
    work = [(t, n) for t in TASKS for n in LRSEL_MODELS]
    work += [("nine", n) for n in NONLIN_MODELS]
    for (task, name) in work:
        key = f"lr|{task}|{name}"
        if key in ck:
            continue
        data = None
        vals = {}
        for lr in LR_GRID:
            rkey = f"lrrun|{task}|{name}|{lr}"
            if rkey not in ck:
                if data is None:
                    data = make_splits(task, 100)
                v, _ = train_eval(name, task, 100, 100, lr, data=data)
                ck[rkey] = float(v)
                save_ck(ck)
                if budget and time.time() - t_start > budget:
                    print("budget reached"); return False
            vals[str(lr)] = ck[rkey]
        best = min(vals, key=vals.get)
        ck[key] = {"choice": float(best), "val": vals}
        save_ck(ck)
        print(f"[lr] {task:5s} {name:10s} -> {best} {vals}", flush=True)
        if budget and time.time() - t_start > budget:
            print("budget reached"); return False
    return True


def get_lr(ck, task, name):
    task = "nine" if task == "cornerx" else task
    return ck[f"lr|{task}|{LR_FAMILY.get(name, name)}"]["choice"]


def stage_grid(budget=None):
    t_start = time.time()
    ck = load_ck()
    CURVE = ["GEQ", "GEQ-noX", "UNC", "STEER", "STEER-w",
             "CNN-4", "CNN-6", "CNN-12"]
    ABLATE = ["GEQ-noEB", "GEQ-noC", "GEQ-noXC"]      # n = 400 only
    work = []
    for task in TASKS:
        for seed in range(N_SEEDS):
            for name in CURVE:
                for n in SIZES:
                    work.append((task, seed, name, n))
            for name in ABLATE:
                work.append((task, seed, name, 400))
            if task == "nine":
                for name in NONLIN_MODELS:
                    for n in SIZES:
                        work.append((task, seed, name, n))
    for seed in range(N_SEEDS):
        for name in ["GEQ", "GEQ-noC", "STEER", "CNN-6"]:
            work.append(("cornerx", seed, name, 100))
    # cache datasets per (task, seed) within this invocation
    cache_key, cache_val = None, None
    for (task, seed, name, n) in work:
        key = f"run|{task}|{name}|{n}|{seed}"
        if key in ck:
            continue
        if cache_key != (task, seed):
            cache_key, cache_val = (task, seed), make_splits(task, seed)
        t0 = time.time()
        lr = get_lr(ck, task, name)
        save_state = (seed == 0 and n == 400)
        val, test = train_eval(name, task, n, seed, lr,
                               save_state=save_state, data=cache_val)
        ck[key] = {"val": val, "test": test}
        save_ck(ck)
        print(f"[run] {task:5s} s{seed} {name:10s} n={n:3d} "
              f"test={test:.5f} ({time.time()-t0:.1f}s)", flush=True)
        if budget and time.time() - t_start > budget:
            print("budget reached"); return False
    print("grid complete")
    return True


BAL_RAD = (DEPTH // 2) * R0_RAD          # balanced erosion radius, Thm 8.9


def certs_for(name, task):
    model, kind = build_model(name)
    state = torch.load(os.path.join(STATE_DIR, f"{task}_{name}.pt"),
                       weights_only=True)
    model = model.to(torch.float32)
    model.load_state_dict(state)
    model = model.to(torch.float64)
    rng = np.random.default_rng(5)
    out = {}
    tests = [("translation (1,0)", R0, np.array([1, 0])),
             ("translation (2,1)", R0, np.array([2, 1])),
             ("translation (3,2)", R0, np.array([3, 2])),
             ("quarter-turn about corner", R1, np.array([W - 1, 0]))]
    for label, A, t in tests:
        er = eroded_pairs(GEOM, A, t, BAL_RAD)
        if kind == "strat":
            r = residual_stratified(model, GEOM, T_IN, T_OUT, A, t, er, rng)
        else:
            r = residual_image(model, GEOM, A, t, er, rng,
                               np.eye(2), np.eye(1))
        out[label] = {"residual": float(r), "n_pairs": len(er)}
    # global D2 of the rectangle: no erosion at any depth (Cor. 8.7)
    c = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    worst = 0.0
    for A in (MV, MH, R2):
        t = np.asarray(np.round(c - A @ c), dtype=int)
        pairs = admissible_set(GEOM, A, t)
        if kind == "strat":
            r = residual_stratified(model, GEOM, T_IN, T_OUT, A, t, pairs, rng)
        else:
            r = residual_image(model, GEOM, A, t, pairs, rng,
                               np.eye(2), np.eye(1))
        worst = max(worst, float(r))
    out["global D2 (rectangle)"] = worst
    return out


def stage_certs():
    ck = load_ck()
    for task in TASKS:
        names = LINEAR_MODELS + (NONLIN_MODELS if task == "nine" else [])
        for name in names:
            key = f"cert|{task}|{name}"
            if key in ck:
                continue
            ck[key] = certs_for(name, task)
            save_ck(ck)
            print(f"[cert] {task} {name}: "
                  + json.dumps(ck[key], default=float), flush=True)
    print("certs complete")


def stage_nl_erosion_sweep():
    """Certify the trained nonlinear GEQ model at erosion radii 4, 5, and 6.

    The balanced radius DEPTH//2 * r0 is sufficient for the linear composite,
    while the nonlinear composite is certified on the one-sided cumulative
    radius (DEPTH - 1) * r0. Empty eroded domains are stored as null/N/A.
    """
    ck = load_ck()
    if ck.get("nl_erosion_sweep") is not None:
        print("nonlinear erosion sweep already complete", flush=True)
        return ck["nl_erosion_sweep"]

    model, kind = build_model("GEQ-nl")
    if kind != "strat":
        raise RuntimeError("GEQ-nl is expected to be a stratified model")
    state_path = os.path.join(STATE_DIR, "nine_GEQ-nl.pt")
    state = torch.load(state_path, weights_only=True)
    model = model.to(torch.float32)
    model.load_state_dict(state)
    model = model.to(torch.float64)
    model.eval()

    tests = [
        ("translation (1,0)", R0, np.array([1, 0])),
        ("translation (2,1)", R0, np.array([2, 1])),
        ("quarter-turn about corner", R1, np.array([W - 1, 0])),
    ]
    radii = sorted(set([BAL_RAD, BAL_RAD + 1, (DEPTH - 1) * R0_RAD]))
    sweep = {}
    for radius in radii:
        entries = {}
        for label, A, t in tests:
            pairs = eroded_pairs(GEOM, A, t, radius)
            if not pairs:
                entries[label] = {"residual": None, "n_pairs": 0}
                continue
            # Reset the probe seed for a clean comparison across radii/tests.
            rng = np.random.default_rng(5)
            residual = residual_stratified(
                model, GEOM, T_IN, T_OUT, A, t, pairs, rng
            )
            entries[label] = {
                "residual": float(residual),
                "n_pairs": len(pairs),
            }
        sweep[str(radius)] = entries

    ck["nl_erosion_sweep"] = sweep
    save_ck(ck)
    print("nonlinear erosion sweep complete", flush=True)
    for radius, entries in sweep.items():
        print(f"  radius {radius}: " + json.dumps(entries, default=float), flush=True)
    return sweep


def stage_finalize():
    ck = load_ck()
    params = {}
    for name in LINEAR_MODELS + NONLIN_MODELS:
        m, _ = build_model(name)
        params[name] = int(m.n_params() if hasattr(m, "n_params")
                           else sum(p.numel() for p in m.parameters()))
    payload = {"grid": [W, H], "depth": DEPTH, "r0": R0_RAD,
               "sizes": SIZES, "n_seeds": N_SEEDS, "steps": [STEPS_STRAT, STEPS_IMG],
               "balanced_erosion_radius": BAL_RAD,
               "params": params,
               "lr": {f"{t}|{m}": ck.get(f"lr|{t}|{m}", {}).get("choice")
                      for t in TASKS for m in LINEAR_MODELS + NONLIN_MODELS
                      if f"lr|{t}|{m}" in ck},
               "results": {}, "certs": {},
               "nl_erosion_sweep": ck.get("nl_erosion_sweep")}
    for task in ALL_TASKS:
        names = LINEAR_MODELS + (NONLIN_MODELS if task == "nine" else [])
        payload["results"][task] = {}
        for name in names:
            payload["results"][task][name] = {}
            for n in SIZES:
                tests = [ck[f"run|{task}|{name}|{n}|{s}"]["test"]
                         for s in range(N_SEEDS)
                         if f"run|{task}|{name}|{n}|{s}" in ck]
                if not tests:
                    continue
                payload["results"][task][name][str(n)] = {
                    "mean": float(np.mean(tests)),
                    "std": float(np.std(tests, ddof=1)),
                    "median": float(np.median(tests)),
                    "q25": float(np.quantile(tests, .25)),
                    "q75": float(np.quantile(tests, .75)),
                    "n_runs": len(tests),
                    "all": [float(x) for x in tests]}
            if f"cert|{task}|{name}" in ck:
                payload["certs"].setdefault(task, {})[name] = \
                    ck[f"cert|{task}|{name}"]
    with open(OUTPUT_DIR / "benchmark_results.json", "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("finalized -> benchmark_results.json")


def run_all(budget=None):
    """Run the complete Section 15 benchmark pipeline.

    The checkpoint file makes this command resumable: completed learning-rate
    selections and training runs are skipped automatically on subsequent calls.
    """
    print("=== Full benchmark pipeline ===", flush=True)
    print("Stage 1/4: learning-rate selection", flush=True)
    if not stage_lrsel(budget):
        return False

    print("Stage 2/4: benchmark training grid", flush=True)
    if not stage_grid(budget):
        return False

    print("Stage 3/4: transport certificates", flush=True)
    stage_certs()
    stage_nl_erosion_sweep()

    print("Stage 4/4: aggregate final results", flush=True)
    stage_finalize()
    print("=== Benchmark pipeline complete ===", flush=True)
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else None

    if mode == "params":
        for name in LINEAR_MODELS + NONLIN_MODELS:
            m, _ = build_model(name)
            npar = m.n_params() if hasattr(m, "n_params") else \
                sum(p.numel() for p in m.parameters())
            print(f"{name:10s} {npar:6d}")
    elif mode == "all":
        run_all(budget)
    elif mode == "lrsel":
        stage_lrsel(budget)
    elif mode == "grid":
        stage_grid(budget)
    elif mode == "certs":
        stage_certs()
    elif mode == "nlsweep":
        stage_nl_erosion_sweep()
    elif mode == "finalize":
        stage_finalize()
    else:
        raise SystemExit(
            f"Unknown mode: {mode!r}. "
            "Use one of: all, params, lrsel, grid, certs, nlsweep, finalize."
        )
