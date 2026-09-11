"""
Groupoid-steerable CNN layers on a bounded pixel grid (reference implementation).

Implements the discrete instantiation of the tangent-cone restricted groupoid
Gamma^tc_{W,H} of a WxH rectangle inside Z^2, with ambient group p4m = Z^2 x| D4,
and the corresponding B^rig-equivariant integral channels ("stratified layers")
described in Sections 5, 7.2 and 7.3 of the paper.

Strata:  bulk  = interior pixels        (isotropy D4)
         edge  = boundary non-corner    (isotropy Z2, axis reflection)
         corner= 4 corner pixels        (isotropy Z2, diagonal reflection)

A layer is a linear map between stratified feature fields
    (bulk: type rho in Rep(D4), edge: type eps in Rep(Z2), corner: type delta)
whose two-point kernel decomposes into blocks (bulk<-bulk, edge<-bulk, ...);
each block's kernel is parameterized by a basis of the nullspace of the
equivariance constraints (Theorem "groupoid convolution theorem"), solved
numerically and exactly (finite linear algebra, no discretization error).

The main correctness certificate is `test_partial_equivariance`, which checks

    P_{U'} . Phi . Lambda_in(b)  ==  Lambda_out(b) . P_U . Phi . P_U

to machine precision for random rigid local bisections b = (A, t) of the
groupoid (partial Euclidean motions of the grid preserving tangent cones),
i.e. the defining equation of B-equivariance of the paper.

Only numpy is required. A thin optional PyTorch wrapper is provided at the end.

Run:  python groupoid_cnn.py
"""

from __future__ import annotations

import itertools
import numpy as np

# ----------------------------------------------------------------------------
# The dihedral group D4 as 2x2 integer matrices
# ----------------------------------------------------------------------------

R0 = np.array([[1, 0], [0, 1]])
R1 = np.array([[0, -1], [1, 0]])          # rotation +90 degrees
R2 = R1 @ R1
R3 = R2 @ R1
MV = np.array([[-1, 0], [0, 1]])          # reflection across the y-axis
                                          # (fixes the reference inward normal (0,1))
MD = np.array([[0, 1], [1, 0]])           # reflection across the diagonal y=x
                                          # (fixes the reference corner bisector (1,1))

ROTATIONS = [R0, R1, R2, R3]
D4 = ROTATIONS + [Rk @ MV for Rk in ROTATIONS]

SIGMA_EDGE = MV        # nontrivial element of the reference edge isotropy Z2
SIGMA_CORNER = MD      # nontrivial element of the reference corner isotropy Z2


def gkey(g: np.ndarray) -> tuple:
    return tuple(int(v) for v in np.asarray(g).ravel())


def ginv(g: np.ndarray) -> np.ndarray:
    return np.asarray(np.round(np.linalg.inv(g)), dtype=int)


def geq(g, h) -> bool:
    return gkey(g) == gkey(h)


# --- irreducible representations of D4 --------------------------------------

def irrep_A1(g): return np.array([[1.0]])
def irrep_A2(g): return np.array([[float(round(np.linalg.det(g)))]])
def irrep_B1(g): return np.array([[1.0 if g[0, 0] != 0 else -1.0]])
def irrep_B2(g): return irrep_B1(g) * irrep_A2(g)
def irrep_E(g):  return np.asarray(g, dtype=float)

D4_IRREPS = {"A1": irrep_A1, "A2": irrep_A2, "B1": irrep_B1,
             "B2": irrep_B2, "E": irrep_E}
D4_IRREP_DIMS = {"A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 2}


class D4Rep:
    """Direct sum of D4 irreps with multiplicities, e.g. {'A1':1,'E':2}."""

    def __init__(self, mults: dict):
        self.mults = {k: v for k, v in mults.items() if v > 0}
        self.dim = sum(D4_IRREP_DIMS[k] * v for k, v in self.mults.items())

    def mat(self, g) -> np.ndarray:
        blocks = []
        for name in ["A1", "A2", "B1", "B2", "E"]:
            for _ in range(self.mults.get(name, 0)):
                blocks.append(D4_IRREPS[name](g))
        out = np.zeros((self.dim, self.dim))
        i = 0
        for b in blocks:
            d = b.shape[0]
            out[i:i + d, i:i + d] = b
            i += d
        return out


class Z2Rep:
    """Z2 representation with multiplicities (m_plus, m_minus).

    `mat(flip)` returns the matrix of the identity (flip=False) or of the
    nontrivial element (flip=True). Which reflection of D4 realizes the
    nontrivial element (axis for edges, diagonal for corners) is decided by
    the caller through the cocycle computation.
    """

    def __init__(self, m_plus: int, m_minus: int):
        self.m_plus, self.m_minus = m_plus, m_minus
        self.dim = m_plus + m_minus

    def mat(self, flip: bool) -> np.ndarray:
        d = np.ones(self.dim)
        if flip:
            d[self.m_plus:] = -1.0
        return np.diag(d)


def hom_dim(pairs) -> int:
    """dim Hom_S(rep_in, rep_out) for a finite group S given as a list of
    (M_out(s), M_in(s)) matrix pairs: (1/|S|) sum_s chi_out(s) chi_in(s^-1),
    computed as the trace of the averaging projector (works for real reps)."""
    tot = 0.0
    for Mo, Mi in pairs:
        tot += np.trace(Mo) * np.trace(np.linalg.inv(Mi))
    return int(round(tot / len(pairs)))


# ----------------------------------------------------------------------------
# Geometry of the stratified rectangle
# ----------------------------------------------------------------------------

class RectGeometry:
    """The WxH pixel rectangle with its strata, adapted frames and gauges.

    Edge gauges A_y in D4 (pure rotations) map the reference frame
    (tangent (1,0), inward normal (0,1)) of the *bottom* edge to the frame of
    the edge containing y, with counterclockwise boundary orientation:
        bottom -> R0, right -> R1, top -> R2, left -> R3.
    Corner gauges map the reference quadrant cone {x>=0, y>=0} of the
    *bottom-left* corner to the cone at each corner:
        BL -> R0, BR -> R1, TR -> R2, TL -> R3.
    """

    def __init__(self, W: int, H: int):
        assert W >= 4 and H >= 4
        self.W, self.H = W, H
        self.corners = [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]
        self.corner_gauges = {self.corners[k]: ROTATIONS[k] for k in range(4)}
        self.bulk_sites, self.edge_sites = [], []
        self.edge_gauges = {}
        for j in range(H):
            for i in range(W):
                p = (i, j)
                if p in self.corners:
                    continue
                if 1 <= i <= W - 2 and 1 <= j <= H - 2:
                    self.bulk_sites.append(p)
                elif i == 0 or i == W - 1 or j == 0 or j == H - 1:
                    self.edge_sites.append(p)
                    if j == 0:
                        self.edge_gauges[p] = R0
                    elif i == W - 1:
                        self.edge_gauges[p] = R1
                    elif j == H - 1:
                        self.edge_gauges[p] = R2
                    else:
                        self.edge_gauges[p] = R3
        self.corner_sites = list(self.corners)

    def stratum(self, p) -> str | None:
        i, j = p
        if not (0 <= i < self.W and 0 <= j < self.H):
            return None
        if p in self.corner_gauges:
            return "corner"
        if i == 0 or i == self.W - 1 or j == 0 or j == self.H - 1:
            return "edge"
        return "bulk"

    def gauge(self, p) -> np.ndarray:
        s = self.stratum(p)
        if s == "edge":
            return self.edge_gauges[p]
        if s == "corner":
            return self.corner_gauges[p]
        return R0  # bulk: ambient gauge


class Types:
    """Feature types of a stratified field: (rho, eps, delta)."""

    def __init__(self, rho: D4Rep, eps: Z2Rep, delta: Z2Rep):
        self.rho, self.eps, self.delta = rho, eps, delta

    def dim(self, stratum: str) -> int:
        return {"bulk": self.rho.dim, "edge": self.eps.dim,
                "corner": self.delta.dim}[stratum]


class Layout:
    """Flat indexing of a stratified feature field into a single vector."""

    def __init__(self, geom: RectGeometry, types: Types):
        self.geom, self.types = geom, types
        self.offsets, off = {}, 0
        for stratum, sites in [("bulk", geom.bulk_sites),
                               ("edge", geom.edge_sites),
                               ("corner", geom.corner_sites)]:
            d = types.dim(stratum)
            for p in sites:
                self.offsets[(stratum, p)] = (off, off + d)
                off += d
        self.total = off

    def sl(self, p) -> slice:
        a, b = self.offsets[(self.geom.stratum(p), p)]
        return slice(a, b)


def z2_flip(eta: np.ndarray, sigma: np.ndarray) -> bool:
    """Classify a Z2 cocycle: eta must be the identity or `sigma`."""
    if geq(eta, R0):
        return False
    if geq(eta, sigma):
        return True
    raise AssertionError("cocycle escaped Z2 -- inconsistent gauges: %s" % (eta,))


def fiber_matrix(types: Types, geom: RectGeometry, A: np.ndarray,
                 src, dst) -> np.ndarray:
    """Matrix of the arrow (linear part A) : src -> dst on gauged fibers."""
    s = geom.stratum(src)
    assert s == geom.stratum(dst)
    if s == "bulk":
        return types.rho.mat(A)
    eta = ginv(geom.gauge(dst)) @ A @ geom.gauge(src)
    if s == "edge":
        return types.eps.mat(z2_flip(eta, SIGMA_EDGE))
    return types.delta.mat(z2_flip(eta, SIGMA_CORNER))


# ----------------------------------------------------------------------------
# Constraint solving: bases of equivariant kernels per block
# ----------------------------------------------------------------------------

def _vec(M):  # row-major
    return np.asarray(M).ravel()


def _sandwich(Mo, Mi):
    """Matrix S with vec(Mo @ K @ Mi) = S @ vec(K)  (row-major vec)."""
    return np.kron(Mo, Mi.T)


def nullspace(C: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if C.shape[0] == 0:
        return np.eye(C.shape[1])
    _, s, Vt = np.linalg.svd(C)
    rank = int(np.sum(s > tol * (s[0] if s.size else 1.0)))
    return Vt[rank:].T  # columns = orthonormal basis of the nullspace


class Block:
    """A kernel block: `slots` of shape (dout, din) indexed by a config key,
    linear equivariance constraints among them, and grid `taps`.

    Subclasses fill `self.slots` (ordered list of keys), `constraints()`
    (list of (key_a, key_b, Mo, Mi) meaning slot[a] = Mo @ slot[b] @ Mi) and
    `taps(geom)` (list of (dst_pixel, src_pixel, Mo, key, Mi) meaning the
    layer matrix gets the contribution  Mo @ k[key] @ Mi  at (dst, src))."""

    name = "block"

    def __init__(self, dout: int, din: int):
        self.dout, self.din = dout, din
        self.slots: list = []

    # -- parameter basis ------------------------------------------------------
    def solve_basis(self):
        idx = {k: n for n, k in enumerate(self.slots)}
        m = self.dout * self.din
        n_unknowns = len(self.slots) * m
        rows = []
        for (ka, kb, Mo, Mi) in self.constraints():
            a, b = idx[ka], idx[kb]
            S = _sandwich(Mo, Mi)
            for r in range(m):
                row = np.zeros(n_unknowns)
                row[a * m + r] += 1.0
                row[b * m:(b + 1) * m] -= S[r]
                rows.append(row)
        C = np.array(rows) if rows else np.zeros((0, n_unknowns))
        self.Q = nullspace(C)           # (n_unknowns, n_params)
        self.n_params = self.Q.shape[1]
        self._idx = idx
        return self.n_params

    def kernels(self, theta, xp=np):
        """theta (n_params,) -> dict key -> (dout, din) slot matrix."""
        v = xp.matmul(xp.asarray(self.Q, dtype=theta.dtype), theta)
        out = {}
        m = self.dout * self.din
        for k, n in self._idx.items():
            out[k] = v[n * m:(n + 1) * m].reshape(self.dout, self.din)
        return out

    def constraints(self):
        raise NotImplementedError

    def taps(self, geom):
        raise NotImplementedError


# --- concrete blocks ---------------------------------------------------------

class BulkBulk(Block):
    name = "bulk<-bulk"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.rho.dim, t_in.rho.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(a, b) for a in range(-r0, r0 + 1)
                      for b in range(-r0, r0 + 1)]

    def constraints(self):
        cons = []
        for A in (R1, MV):          # generators of D4
            for xi in self.slots:
                Axi = tuple(int(v) for v in (A @ np.array(xi)))
                cons.append((Axi, xi, self.t_out.rho.mat(A),
                             self.t_in.rho.mat(ginv(A))))
        return cons

    def taps(self, geom):
        I = np.eye(self.dout), np.eye(self.din)
        out = []
        for y in geom.bulk_sites:
            for xi in self.slots:
                x = (y[0] + xi[0], y[1] + xi[1])
                if geom.stratum(x) == "bulk":
                    out.append((y, x, I[0], xi, I[1]))
        return out


class EdgeEdge(Block):
    name = "edge<-edge (same edge)"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.eps.dim, t_in.eps.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = list(range(-r0, r0 + 1))

    def constraints(self):
        return [(-u, u, self.t_out.eps.mat(True), self.t_in.eps.mat(True))
                for u in self.slots]

    def taps(self, geom):
        I = np.eye(self.dout), np.eye(self.din)
        out = []
        for y in geom.edge_sites:
            t = geom.gauge(y) @ np.array([1, 0])
            for u in self.slots:
                x = (y[0] + u * int(t[0]), y[1] + u * int(t[1]))
                if geom.stratum(x) == "edge" and geq(geom.gauge(x), geom.gauge(y)):
                    out.append((y, x, I[0], u, I[1]))
        return out


class CornerCorner(Block):
    name = "corner<-corner"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.delta.dim, t_in.delta.dim)
        self.t_in, self.t_out = t_in, t_out
        self.slots = ["c"]

    def constraints(self):
        return [("c", "c", self.t_out.delta.mat(True), self.t_in.delta.mat(True))]

    def taps(self, geom):
        I = np.eye(self.dout), np.eye(self.din)
        return [(p, p, I[0], "c", I[1]) for p in geom.corner_sites]


class EdgeBulk(Block):
    """Boundary read-out: edge targets, interior sources."""
    name = "edge<-bulk"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.eps.dim, t_in.rho.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(t, n) for t in range(-r0, r0 + 1) for n in range(1, r0 + 1)]

    def constraints(self):
        return [((-t, n), (t, n), self.t_out.eps.mat(True),
                 self.t_in.rho.mat(SIGMA_EDGE)) for (t, n) in self.slots]

    def taps(self, geom):
        Io = np.eye(self.dout)
        out = []
        for y in geom.edge_sites:
            A = geom.gauge(y)
            Mi = self.t_in.rho.mat(ginv(A))
            for xi in self.slots:
                v = A @ np.array(xi)
                x = (y[0] + int(v[0]), y[1] + int(v[1]))
                if geom.stratum(x) == "bulk":
                    out.append((y, x, Io, xi, Mi))
        return out


class BulkEdge(Block):
    """Boundary injection: interior targets, edge sources."""
    name = "bulk<-edge"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.rho.dim, t_in.eps.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(t, n) for t in range(-r0, r0 + 1) for n in range(1, r0 + 1)]

    def constraints(self):
        return [((-t, n), (t, n), self.t_out.rho.mat(SIGMA_EDGE),
                 self.t_in.eps.mat(True)) for (t, n) in self.slots]

    def taps(self, geom):
        Ii = np.eye(self.din)
        out = []
        for x in geom.edge_sites:          # iterate over sources
            A = geom.gauge(x)
            Mo = self.t_out.rho.mat(A)
            for zt in self.slots:
                v = A @ np.array(zt)
                y = (x[0] + int(v[0]), x[1] + int(v[1]))
                if geom.stratum(y) == "bulk":
                    out.append((y, x, Mo, zt, Ii))
        return out


class CornerBulk(Block):
    name = "corner<-bulk"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.delta.dim, t_in.rho.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(a, b) for a in range(1, r0 + 1) for b in range(1, r0 + 1)]

    def constraints(self):
        return [((b, a), (a, b), self.t_out.delta.mat(True),
                 self.t_in.rho.mat(SIGMA_CORNER)) for (a, b) in self.slots]

    def taps(self, geom):
        Io = np.eye(self.dout)
        out = []
        for p in geom.corner_sites:
            A = geom.gauge(p)
            Mi = self.t_in.rho.mat(ginv(A))
            for xi in self.slots:
                v = A @ np.array(xi)
                x = (p[0] + int(v[0]), p[1] + int(v[1]))
                if geom.stratum(x) == "bulk":
                    out.append((p, x, Io, xi, Mi))
        return out


class BulkCorner(Block):
    name = "bulk<-corner"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.rho.dim, t_in.delta.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(a, b) for a in range(1, r0 + 1) for b in range(1, r0 + 1)]

    def constraints(self):
        return [((b, a), (a, b), self.t_out.rho.mat(SIGMA_CORNER),
                 self.t_in.delta.mat(True)) for (a, b) in self.slots]

    def taps(self, geom):
        Ii = np.eye(self.din)
        out = []
        for p in geom.corner_sites:
            A = geom.gauge(p)
            Mo_g = self.t_out.rho.mat(A)
            for zt in self.slots:
                v = A @ np.array(zt)
                y = (p[0] + int(v[0]), p[1] + int(v[1]))
                if geom.stratum(y) == "bulk":
                    out.append((y, p, Mo_g, zt, Ii))
        return out


# Reference configurations near the reference (bottom-left) corner:
#   cfg 0: points (u, 0) on the bottom edge (edge gauge R0)
#   cfg 1: points (0, u) on the left edge  (edge gauge R3)
_REF_EDGE_GAUGE = {0: R0, 1: R3}
_REF_EDGE_VEC = {0: np.array([1, 0]), 1: np.array([0, 1])}


class CornerEdge(Block):
    name = "corner<-edge"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.delta.dim, t_in.eps.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(c, u) for c in (0, 1) for u in range(1, r0 + 1)]

    def constraints(self):
        # sigma_corner maps cfg0 sources to cfg1 sources; source cocycle:
        eta = ginv(_REF_EDGE_GAUGE[1]) @ SIGMA_CORNER @ _REF_EDGE_GAUGE[0]
        flip_in = z2_flip(eta, SIGMA_EDGE)
        return [((1, u), (0, u), self.t_out.delta.mat(True),
                 self.t_in.eps.mat(flip_in)) for u in range(1, self.r0 + 1)]

    def taps(self, geom):
        Io = np.eye(self.dout)
        out = []
        for p in geom.corner_sites:
            A = geom.gauge(p)
            for (c, u) in self.slots:
                v = A @ (_REF_EDGE_VEC[c] * u)
                x = (p[0] + int(v[0]), p[1] + int(v[1]))
                if geom.stratum(x) != "edge":
                    continue
                eta = ginv(geom.gauge(x)) @ A @ _REF_EDGE_GAUGE[c]
                Mi = self.t_in.eps.mat(z2_flip(eta, SIGMA_EDGE))
                out.append((p, x, Io, (c, u), np.linalg.inv(Mi)))
        return out


class EdgeCorner(Block):
    name = "edge<-corner"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.eps.dim, t_in.delta.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(c, u) for c in (0, 1) for u in range(1, r0 + 1)]

    def constraints(self):
        eta = ginv(_REF_EDGE_GAUGE[1]) @ SIGMA_CORNER @ _REF_EDGE_GAUGE[0]
        flip_out = z2_flip(eta, SIGMA_EDGE)
        return [((1, u), (0, u), self.t_out.eps.mat(flip_out),
                 self.t_in.delta.mat(True)) for u in range(1, self.r0 + 1)]

    def taps(self, geom):
        Ii = np.eye(self.din)
        out = []
        for p in geom.corner_sites:
            A = geom.gauge(p)
            for (c, u) in self.slots:
                v = A @ (_REF_EDGE_VEC[c] * u)
                y = (p[0] + int(v[0]), p[1] + int(v[1]))
                if geom.stratum(y) != "edge":
                    continue
                eta = ginv(geom.gauge(y)) @ A @ _REF_EDGE_GAUGE[c]
                Mo = self.t_out.eps.mat(z2_flip(eta, SIGMA_EDGE))
                out.append((y, p, Mo, (c, u), Ii))
        return out


class EdgeEdgeCorner(Block):
    """edge<-edge taps across a corner (target and source on perpendicular
    edges adjacent to the same corner). cfg 'A': target cfg0/source cfg1;
    cfg 'B': target cfg1/source cfg0. Slot key = (cfg, u_target, u_source)."""
    name = "edge<-edge (across corner)"

    def __init__(self, t_in: Types, t_out: Types, r0: int):
        super().__init__(t_out.eps.dim, t_in.eps.dim)
        self.t_in, self.t_out, self.r0 = t_in, t_out, r0
        self.slots = [(c, uy, ux) for c in ("A", "B")
                      for uy in range(1, r0 + 1) for ux in range(1, r0 + 1)]

    def constraints(self):
        eta = ginv(_REF_EDGE_GAUGE[1]) @ SIGMA_CORNER @ _REF_EDGE_GAUGE[0]
        f = z2_flip(eta, SIGMA_EDGE)
        cons = []
        for uy in range(1, self.r0 + 1):
            for ux in range(1, self.r0 + 1):
                cons.append((("B", uy, ux), ("A", uy, ux),
                             self.t_out.eps.mat(f),
                             np.linalg.inv(self.t_in.eps.mat(f))))
        return cons

    def taps(self, geom):
        out = []
        cfg_pair = {"A": (0, 1), "B": (1, 0)}
        for p in geom.corner_sites:
            A = geom.gauge(p)
            for (c, uy, ux) in self.slots:
                ct, cs = cfg_pair[c]
                vy = A @ (_REF_EDGE_VEC[ct] * uy)
                vx = A @ (_REF_EDGE_VEC[cs] * ux)
                y = (p[0] + int(vy[0]), p[1] + int(vy[1]))
                x = (p[0] + int(vx[0]), p[1] + int(vx[1]))
                if geom.stratum(y) != "edge" or geom.stratum(x) != "edge":
                    continue
                eta_o = ginv(geom.gauge(y)) @ A @ _REF_EDGE_GAUGE[ct]
                eta_i = ginv(geom.gauge(x)) @ A @ _REF_EDGE_GAUGE[cs]
                Mo = self.t_out.eps.mat(z2_flip(eta_o, SIGMA_EDGE))
                Mi = np.linalg.inv(self.t_in.eps.mat(z2_flip(eta_i, SIGMA_EDGE)))
                out.append((y, x, Mo, (c, uy, ux), Mi))
        return out


# ----------------------------------------------------------------------------
# The stratified layer
# ----------------------------------------------------------------------------

class StratifiedLayer:
    """A full B^rig-equivariant layer on the stratified rectangle."""

    def __init__(self, geom: RectGeometry, t_in: Types, t_out: Types, r0: int):
        assert 2 * r0 + 2 <= min(geom.W, geom.H), "small-filter regime violated"
        self.geom, self.t_in, self.t_out, self.r0 = geom, t_in, t_out, r0
        self.lay_in, self.lay_out = Layout(geom, t_in), Layout(geom, t_out)
        self.blocks = [cls(t_in, t_out, r0) for cls in
                       (BulkBulk, EdgeEdge, CornerCorner, EdgeBulk, BulkEdge,
                        CornerBulk, BulkCorner, CornerEdge, EdgeCorner,
                        EdgeEdgeCorner)]
        self.param_slices, off = [], 0
        for blk in self.blocks:
            n = blk.solve_basis()
            self.param_slices.append(slice(off, off + n))
            off += n
        self.n_params = off
        self._taps = [blk.taps(geom) for blk in self.blocks]

    def assemble(self, theta, xp=np):
        L = xp.zeros((self.lay_out.total, self.lay_in.total),
                     dtype=theta.dtype)
        for blk, sl, taps in zip(self.blocks, self.param_slices, self._taps):
            ks = blk.kernels(theta[sl], xp=xp)
            for (y, x, Mo, key, Mi) in taps:
                M = xp.asarray(Mo, dtype=theta.dtype) @ ks[key] \
                    @ xp.asarray(Mi, dtype=theta.dtype)
                L[self.lay_out.sl(y), self.lay_in.sl(x)] += M
        return L

    def param_report(self):
        return {blk.name: blk.n_params for blk in self.blocks}


# ----------------------------------------------------------------------------
# Rigid local bisections and their transport operators
# ----------------------------------------------------------------------------

def admissible_set(geom: RectGeometry, A: np.ndarray, t: np.ndarray):
    """Pixels x such that (g, x), g = (t, A), is an arrow of Gamma^tc:
    g.x in the grid, same stratum, and matching tangent cones."""
    U = []
    for j in range(geom.H):
        for i in range(geom.W):
            x = (i, j)
            sx = geom.stratum(x)
            v = A @ np.array(x) + t
            y = (int(v[0]), int(v[1]))
            if geom.stratum(y) != sx:
                continue
            if sx == "bulk":
                U.append((x, y))
            elif sx == "edge":
                eta = ginv(geom.gauge(y)) @ A @ geom.gauge(x)
                if geq(eta, R0) or geq(eta, SIGMA_EDGE):
                    U.append((x, y))
            else:
                eta = ginv(geom.gauge(y)) @ A @ geom.gauge(x)
                if geq(eta, R0) or geq(eta, SIGMA_CORNER):
                    U.append((x, y))
    return U


def eroded_pairs(geom: RectGeometry, A, t, r: int):
    """Pairs (x, g.x) of the maximal admissible set whose full r-neighborhood
    (sup-norm, within the grid) is admissible on both the source and target
    side. Equivariance of a *composite* of layers with total receptive radius
    r holds exactly on this eroded domain (information routed through
    intermediate sites must itself be transported by the bisection)."""
    pairs = admissible_set(geom, A, t)
    src = {p[0] for p in pairs}
    dst = {p[1] for p in pairs}

    def nbhd_ok(p, S):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                q = (p[0] + dx, p[1] + dy)
                if geom.stratum(q) is not None and q not in S:
                    return False
        return True

    return [(x, y) for (x, y) in pairs if nbhd_ok(x, src) and nbhd_ok(y, dst)]


def build_transport(geom, types: Types, A, pairs):
    lay = Layout(geom, types)
    Lam = np.zeros((lay.total, lay.total))
    P_U = np.zeros((lay.total, lay.total))
    P_Up = np.zeros((lay.total, lay.total))
    for (x, y) in pairs:
        M = fiber_matrix(types, geom, A, x, y)
        Lam[lay.sl(y), lay.sl(x)] = M
        P_U[lay.sl(x), lay.sl(x)] = np.eye(types.dim(geom.stratum(x)))
        P_Up[lay.sl(y), lay.sl(y)] = np.eye(types.dim(geom.stratum(y)))
    return Lam, P_U, P_Up


def equivariance_residual(geom, t_in: Types, t_out: Types, L: np.ndarray,
                          A: np.ndarray, t: np.ndarray,
                          pairs=None) -> float:
    """Sup-norm residual of  P_{U'} L Lambda_in(b) - Lambda_out(b) P_U L P_U
    for the rigid bisection b = (A, t), on its maximal admissible domain by
    default, or on an explicitly given pair list (e.g. an eroded domain)."""
    if pairs is None:
        pairs = admissible_set(geom, A, t)
    if not pairs:
        return 0.0
    Lam_i, PU_i, _ = build_transport(geom, t_in, A, pairs)
    Lam_o, PU_o, PUp_o = build_transport(geom, t_out, A, pairs)
    lhs = PUp_o @ L @ Lam_i
    rhs = Lam_o @ PU_o @ L @ PU_i
    return float(np.abs(lhs - rhs).max())


# ----------------------------------------------------------------------------
# Predicted parameter counts (Section 7.3 formulas) for cross-checking
# ----------------------------------------------------------------------------

def predicted_counts(t_in: Types, t_out: Types, r0: int):
    ax = [R0, SIGMA_EDGE]
    dg = [R0, SIGMA_CORNER]
    d4 = D4
    h = {}
    h["D4"] = hom_dim([(t_out.rho.mat(g), t_in.rho.mat(g)) for g in d4])
    h["ax"] = hom_dim([(t_out.rho.mat(g), t_in.rho.mat(g)) for g in ax])
    h["dg"] = hom_dim([(t_out.rho.mat(g), t_in.rho.mat(g)) for g in dg])
    h["eps"] = hom_dim([(t_out.eps.mat(f), t_in.eps.mat(f)) for f in (False, True)])
    h["dlt"] = hom_dim([(t_out.delta.mat(f), t_in.delta.mat(f)) for f in (False, True)])
    h["mix_eb"] = hom_dim([(t_out.eps.mat(False), t_in.rho.mat(R0)),
                           (t_out.eps.mat(True), t_in.rho.mat(SIGMA_EDGE))])
    h["mix_be"] = hom_dim([(t_out.rho.mat(R0), t_in.eps.mat(False)),
                           (t_out.rho.mat(SIGMA_EDGE), t_in.eps.mat(True))])
    h["mix_cb"] = hom_dim([(t_out.delta.mat(False), t_in.rho.mat(R0)),
                           (t_out.delta.mat(True), t_in.rho.mat(SIGMA_CORNER))])
    h["mix_bc"] = hom_dim([(t_out.rho.mat(R0), t_in.delta.mat(False)),
                           (t_out.rho.mat(SIGMA_CORNER), t_in.delta.mat(True))])
    dri, dro = t_in.rho.dim, t_out.rho.dim
    dei, deo = t_in.eps.dim, t_out.eps.dim
    ddi, ddo = t_in.delta.dim, t_out.delta.dim
    g = r0 * (r0 - 1) // 2
    return {
        "bulk<-bulk": h["D4"] + r0 * (h["ax"] + h["dg"]) + g * dri * dro,
        "edge<-edge (same edge)": h["eps"] + r0 * dei * deo,
        "corner<-corner": h["dlt"],
        "edge<-bulk": r0 * h["mix_eb"] + r0 * r0 * dri * deo,
        "bulk<-edge": r0 * h["mix_be"] + r0 * r0 * dei * dro,
        "corner<-bulk": r0 * h["mix_cb"] + g * dri * ddo,
        "bulk<-corner": r0 * h["mix_bc"] + g * ddi * dro,
        "corner<-edge": r0 * dei * ddo,
        "edge<-corner": r0 * ddi * deo,
        "edge<-edge (across corner)": r0 * r0 * dei * deo,
    }


# ----------------------------------------------------------------------------
# Optional PyTorch wrapper
# ----------------------------------------------------------------------------

try:
    import torch

    class StratifiedLayerModule(torch.nn.Module):
        """PyTorch module wrapping a StratifiedLayer (dense reference)."""

        def __init__(self, layer: StratifiedLayer):
            super().__init__()
            self.layer = layer
            self.theta = torch.nn.Parameter(
                torch.randn(layer.n_params, dtype=torch.float64)
                / max(1.0, layer.n_params) ** 0.5)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            L = self.layer.assemble(self.theta, xp=torch)
            return features @ L.T   # features: (batch, N_in)

except ImportError:  # torch not installed; numpy API remains fully usable
    pass


# ----------------------------------------------------------------------------
# Tests / demo
# ----------------------------------------------------------------------------

def _selftest_vec_identity(rng):
    A = rng.standard_normal((3, 4)); K = rng.standard_normal((4, 2))
    B = rng.standard_normal((2, 5))
    assert np.allclose(_vec(A @ K @ B), _sandwich(A, B) @ _vec(K))


def main():
    rng = np.random.default_rng(0)
    _selftest_vec_identity(rng)

    W, H, r0 = 11, 9, 2
    geom = RectGeometry(W, H)
    t_in = Types(D4Rep({"A1": 1, "E": 1}), Z2Rep(1, 1), Z2Rep(1, 1))
    t_out = Types(D4Rep({"A1": 1, "B1": 1, "E": 1}), Z2Rep(2, 1), Z2Rep(1, 2))

    layer = StratifiedLayer(geom, t_in, t_out, r0)
    print("Grid %dx%d, r0=%d.  Feature dims: in=%d out=%d, params=%d"
          % (W, H, r0, layer.lay_in.total, layer.lay_out.total, layer.n_params))

    # --- parameter counts vs closed-form predictions ------------------------
    pred = predicted_counts(t_in, t_out, r0)
    print("\n%-30s %8s %8s" % ("block", "solved", "formula"))
    ok = True
    for blk in layer.blocks:
        p = pred[blk.name]
        flag = "" if p == blk.n_params else "  <-- MISMATCH"
        ok &= (p == blk.n_params)
        print("%-30s %8d %8d%s" % (blk.name, blk.n_params, p, flag))
    assert ok, "parameter count mismatch"

    # --- exact partial equivariance -----------------------------------------
    theta = rng.standard_normal(layer.n_params)
    L = layer.assemble(theta)

    tests = []
    for A in D4:                            # all point-group parts
        for _ in range(3):                  # random translations each
            t = rng.integers(-max(W, H), max(W, H), size=2)
            tests.append((A, t))
    tests.append((R0, np.array([3, 0])))    # pure translation
    tests.append((R1, np.array([W - 1, 0])))  # rotation about a corner-ish

    worst = 0.0
    for (A, t) in tests:
        res = equivariance_residual(geom, t_in, t_out, L, A, np.asarray(t))
        worst = max(worst, res)
    print("\nSingle layer: partial-equivariance residual over %d rigid "
          "bisections (maximal domains): %.2e" % (len(tests), worst))
    assert worst < 1e-10, "equivariance violated"

    # --- negative control: unconstrained layer breaks equivariance ----------
    L_bad = rng.standard_normal(L.shape)
    bad = max(equivariance_residual(geom, t_in, t_out, L_bad, A, np.asarray(t))
              for (A, t) in tests[:8])
    print("Residual of an unconstrained random layer (control): %.2e" % bad)
    assert bad > 1e-2

    # --- composition: erosion by the receptive-field radius ------------------
    layer2 = StratifiedLayer(geom, t_out, t_in, r0)
    L2 = layer2.assemble(rng.standard_normal(layer2.n_params))
    L21 = L2 @ L                            # receptive radius 2*r0

    worst_eroded, worst_maximal = 0.0, 0.0
    for (A, t) in tests[:12]:
        t = np.asarray(t)
        er = eroded_pairs(geom, A, t, r0)
        worst_eroded = max(worst_eroded, equivariance_residual(
            geom, t_in, t_in, L21, A, t, pairs=er))
        worst_maximal = max(worst_maximal, equivariance_residual(
            geom, t_in, t_in, L21, A, t))
    print("Two-layer composite: residual on domains eroded by r0: %.2e"
          % worst_eroded)
    print("Two-layer composite: residual on maximal domains (expected to be "
          "nonzero -- partial symmetry erodes with depth): %.2e" % worst_maximal)
    assert worst_eroded < 1e-10

    # --- composition under a *global* symmetry: exact, no erosion ------------
    Wsq = 9
    geo2 = RectGeometry(Wsq, Wsq)
    la = StratifiedLayer(geo2, t_in, t_out, r0)
    lb = StratifiedLayer(geo2, t_out, t_in, r0)
    Lg = lb.assemble(rng.standard_normal(lb.n_params)) \
        @ la.assemble(rng.standard_normal(la.n_params))
    c = (Wsq - 1) / 2.0
    worst_glob = 0.0
    for A in D4:                            # x -> A(x-c)+c maps the square to itself
        t = np.asarray(np.round(np.array([c, c]) - A @ np.array([c, c])),
                       dtype=int)
        worst_glob = max(worst_glob, equivariance_residual(
            geo2, t_in, t_in, Lg, A, t))
    print("Two-layer composite on a square, global D4 about the center: %.2e"
          % worst_glob)
    assert worst_glob < 1e-10

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
