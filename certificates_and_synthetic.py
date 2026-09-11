from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from groupoid_cnn import (
    D4, R0, R1, R2, R3, MV,
    D4Rep, Z2Rep, Types, RectGeometry, Layout,
    StratifiedLayer, BulkBulk, EdgeEdge, CornerCorner, EdgeBulk, BulkEdge,
    CornerBulk, BulkCorner, CornerEdge, EdgeCorner, EdgeEdgeCorner,
    predicted_counts, admissible_set, eroded_pairs, equivariance_residual,
    build_transport,
)
from experiment_core import TorchStratifiedLayer, flat_to_dict, dict_to_flat, poisson_solver

BLOCK_CLASSES = (
    BulkBulk, EdgeEdge, CornerCorner, EdgeBulk, BulkEdge,
    CornerBulk, BulkCorner, CornerEdge, EdgeCorner, EdgeEdgeCorner,
)


def exact_certificates(seed: int = 7):
    rng = np.random.default_rng(seed)

    # Rich feature types: parameter counts and dense/sparse agreement.
    geom_rich = RectGeometry(11, 9)
    r0_rich = 2
    t_in_rich = Types(D4Rep({"A1": 1, "E": 1}), Z2Rep(1, 1), Z2Rep(1, 1))
    t_out_rich = Types(D4Rep({"A1": 1, "B1": 1, "E": 1}), Z2Rep(2, 1), Z2Rep(1, 2))
    layer_rich = StratifiedLayer(geom_rich, t_in_rich, t_out_rich, r0_rich)
    counts_solved = layer_rich.param_report()
    counts_formula = predicted_counts(t_in_rich, t_out_rich, r0_rich)

    theta = rng.standard_normal(layer_rich.n_params)
    dense = layer_rich.assemble(theta)
    torch_layer = TorchStratifiedLayer(geom_rich, t_in_rich, t_out_rich, r0_rich)
    import torch
    with torch.no_grad():
        torch_layer.theta.copy_(torch.as_tensor(theta))
    sparse_dense_worst = 0.0
    for _ in range(8):
        psi = rng.standard_normal(dense.shape[1])
        with torch.no_grad():
            out = dict_to_flat(torch_layer(flat_to_dict(psi, geom_rich, t_in_rich)),
                               geom_rich, t_out_rich)
        sparse_dense_worst = max(sparse_dense_worst,
                                 float(np.max(np.abs(out - dense @ psi))))

    # Exhaustive rigid-motion certificates on a small scalar grid.
    W, H, r0 = 7, 6, 1
    geom = RectGeometry(W, H)
    scalar = Types(D4Rep({"A1": 1}), Z2Rep(1, 0), Z2Rep(1, 0))
    layer1 = StratifiedLayer(geom, scalar, scalar, r0)
    layer2 = StratifiedLayer(geom, scalar, scalar, r0)
    L1 = layer1.assemble(rng.standard_normal(layer1.n_params))
    L2 = layer2.assemble(rng.standard_normal(layer2.n_params))
    L21 = L2 @ L1

    motions = []
    for A in D4:
        for tx in range(-(W - 1), W):
            for ty in range(-(H - 1), H):
                t = np.array([tx, ty], dtype=int)
                pairs = admissible_set(geom, A, t)
                if pairs:
                    motions.append((A, t, pairs))

    worst_single = 0.0
    worst_composite_eroded = 0.0
    worst_composite_maximal = 0.0
    n_eroded_nonempty = 0
    n_maximal_failure = 0
    for A, t, pairs in motions:
        worst_single = max(worst_single,
                           equivariance_residual(geom, scalar, scalar, L1, A, t, pairs))
        er = eroded_pairs(geom, A, t, r0)
        if er:
            n_eroded_nonempty += 1
            worst_composite_eroded = max(
                worst_composite_eroded,
                equivariance_residual(geom, scalar, scalar, L21, A, t, er))
        res_max = equivariance_residual(geom, scalar, scalar, L21, A, t, pairs)
        worst_composite_maximal = max(worst_composite_maximal, res_max)
        if res_max > 1e-8:
            n_maximal_failure += 1

    # Global D4 on a square: no erosion at any depth.
    Wsq = 7
    geosq = RectGeometry(Wsq, Wsq)
    ls1 = StratifiedLayer(geosq, scalar, scalar, r0)
    ls2 = StratifiedLayer(geosq, scalar, scalar, r0)
    Lsq = ls2.assemble(rng.standard_normal(ls2.n_params)) @ \
          ls1.assemble(rng.standard_normal(ls1.n_params))
    c = np.array([(Wsq - 1) / 2, (Wsq - 1) / 2])
    worst_global = 0.0
    for A in D4:
        t = np.asarray(np.round(c - A @ c), dtype=int)
        pairs = admissible_set(geosq, A, t)
        worst_global = max(worst_global,
                           equivariance_residual(geosq, scalar, scalar, Lsq, A, t, pairs))

    return {
        "rich_grid": [geom_rich.W, geom_rich.H],
        "rich_radius": r0_rich,
        "counts_solved": counts_solved,
        "counts_formula": counts_formula,
        "total_parameters": layer_rich.n_params,
        "sparse_dense_max_abs": sparse_dense_worst,
        "exhaustive_grid": [W, H],
        "exhaustive_radius": r0,
        "nonempty_rigid_motions": len(motions),
        "nonempty_eroded_motions": n_eroded_nonempty,
        "single_layer_worst_abs": worst_single,
        "two_layer_eroded_worst_abs": worst_composite_eroded,
        "two_layer_maximal_worst_abs": worst_composite_maximal,
        "two_layer_maximal_failures": n_maximal_failure,
        "global_D4_two_layer_worst_abs": worst_global,
    }


def _basis_matrices_geq(geom, types, r0):
    layer = StratifiedLayer(geom, types, types, r0)
    P = layer.n_params
    mats = np.empty((P, layer.lay_out.total, layer.lay_in.total))
    for j in range(P):
        e = np.zeros(P)
        e[j] = 1.0
        mats[j] = layer.assemble(e)
    names = []
    for blk, sl in zip(layer.blocks, layer.param_slices):
        names.extend([blk.name] * (sl.stop - sl.start))
    return layer, mats, names


def _basis_matrices_unc_scalar(geom, types, r0):
    """Same tap connectivity as the stratified layer, but one free scalar per slot."""
    lay = Layout(geom, types)
    mats = []
    names = []
    for cls in BLOCK_CLASSES:
        blk = cls(types, types, r0)
        taps = blk.taps(geom)
        slot_index = {key: i for i, key in enumerate(blk.slots)}
        for key in blk.slots:
            L = np.zeros((lay.total, lay.total))
            for y, x, Mo, tap_key, Mi in taps:
                if slot_index[tap_key] != slot_index[key]:
                    continue
                # Scalar representations: all matrices are 1x1.
                val = float(np.asarray(Mo).reshape(-1)[0] *
                            np.asarray(Mi).reshape(-1)[0])
                L[lay.sl(y), lay.sl(x)] += val
            mats.append(L)
            names.append(blk.name)
    return np.stack(mats), names


def _fit_probe_model(bases, psi, ell, z):
    X = np.einsum("no,poi,ni->np", ell, bases, psi, optimize=True)
    theta, *_ = np.linalg.lstsq(X, z, rcond=None)
    return theta


def synthetic_recovery(seed: int = 11, n_trials: int = 40):
    rng = np.random.default_rng(seed)
    geom = RectGeometry(9, 8)
    r0 = 2
    scalar = Types(D4Rep({"A1": 1}), Z2Rep(1, 0), Z2Rep(1, 0))
    layer, B_geq, geq_names = _basis_matrices_geq(geom, scalar, r0)
    B_unc, unc_names = _basis_matrices_unc_scalar(geom, scalar, r0)

    keep_nox = np.array([n in {"bulk<-bulk", "edge<-edge (same edge)",
                               "corner<-corner"} for n in geq_names])
    keep_nocorner = np.array(["corner" not in n for n in geq_names])
    models = {
        "GEQ": B_geq,
        "GEQ-noX": B_geq[keep_nox],
        "GEQ-noCorner": B_geq[keep_nocorner],
        "UNC": B_unc,
    }
    sizes = np.array([10, 20, 40, 80, 160, 320])
    errors = {name: np.empty((n_trials, len(sizes))) for name in models}

    Nin = layer.lay_in.total
    Nout = layer.lay_out.total
    for trial in range(n_trials):
        # Normalize a random complete equivariant target.  Draw each block at
        # comparable energy so the cross/corner components cannot vanish by chance.
        theta_star = rng.standard_normal(layer.n_params)
        Lstar = np.tensordot(theta_star, B_geq, axes=(0, 0))
        Lstar /= np.linalg.norm(Lstar, "fro")

        nmax = int(sizes[-1])
        psi_all = rng.standard_normal((nmax, Nin)) / np.sqrt(Nin)
        ell_all = rng.standard_normal((nmax, Nout)) / np.sqrt(Nout)
        z_all = np.einsum("no,oi,ni->n", ell_all, Lstar, psi_all,
                          optimize=True)

        for name, bases in models.items():
            for k, n in enumerate(sizes):
                theta = _fit_probe_model(bases, psi_all[:n], ell_all[:n], z_all[:n])
                Lhat = np.tensordot(theta, bases, axes=(0, 0))
                errors[name][trial, k] = (
                    np.linalg.norm(Lhat - Lstar, "fro") ** 2 /
                    np.linalg.norm(Lstar, "fro") ** 2
                )

    summary = {}
    for name, arr in errors.items():
        summary[name] = {
            "median": np.median(arr, axis=0).tolist(),
            "q25": np.quantile(arr, 0.25, axis=0).tolist(),
            "q75": np.quantile(arr, 0.75, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0, ddof=1).tolist(),
        }

    return {
        "grid": [geom.W, geom.H],
        "radius": r0,
        "trials": n_trials,
        "sizes": sizes.tolist(),
        "parameters": {name: int(b.shape[0]) for name, b in models.items()},
        "errors": summary,
    }


def _poisson_operator_matrix(geom: RectGeometry):
    scalar = Types(D4Rep({"A1": 1}), Z2Rep(1, 0), Z2Rep(1, 0))
    lay = Layout(geom, scalar)
    solve, _ = poisson_solver(geom)
    S = np.zeros((lay.total, lay.total))
    for j in range(lay.total):
        f = np.zeros((geom.H, geom.W))
        g = np.zeros((geom.H, geom.W))
        e = np.zeros(lay.total)
        e[j] = 1.0
        for p in geom.bulk_sites:
            f[p[1], p[0]] = e[lay.sl(p)][0]
        for p in geom.edge_sites + geom.corner_sites:
            g[p[1], p[0]] = e[lay.sl(p)][0]
        u = solve(f, g)
        out = np.zeros(lay.total)
        for p in geom.bulk_sites:
            out[lay.sl(p)] = u[p[1], p[0]]
        S[:, j] = out
    return scalar, S


def _relative_transport_residual(geom, types, L, A, t, pairs):
    if not pairs:
        return None
    Li, PUi, _ = build_transport(geom, types, A, pairs)
    Lo, PUo, PUp_o = build_transport(geom, types, A, pairs)
    lhs = PUp_o @ L @ Li
    rhs = Lo @ PUo @ L @ PUi
    den = max(np.linalg.norm(rhs, "fro"), 1e-15)
    return float(np.linalg.norm(lhs - rhs, "fro") / den)


def poisson_diagnostic():
    geom = RectGeometry(8, 6)
    scalar, S = _poisson_operator_matrix(geom)
    tests = {
        "partial translation (1,0)": (R0, np.array([1, 0])),
        "partial translation (2,1)": (R0, np.array([2, 1])),
        "partial quarter-turn": (R1, np.array([geom.W - 1, 0])),
    }
    residuals = {}
    for name, (A, t) in tests.items():
        pairs = admissible_set(geom, A, t)
        residuals[name] = {
            "n_pairs": len(pairs),
            "relative_frobenius": _relative_transport_residual(
                geom, scalar, S, A, t, pairs),
        }

    # Exact global symmetries of a rectangle: D2 generated by horizontal and
    # vertical reflections (and their product, the half-turn).
    c = np.array([(geom.W - 1) / 2, (geom.H - 1) / 2])
    MH = np.array([[1, 0], [0, -1]])
    global_As = [R0, MV, MH, R2]
    glob = []
    for A in global_As:
        t = np.asarray(np.round(c - A @ c), dtype=int)
        pairs = admissible_set(geom, A, t)
        glob.append(_relative_transport_residual(geom, scalar, S, A, t, pairs))
    return {
        "grid": [geom.W, geom.H],
        "partial": residuals,
        "global_rectangle_D2_worst_relative_frobenius": float(max(glob)),
    }


def make_figures(payload, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    syn = payload["synthetic_recovery"]
    sizes = np.array(syn["sizes"])
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    markers = {"GEQ": "o", "GEQ-noX": "s", "GEQ-noCorner": "^", "UNC": "d"}
    for name in ["GEQ", "GEQ-noX", "GEQ-noCorner", "UNC"]:
        med = np.maximum(np.array(syn["errors"][name]["median"]), 1e-16)
        q25 = np.maximum(np.array(syn["errors"][name]["q25"]), 1e-16)
        q75 = np.maximum(np.array(syn["errors"][name]["q75"]), 1e-16)
        line, = ax.loglog(sizes, med, marker=markers[name], label=f"{name} ({syn['parameters'][name]} par.)")
        ax.fill_between(sizes, q25, q75, alpha=0.16, color=line.get_color())
    ax.set_xlabel("scalar training probes")
    ax.set_ylabel("relative operator error")
    ax.set_title("Recovery of a random complete GEQ layer")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "synthetic_geq_recovery.pdf")
    fig.savefig(outdir / "synthetic_geq_recovery.png", dpi=180)
    plt.close(fig)



def main():
    payload = {
        "exact_certificates": exact_certificates(),
        "synthetic_recovery": synthetic_recovery(),
        "poisson_equivariance_diagnostic": poisson_diagnostic(),
    }
    out = OUTPUT_DIR / "certificates_and_synthetic_results.json"
    out.write_text(json.dumps(payload, indent=2))
    make_figures(payload, OUTPUT_DIR)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
