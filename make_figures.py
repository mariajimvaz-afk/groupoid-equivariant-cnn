"""Figures and LaTeX tables for the definitive Section 15 benchmark."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUT / "benchmark_results.json"
if not RESULTS_FILE.exists():
    RESULTS_FILE = ROOT / "results" / "benchmark_results.json"
R = json.load(open(RESULTS_FILE))
CURVE = ["GEQ", "GEQ-noX", "UNC", "STEER", "STEER-w", "CNN-4", "CNN-6",
         "CNN-12"]
ABL = ["GEQ", "GEQ-noEB", "GEQ-noC", "GEQ-noXC", "GEQ-noX"]
STYLE = {"GEQ": ("o-", "C0"), "GEQ-noX": ("s--", "C0"), "UNC": ("^-", "C1"),
         "STEER": ("v-", "C2"), "STEER-w": ("v--", "C2"),
         "CNN-4": ("d:", "C3"), "CNN-6": ("d-", "C3"),
         "CNN-12": ("x--", "C3"), "GEQ-nl": ("o-", "C4"),
         "CNN-6-nl": ("d-", "C5")}
TITLES = {"five": "five-point (corner-insensitive)",
          "nine": "nine-point (corner-sensitive)"}


def curves():
    sizes = np.array(R["sizes"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, task in zip(axes, ["five", "nine"]):
        for name in CURVE:
            res = R["results"][task].get(name, {})
            if not all(str(n) in res for n in sizes):
                continue
            med = [res[str(n)]["median"] for n in sizes]
            q25 = [res[str(n)]["q25"] for n in sizes]
            q75 = [res[str(n)]["q75"] for n in sizes]
            fmt, c = STYLE[name]
            ax.loglog(sizes, med, fmt, color=c, ms=4,
                      label=f"{name} ({R['params'][name]})")
            ax.fill_between(sizes, q25, q75, color=c, alpha=0.13)
        ax.set_xlabel("training fields")
        ax.set_title(TITLES[task], fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("relative test MSE")
    axes[0].legend(fontsize=6.8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "poisson_definitive_curves.pdf")
    fig.savefig(OUT / "poisson_definitive_curves.png", dpi=170)


def nonlin_fig():
    sizes = np.array(R["sizes"])
    fig, ax = plt.subplots(figsize=(4.9, 3.6))
    for name in ["GEQ", "GEQ-nl", "CNN-6", "CNN-6-nl"]:
        res = R["results"]["nine"].get(name, {})
        if not all(str(n) in res for n in sizes):
            continue
        med = [res[str(n)]["median"] for n in sizes]
        q25 = [res[str(n)]["q25"] for n in sizes]
        q75 = [res[str(n)]["q75"] for n in sizes]
        fmt, c = STYLE[name]
        ax.loglog(sizes, med, fmt, color=c, ms=4,
                  label=f"{name} ({R['params'][name]})")
        ax.fill_between(sizes, q25, q75, color=c, alpha=0.13)
    ax.set_xlabel("training fields")
    ax.set_ylabel("relative test MSE")
    ax.set_title("nonlinear round, nine-point task", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "poisson_nonlinear_round.pdf")
    fig.savefig(OUT / "poisson_nonlinear_round.png", dpi=170)


def fmt_ms(d):
    return f"${d['mean']:.4f} \\pm {d['std']:.4f}$"


def table_curves():
    lines = []
    for task in ["five", "nine"]:
        lines.append(f"% ---- {task}-point task ----")
        for name in CURVE:
            res = R["results"][task].get(name, {})
            if "400" not in res:
                continue
            row = [name, str(R["params"][name])]
            for n in R["sizes"]:
                row.append(fmt_ms(res[str(n)]))
            lines.append(" & ".join(row) + r" \\")
    return "\n".join(lines)


def table_ablation():
    lines = []
    for task in ["five", "nine"]:
        lines.append(f"% ---- {task}-point, n = 400 ----")
        for name in ABL:
            res = R["results"][task].get(name, {})
            if "400" not in res:
                continue
            d = res["400"]
            lines.append(
                f"{name} & {R['params'][name]} & {fmt_ms(d)} & "
                f"${d['median']:.4f}$ \\\\")
    return "\n".join(lines)


def table_certs():
    lines = []
    for task in ["five", "nine"]:
        certs = R["certs"].get(task, {})
        lines.append(f"% ---- trained-operator transport residuals, {task} ----")
        for name, c in certs.items():
            row = [name]
            for k in ["translation (1,0)", "translation (2,1)",
                      "quarter-turn about corner"]:
                row.append(f"{c[k]['residual']:.1e} ({c[k]['n_pairs']})")
            row.append(f"{c['global D2 (rectangle)']:.1e}")
            lines.append(" & ".join(row) + r" \\")
    return "\n".join(lines)


if __name__ == "__main__":
    curves()
    nonlin_fig()
    with open(OUT / "benchmark_tables.tex", "w") as fh:
        fh.write("% Sample-efficiency table rows\n" + table_curves()
                 + "\n\n% Ablation table rows\n" + table_ablation()
                 + "\n\n% Certificate table rows\n" + table_certs() + "\n")
    print("figures + tables written to", OUT)
