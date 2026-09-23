# Groupoid-equivariant neural networks on bounded domains — reproducibility code

Code accompanying the manuscript **“Theory for groupoid equivariant neural networks: an approach for steerable CNNs on bounded domains”**, by **A. Ibort, M. Jiménez-Vázquez, and J.M. Pérez-Pardo**.

This repository contains the reference finite-grid implementation, numerical certificates, synthetic recovery experiment, and the final Poisson–Dirichlet benchmark reported in the manuscript. All file paths are repository-relative; no cloud-runtime or machine-specific paths are required.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── environment-tested.txt
├── groupoid_cnn.py
├── experiment_core.py
├── certificates_and_synthetic.py
├── benchmark.py
├── make_figures.py
├── results/
│   ├── certificates_and_synthetic_results.json
│   ├── benchmark_results.json
│   └── nl_erosion_sweep.json
└── outputs/
    └── .gitkeep
```

### What each file does

- `groupoid_cnn.py` — reference discrete groupoid-steerable construction and fast algebraic/equivariance sanity checks.
- `experiment_core.py` — reusable PyTorch layers, baselines, data utilities, and transport-residual helpers. This is a library module, not a standalone experiment.
- `certificates_and_synthetic.py` — experiments corresponding to the exact certificates, synthetic recovery study, and Poisson symmetry diagnostic.
- `benchmark.py` — final Section 15 benchmark, including learning-rate selection, the complete training grid, ablations, nonlinear round, and trained-operator certificates.
- `make_figures.py` — regenerates the final Poisson benchmark figures and LaTeX table rows from the supplied benchmark results.
- `results/` — reference numerical results supplied with the code, including the final nonlinear erosion-radius sweep.
- `outputs/` — local generated results, figures, tables, checkpoints, and trained states. Generated contents are not version-controlled.

## Installation

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A GPU is not required. CPU execution is supported. Package versions used to verify this prepared repository are recorded in `environment-tested.txt`.

## Reproduction map

### 1. Fast implementation checks

```bash
python groupoid_cnn.py
```

A successful run ends with:

```text
All tests passed.
```

This is the recommended first command after installation.

### 2. Exact certificates, synthetic recovery, and Poisson symmetry diagnostic

```bash
python certificates_and_synthetic.py
```

Outputs are written under `outputs/`, including:

- `certificates_and_synthetic_results.json`
- `synthetic_geq_recovery.pdf`
- `synthetic_geq_recovery.png`

This script covers the numerical checks and synthetic identification experiments preceding the final Poisson benchmark, together with the diagnostic showing the distinction between global rectangle symmetries and proper local bisections for the Poisson–Dirichlet inverse.

### 3. Regenerate final manuscript benchmark figures and tables without retraining

```bash
python make_figures.py
```

The script first looks for a newly generated `outputs/benchmark_results.json`. If none exists, it uses the supplied reference file `results/benchmark_results.json`.

It writes:

- `outputs/poisson_definitive_curves.pdf`
- `outputs/poisson_definitive_curves.png`
- `outputs/poisson_nonlinear_round.pdf`
- `outputs/poisson_nonlinear_round.png`
- `outputs/benchmark_tables.tex`

### 4. Full final Poisson–Dirichlet benchmark

Run the complete resumable pipeline with:

```bash
python benchmark.py
```

The final benchmark uses the manuscript configuration: a `16 × 12` grid, depth `4`, propagation radius `2`, training-set sizes `25`, `100`, and `400`, and `10` independent seeds.

Check model parameter counts:

```bash
python benchmark.py params
```

Select learning rates on validation data:

```bash
python benchmark.py lrsel
```

Run the full training grid:

```bash
python benchmark.py grid
```

Compute trained-operator transport certificates:

```bash
python benchmark.py certs
```

Compute the nonlinear erosion-radius diagnostic (radii 4, 5, and 6):

```bash
python benchmark.py nlsweep
```

Aggregate completed runs:

```bash
python benchmark.py finalize
```

Then regenerate figures and tables:

```bash
python make_figures.py
```

The benchmark is checkpointed under `outputs/checkpoints/`, so an interrupted run can be resumed by invoking the same stage again. The default `python benchmark.py` command runs all stages and skips completed checkpoint entries.

An optional wall-clock budget in seconds can be supplied to the learning-rate-selection and grid stages, for example:

```bash
python benchmark.py grid 3600
```

## Reference results versus newly generated results

The files in `results/` are the supplied reference numerical outputs. New runs never overwrite them: newly generated files are written to `outputs/`.

This separation lets reviewers immediately regenerate figures from the reported results while also retaining the option to rerun the experiments from scratch.

## Reproducibility notes

- Random seeds are set explicitly in the experiment scripts.
- Scientific hyperparameters and model definitions are kept consistent with the manuscript protocol.
- Repository preparation only changes portability/packaging and separates reusable code from obsolete pilot workflows.
- Floating-point training results may vary slightly across PyTorch/SciPy versions and hardware.
- Exact algebraic and transport certificates should be at numerical floating-point precision.

## Citation

### 📄 Preprint

**A. Ibort, M. Jimenez-Vazquez, and J. M. Perez-Pardo.**
*Theory for groupoid equivariant neural networks: an approach for steerable CNNs on bounded domains.*
arXiv:2609.25987 [cs.LG], 2026.
https://arxiv.org/abs/2609.25987

