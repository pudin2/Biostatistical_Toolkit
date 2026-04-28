"""
generate_all_figures.py
=======================
Orquestador unico que regenera las 4 figuras de referencia consumidas
por las presentaciones LaTeX (`Slides_Cosine_Approx.tex`,
`Slides_Kernels_KDE_3.tex`, `Informe_Kernels_Integrado.tex`).

Las figuras se guardan en dos sitios:
  - Kernel_Tests/img/             figuras committed (estables, bajo DPI controlado)
  - Kernel_Tests/out/figures/     copias regenerables a 200 DPI (gitignored)

Backend GPU automatico: usa CuPy si esta disponible, sino NumPy puro.
Para velocidad, los KDE se calculan con bandwidth Scott escalado por el
factor AMISE-equivalente de cada kernel (mismo criterio que
`build_kernel_report_assets.py`). El bandwidth optimo por CV vive en
`compute_cv_all_kernels.py`.

Ejecutar:
    python Kernel_Tests/scripts/generate_all_figures.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Permite ejecutar desde cualquier cwd anadiendo Kernel_Tests/ al sys.path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernels import COLOR_MAP, KERNELS
from kernels.bandwidth import scott_h
from kernels.cosine_approx import APPROXIMATIONS, _PI_OVER_2
from kernels.data import load_otu_positives
from kernels.kde import KDEEvaluator, gpu_available
from kernels.metadata import H_EQ_FACTORS
from kernels.stats import jensen_shannon, ks_distance, normalize_conditional, positive_mass


IMG_DIR = ROOT / "img"
OUT_DIR = ROOT / "out" / "figures"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    fig.savefig(IMG_DIR / name, dpi=140, bbox_inches="tight")
    fig.savefig(OUT_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_kde_all_kernels(values: np.ndarray, backend: str) -> dict:
    """Los 6 KDE en el mismo eje, bandwidth AMISE-equivalente por kernel."""
    bw_scott = scott_h(values)
    x_grid = np.logspace(
        np.log10(values.min() * 1e-3),
        np.log10(values.max() + 8 * bw_scott * max(H_EQ_FACTORS.values())),
        2000,
    )

    results: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(10, 5))

    pdf_gauss = None
    for kernel in KERNELS:
        h = bw_scott * H_EQ_FACTORS[kernel]
        kde = KDEEvaluator(values, kernel=kernel, bandwidth=h, backend=backend)
        pdf = kde.evaluate(x_grid)
        cond, mass = normalize_conditional(pdf, x_grid)
        if kernel == "gaussian":
            pdf_gauss = cond
            ks_vs_gauss = 0.0
        else:
            assert pdf_gauss is not None
            ks_vs_gauss = float(np.max(np.abs(np.cumsum(cond) - np.cumsum(pdf_gauss)) /
                                       max(1.0, np.cumsum(pdf_gauss).max())))
        results[kernel] = {"h": h, "mass": mass, "ks_vs_gauss": ks_vs_gauss}
        ax.plot(x_grid, cond, color=COLOR_MAP[kernel], linewidth=1.8,
                label=f"{kernel} (h={h:.1f})")

    ax.set_xscale("log")
    ax.set_xlabel("Valor (escala log)")
    ax.set_ylabel("Densidad condicional en x > 0")
    ax.set_title("KDE de los 6 kernels sobre los OTU positivos\n"
                 f"(bandwidth AMISE-equivalente vs Scott; backend={backend.upper()})")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    _save(fig, "kde_all_kernels.png")
    return results


def figure_kde_gridsize_convergence(values: np.ndarray, backend: str) -> None:
    """Convergencia del KDE gaussian con tamano del grid (100..5000)."""
    bw_scott = scott_h(values)
    gridsizes = [100, 500, 1000, 2000, 5000]
    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(gridsizes):
        x = np.logspace(np.log10(values.min()), np.log10(values.max() + 10 * bw_scott), m)
        kde = KDEEvaluator(values, kernel="gaussian", bandwidth=bw_scott, backend=backend)
        pdf = kde.evaluate(x)
        cond, _ = normalize_conditional(pdf, x)
        ax.plot(x, cond, color=cmap(i / max(1, len(gridsizes) - 1)),
                linewidth=1.6, label=f"M={m}")

    ax.set_xscale("log")
    ax.set_xlabel("Valor (escala log)")
    ax.set_ylabel("Densidad condicional")
    ax.set_title("Convergencia del KDE Gaussian con el tamano del grid\n"
                 f"(Scott h={bw_scott:.2f}; backend={backend.upper()})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "kde_gridsize_convergence.png")


def figure_cosine_approx_error() -> dict:
    """Curvas de error de las aproximaciones del coseno sobre [-1, 1]."""
    u = np.linspace(-1.0, 1.0, 200_001)
    cos_exact = np.cos(_PI_OVER_2 * u)

    fig, ax = plt.subplots(figsize=(10, 5))
    errors: dict[str, dict] = {}

    palette = {
        "taylor4":  "#e45756",
        "taylor6":  "#b279a2",
        "cheb4":    "#4c78a8",
        "cheb6":    "#54a24b",
        "remez4":   "#f58518",
        "bhaskara": "#9d755d",
    }

    for name in ("taylor4", "taylor6", "cheb4", "cheb6", "remez4", "bhaskara"):
        ap = APPROXIMATIONS[name]
        approx = ap.approx_fn(u, np)
        err = np.abs(approx - cos_exact)
        errors[name] = {"max": float(err.max()),
                        "rms": float(np.sqrt(np.mean(err ** 2)))}
        ax.semilogy(u, np.maximum(err, 1e-18),
                    color=palette[name], linewidth=1.4,
                    label=f"{name} (max={err.max():.2e})")

    ax.set_xlabel("u")
    ax.set_ylabel("|aprox(u) - cos(πu/2)|  (log)")
    ax.set_title("Error de las aproximaciones polinomicas/racionales del coseno\n"
                 "(sobre 200 001 puntos uniformes en [-1, 1])")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, "cosine_approx_error.png")
    return errors


def figure_cosine_approx_pareto(errors: dict) -> dict:
    """Pareto: error_max vs ns/eval. Tiempos medidos sobre el mismo array."""
    u = np.linspace(-1.0, 1.0, 200_001)

    timings: dict[str, float] = {}
    for name in ("exact", "taylor4", "taylor6", "cheb4", "cheb6", "remez4", "bhaskara"):
        ap = APPROXIMATIONS[name]
        # warmup
        _ = ap.approx_fn(u, np)
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            _ = ap.approx_fn(u, np)
        elapsed = (time.perf_counter() - t0) / reps
        timings[name] = elapsed / u.size * 1e9   # ns/eval

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {
        "exact":    "#000000",
        "taylor4":  "#e45756",
        "taylor6":  "#b279a2",
        "cheb4":    "#4c78a8",
        "cheb6":    "#54a24b",
        "remez4":   "#f58518",
        "bhaskara": "#9d755d",
    }
    for name, ns in timings.items():
        if name == "exact":
            err_val = 1e-16
        else:
            err_val = max(errors[name]["max"], 1e-18)
        ax.scatter(ns, err_val, s=120, color=palette[name],
                   edgecolors="black", linewidths=0.6, label=name)
        ax.annotate(name, (ns, err_val), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("ns por evaluacion (CPU)")
    ax.set_ylabel("Error maximo (log)")
    ax.set_title("Pareto coste vs precision — aproximaciones del coseno")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "cosine_approx_pareto.png")
    return timings


def main() -> None:
    print("[orquestador] cargando OTU positives...", flush=True)
    values = load_otu_positives(verbose=True)

    backend = "gpu" if gpu_available() else "cpu"
    tag = "[GPU]" if backend == "gpu" else "[CPU]"
    print(f"\n{tag} backend={backend}\n")

    t_start = time.perf_counter()

    print(f"{tag} figura 1/4 — KDE de los 6 kernels...", flush=True)
    summary_kernels = figure_kde_all_kernels(values, backend)

    print(f"{tag} figura 2/4 — convergencia del gridsize (gaussian)...", flush=True)
    figure_kde_gridsize_convergence(values, backend)

    print(f"{tag} figura 3/4 — error de aproximaciones del coseno...", flush=True)
    cosine_errors = figure_cosine_approx_error()

    print(f"{tag} figura 4/4 — Pareto coste/precision...", flush=True)
    cosine_timings = figure_cosine_approx_pareto(cosine_errors)

    elapsed = time.perf_counter() - t_start

    print()
    print("=" * 72)
    print(f"Resumen ({tag} backend={backend}, total={elapsed:.1f}s)")
    print("=" * 72)
    print(f"{'kernel':<14} {'h_AMISE':>10} {'masa(x>0)':>12} {'KS_vs_gauss':>12}")
    for k in KERNELS:
        d = summary_kernels[k]
        print(f"  {k:<12} {d['h']:>10.2f} {d['mass']:>12.4f} {d['ks_vs_gauss']:>12.4f}")
    print()
    print(f"{'aprox':<10} {'max_err':>12} {'rms_err':>12} {'ns/eval':>10}")
    for name in ("taylor4", "taylor6", "cheb4", "cheb6", "remez4", "bhaskara"):
        e = cosine_errors[name]
        ns = cosine_timings.get(name, float("nan"))
        print(f"  {name:<8} {e['max']:>12.3e} {e['rms']:>12.3e} {ns:>10.2f}")
    print()
    print(f"Figuras escritas en:")
    print(f"  {IMG_DIR}    (committed, dpi=140)")
    print(f"  {OUT_DIR}    (regenerable, dpi=200)")


if __name__ == "__main__":
    main()
