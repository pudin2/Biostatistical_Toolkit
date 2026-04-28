"""
benchmark_cosine_approx.py
==========================
Benchmark de las aproximaciones del kernel coseno definidas en
``kernels.cosine_approx``.

Reporta tres bloques:

1. **Error de aproximacion** sobre ``u in [-1, 1]`` con grilla densa:
   max-error y RMS de cada aproximacion contra ``cos(pi*u/2)`` exacto.

2. **Tiempo GPU/CPU** de evaluar el kernel coseno completo K(u) sobre el
   dataset OTU positivo (N=105,420) en gridsizes 10k / 50k / 100k.
   Se reporta CPU si CuPy no esta disponible (ambas rutas usan el mismo
   ``KDEEvaluator``).

3. **Impacto en KDE**: KS, masa positiva y JS entre la PDF coseno-exacto
   y la PDF coseno-aproximado, usando el mismo flujo que
   ``build_kernel_report_assets.py`` (Scott h, train/test split 80/20,
   muestra test 3000).

Salidas (en ``Kernel_Tests/report_assets/``):
    - cosine_approx_benchmark.csv
    - cosine_approx_error.png
    - cosine_approx_pareto.png
    - cosine_approx_summary.json

Ejecucion:
    python Kernel_Tests/benchmark_cosine_approx.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Garantiza que el modulo kernels/ se encuentre cuando ejecutamos como script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernels.bandwidth import scott_h
from kernels.cosine_approx import APPROXIMATIONS, _PI_OVER_2  # noqa: E402
from kernels.data import load_otu_positives
from kernels.kde import KDEEvaluator, gpu_available
from kernels.stats import (
    cdf_from_pdf,
    jensen_shannon,
    ks_distance,
    normalize_conditional,
    positive_mass,
)

OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Bloque 1: error de aproximacion
# ---------------------------------------------------------------------
def measure_approx_error(n_samples: int = 200_001) -> pd.DataFrame:
    """Calcula max-error y RMS contra cos(pi*u/2) exacto en [-1, 1]."""
    u = np.linspace(-1.0, 1.0, n_samples)
    exact = np.cos(_PI_OVER_2 * u)

    rows = []
    for name, spec in APPROXIMATIONS.items():
        if name == "exact":
            continue
        approx = spec.approx_fn(u, np)
        err = approx - exact
        rows.append({
            "approx":      name,
            "family":      spec.family,
            "degree":      spec.degree,
            "max_error":   float(np.max(np.abs(err))),
            "rms_error":   float(np.sqrt(np.mean(err ** 2))),
            "ops":         spec.ops_per_eval,
            "notes":       spec.notes,
        })
    df = pd.DataFrame(rows).sort_values("max_error").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Bloque 2: tiempo de evaluacion KDE sobre el dataset
# ---------------------------------------------------------------------
class _CosineKernelInjection:
    """Patch local de ``kernels.core.kernel_eval`` para sustituir 'cosine'
    por una funcion arbitraria. Solo afecta a este proceso y se restaura
    en el __exit__.

    Esto permite reutilizar ``KDEEvaluator`` sin modificarlo.
    """

    def __init__(self, kernel_fn: Callable):
        self.kernel_fn = kernel_fn
        self._original = None

    def __enter__(self):
        from kernels import core as _core
        self._original = _core.kernel_eval

        def patched(u, name, xp=np, _orig=self._original, _new=self.kernel_fn):
            if name == "cosine":
                return _new(u, xp=xp)
            return _orig(u, name, xp=xp)

        _core.kernel_eval = patched
        return self

    def __exit__(self, *exc):
        from kernels import core as _core
        _core.kernel_eval = self._original


def measure_kde_timing(values: np.ndarray, gridsizes: tuple[int, ...]) -> pd.DataFrame:
    """Evalua KDE coseno con cada aproximacion y reporta tiempos por gridsize."""
    bw = scott_h(values)
    backend = "gpu" if gpu_available() else "cpu"
    print(f"[timing] backend = {backend}, h_scott = {bw:.4g}")

    rows = []
    for name, spec in APPROXIMATIONS.items():
        for g in gridsizes:
            x_grid = np.logspace(
                np.log10(values.min()),
                np.log10(values.max() + 10 * bw),
                g,
            )
            with _CosineKernelInjection(spec.kernel_fn):
                kde = KDEEvaluator(values, kernel="cosine", bandwidth=bw, backend=backend)
                # warmup (descarta primer lote, especialmente en GPU).
                _ = kde.evaluate(x_grid[: min(1024, g)])
                t0 = time.perf_counter()
                pdf = kde.evaluate(x_grid)
                elapsed = time.perf_counter() - t0
            rows.append({
                "approx":   name,
                "gridsize": g,
                "time_s":   round(elapsed, 4),
                "pts_per_s": round(g / elapsed, 0),
                "backend":  backend,
            })
            print(f"  {name:>10}  gridsize={g:>6}  {elapsed:7.3f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Bloque 3: impacto en KDE estadistico
# ---------------------------------------------------------------------
def measure_kde_impact(values: np.ndarray, gridsize: int = 3000) -> pd.DataFrame:
    """Compara metricas estadisticas (KS, masa, JS) entre coseno exacto y
    cada aproximacion. Emula el split y x_grid de
    ``compute_cv_all_kernels``."""
    train_values, test_values = train_test_split(values, test_size=0.2, random_state=42)
    test_sample = test_values[:3000]

    bw = scott_h(values)
    x_grid = np.logspace(
        np.log10(values.min()),
        np.log10(values.max() + 10 * bw),
        gridsize,
    )
    backend = "gpu" if gpu_available() else "cpu"

    # Referencia: coseno exacto
    with _CosineKernelInjection(APPROXIMATIONS["exact"].kernel_fn):
        pdf_ref = KDEEvaluator(train_values, "cosine", bw, backend=backend).evaluate(x_grid)
    pdf_ref_norm, mass_ref = normalize_conditional(pdf_ref, x_grid)
    ks_ref = ks_distance(test_sample, pdf_ref_norm, x_grid)

    rows = []
    for name, spec in APPROXIMATIONS.items():
        with _CosineKernelInjection(spec.kernel_fn):
            pdf = KDEEvaluator(train_values, "cosine", bw, backend=backend).evaluate(x_grid)
        pdf_norm, mass = normalize_conditional(pdf, x_grid)
        ks = ks_distance(test_sample, pdf_norm, x_grid)
        js = jensen_shannon(pdf_ref_norm, pdf_norm, x_grid)
        rows.append({
            "approx":         name,
            "positive_mass":  round(mass, 6),
            "ks_stat":        round(ks, 6),
            "delta_ks_vs_exact":  round(ks - ks_ref, 6),
            "js_vs_exact":    round(js, 8),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_error_curves(out_path: Path) -> None:
    u = np.linspace(-1.0, 1.0, 4001)
    exact = np.cos(_PI_OVER_2 * u)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name, spec in APPROXIMATIONS.items():
        if name == "exact":
            continue
        err = spec.approx_fn(u, np) - exact
        ax.plot(u, err, label=f"{name} (deg {spec.degree})", linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("u")
    ax.set_ylabel("approx(u) - cos(pi u / 2)")
    ax.set_title("Error de aproximacion sobre [-1, 1]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(error_df: pd.DataFrame, time_df: pd.DataFrame, out_path: Path,
                gridsize_for_time: int) -> None:
    """Pareto error vs tiempo para gridsize_for_time."""
    sub = time_df[time_df["gridsize"] == gridsize_for_time].set_index("approx")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for _, row in error_df.iterrows():
        approx = row["approx"]
        if approx not in sub.index:
            continue
        t = float(sub.loc[approx, "time_s"])
        e = float(row["max_error"])
        ax.scatter(t, e, s=70)
        ax.annotate(approx, (t, e), xytext=(5, 4), textcoords="offset points", fontsize=9)
    # Punto de referencia: coseno exacto
    if "exact" in sub.index:
        t_ex = float(sub.loc["exact", "time_s"])
        ax.axvline(t_ex, color="gray", linestyle="--", linewidth=1.0, label=f"exact ({t_ex:.3f}s)")
    ax.set_yscale("log")
    ax.set_xlabel(f"Tiempo evaluacion KDE (s) @ gridsize={gridsize_for_time:,}")
    ax.set_ylabel("max error |approx - cos|  (log)")
    ax.set_title("Trade-off precision vs velocidad")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    print("Cargando datos OTU...")
    values = load_otu_positives(verbose=True)

    print("\n[1/3] Error de aproximacion ...")
    err_df = measure_approx_error()
    print(err_df.to_string(index=False))

    gridsizes = (10_000, 50_000, 100_000)
    print(f"\n[2/3] Tiempos KDE en gridsizes {gridsizes} ...")
    time_df = measure_kde_timing(values, gridsizes)

    print("\n[3/3] Impacto estadistico (KS, masa, JS) ...")
    impact_df = measure_kde_impact(values)
    print(impact_df.to_string(index=False))

    # Combinar en un CSV unico
    pivot_time = time_df.pivot(index="approx", columns="gridsize", values="time_s")
    pivot_time.columns = [f"time_s_{c}" for c in pivot_time.columns]
    combined = (err_df.set_index("approx")
                .join(pivot_time, how="outer")
                .join(impact_df.set_index("approx"), how="outer")
                .reset_index())
    combined.to_csv(OUT_DIR / "cosine_approx_benchmark.csv", index=False)

    plot_error_curves(OUT_DIR / "cosine_approx_error.png")
    plot_pareto(err_df, time_df, OUT_DIR / "cosine_approx_pareto.png", gridsize_for_time=100_000)

    summary = {
        "backend": "gpu" if gpu_available() else "cpu",
        "n_data_points": int(values.size),
        "bandwidth_scott": float(scott_h(values)),
        "gridsizes": list(gridsizes),
        "approximations": {k: {
            "family": v.family,
            "degree": v.degree,
            "coefficients": list(v.coefficients),
            "ops_per_eval": v.ops_per_eval,
            "notes": v.notes,
        } for k, v in APPROXIMATIONS.items()},
    }
    (OUT_DIR / "cosine_approx_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nArchivos generados:")
    for f in sorted(OUT_DIR.glob("cosine_approx_*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
