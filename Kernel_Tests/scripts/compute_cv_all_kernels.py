"""
compute_cv_all_kernels.py
=========================
Calcula el bandwidth óptimo por cross-validation para cada uno de los
6 kernels disponibles en sklearn, usando los mismos datos positivos OTU.

Además obtiene KS, CvM y masa x>0 para cada kernel con su propio h_CV,
permitiendo la tabla compendio completa.

Ejecutar:
    python Kernel_Tests/scripts/compute_cv_all_kernels.py

Salida en Kernel_Tests/report_assets/:
    kernel_cv_all.csv          — h_cv, KS, CvM, masa bajo h_cv propio
    kernel_cv_curves.png       — curvas de log-verosimilitud CV por kernel
    kernel_final_compendium.json — todos los datos para la tabla final
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import cramervonmises, kstest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernels import KERNELS, COLOR_MAP
from kernels.bandwidth import cv_loglik, scott_h, silverman_h
from kernels.data import load_otu_positives
from kernels.metadata import AMISE_PROPS, GPU_TIMING, GRIDSIZE_CONVERGENCE

OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(exist_ok=True)


def evaluate_kernel(
    train_values: np.ndarray,
    test_sample:  np.ndarray,
    x_grid:       np.ndarray,
    bw:           float,
    kernel:       str,
) -> dict:
    kde = KernelDensity(kernel=kernel, bandwidth=bw)
    t0 = time.perf_counter()
    kde.fit(train_values.reshape(-1, 1))
    density = np.exp(kde.score_samples(x_grid.reshape(-1, 1)))
    cpu_time = time.perf_counter() - t0

    dx = np.diff(x_grid, prepend=x_grid[0])
    positive_mass = float(np.trapezoid(density, x_grid))
    cond_density  = density / positive_mass
    mode_x        = float(x_grid[np.argmax(cond_density)])

    cdf = np.cumsum(cond_density * dx)
    cdf[-1] = 1.0
    cdf_fn = interp1d(x_grid, cdf, kind="linear",
                      bounds_error=False, fill_value=(0.0, 1.0),
                      assume_sorted=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks  = kstest(test_sample, lambda z, f=cdf_fn: np.asarray(f(z)), method="asymp")
        cvm = cramervonmises(test_sample, lambda z, f=cdf_fn: np.asarray(f(z)))

    return {
        "positive_mass": positive_mass,
        "mode_x":        mode_x,
        "ks_stat":       float(ks.statistic),
        "cvm_stat":      float(cvm.statistic),
        "cpu_time_s":    round(cpu_time, 3),
    }


def main() -> None:
    print("Cargando datos...")
    values = load_otu_positives()
    train_values, test_values = train_test_split(values, test_size=0.2, random_state=42)
    test_sample = test_values[:3_000]

    bw_scott     = scott_h(values)
    bw_silverman = silverman_h(values)

    x_grid = np.logspace(
        np.log10(values.min()),
        np.log10(values.max() + 10 * bw_scott),
        3_000,
    )

    print(f"  N={len(values):,}  bw_scott={bw_scott:.4g}  bw_silverman={bw_silverman:.4g}")
    print()

    # ── CV por kernel ──────────────────────────────────────────────
    cv_results: dict[str, dict] = {}
    fig_cv, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig_cv.suptitle(
        "Curvas de log-verosimilitud CV por kernel\n"
        f"(datos OTU positivos, N={len(values):,}, submuestra CV = 10,000)",
        fontsize=13,
    )

    for ax, kernel in zip(axes.ravel(), KERNELS):
        print(f"  CV {kernel}...", flush=True)
        t0 = time.perf_counter()
        h_cv, bw_grid, scores = cv_loglik(values, kernel)
        t_cv = time.perf_counter() - t0
        print(f"    h_cv = {h_cv:.4g}  ({t_cv:.1f}s)")

        ax.plot(bw_grid, scores, color=COLOR_MAP[kernel], linewidth=1.8)
        ax.axvline(h_cv, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"CV óptimo h={h_cv:.3g}")
        ax.axvline(bw_scott, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Scott h={bw_scott:.1f}")
        ax.set_xscale("log")
        ax.set_title(f"{kernel}  (h_cv={h_cv:.3g}, t={t_cv:.0f}s)")
        ax.set_xlabel("Bandwidth")
        ax.set_ylabel("Log-verosimilitud CV")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        cv_results[kernel] = {"h_cv": h_cv, "t_cv_s": round(t_cv, 1)}

    fig_cv.tight_layout()
    fig_cv.savefig(OUT_DIR / "kernel_cv_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig_cv)
    print("\nFigura CV guardada.")

    # ── Métricas bajo h_cv propio ──────────────────────────────────
    print("\nEvaluando métricas bajo h_cv propio...")
    cv_metric_rows = []
    for kernel in KERNELS:
        h_cv = cv_results[kernel]["h_cv"]
        metrics = evaluate_kernel(train_values, test_sample, x_grid, h_cv, kernel)
        row = {
            "kernel":        kernel,
            "h_cv":          round(h_cv, 4),
            "h_scott":       round(bw_scott, 4),
            "ratio_cv_scott": round(h_cv / bw_scott, 4),
            "mass_cv":       round(metrics["positive_mass"], 4),
            "mode_cv":       round(metrics["mode_x"], 3),
            "ks_cv":         round(metrics["ks_stat"], 4),
            "cvm_cv":        round(metrics["cvm_stat"], 2),
            "cpu_time_s":    metrics["cpu_time_s"],
        }
        cv_metric_rows.append(row)
        print(f"  {kernel}: h_cv={h_cv:.3g}  mass={row['mass_cv']}  "
              f"KS={row['ks_cv']}  CvM={row['cvm_cv']}")

    cv_df = pd.DataFrame(cv_metric_rows).sort_values("ks_cv").reset_index(drop=True)
    cv_df.to_csv(OUT_DIR / "kernel_cv_all.csv", index=False)

    # ── Métricas bajo Scott común ──────────────────────────────────
    print("\nEvaluando métricas bajo Scott h=88.57...")
    scott_rows = []
    for kernel in KERNELS:
        metrics = evaluate_kernel(train_values, test_sample, x_grid, bw_scott, kernel)
        scott_rows.append({
            "kernel":  kernel,
            "h_scott": round(bw_scott, 4),
            "mass_scott": round(metrics["positive_mass"], 4),
            "mode_scott": round(metrics["mode_x"], 3),
            "ks_scott":   round(metrics["ks_stat"], 4),
            "cvm_scott":  round(metrics["cvm_stat"], 2),
        })

    # ── Compendio final ────────────────────────────────────────────
    compendium = {}
    for i, kernel in enumerate(KERNELS):
        cv_row    = cv_metric_rows[i]
        scott_row = next(r for r in scott_rows if r["kernel"] == kernel)
        gpu       = GPU_TIMING[kernel]
        amise     = AMISE_PROPS[kernel]
        gridconv  = GRIDSIZE_CONVERGENCE[kernel]

        compendium[kernel] = {
            # Bandwidth
            "h_cv":           cv_row["h_cv"],
            "h_scott":        bw_scott,
            "h_silverman":    round(bw_silverman, 4),
            "ratio_cv_scott": cv_row["ratio_cv_scott"],
            # Statistical quality — h_cv propio
            "mass_cv":        cv_row["mass_cv"],
            "mode_cv":        cv_row["mode_cv"],
            "ks_cv":          cv_row["ks_cv"],
            "cvm_cv":         cv_row["cvm_cv"],
            # Statistical quality — Scott común
            "mass_scott":     scott_row["mass_scott"],
            "mode_scott":     scott_row["mode_scott"],
            "ks_scott":       scott_row["ks_scott"],
            "cvm_scott":      scott_row["cvm_scott"],
            # Computational
            "gpu_t_ref_s":    gpu["t_ref_s"],
            "gpu_t_sens_s":   gpu["t_sens_s"],
            "gpu_pts_per_s":  round(100_000 / gpu["t_ref_s"], 0),
            "cpu_fit_eval_s": cv_row["cpu_time_s"],
            # Theoretical
            "amise_pct":      amise["amise_pct"],
            "h_eq":           amise["h_eq"],
            "support":        amise["support"],
            "ck":             amise["ck"],
            "ops_gpu":        amise["ops"],
            # Gridsize
            "gridsize_opt":   gridconv["gridsize_opt"],
            "delta_opt":      gridconv["delta_opt"],
            "gridsize_ok":    gridconv["converges"],
        }

    (OUT_DIR / "kernel_final_compendium.json").write_text(
        json.dumps(compendium, indent=2), encoding="utf-8"
    )

    # ── Resumen en consola ─────────────────────────────────────────
    print("\n" + "="*80)
    print("COMPENDIO FINAL")
    print("="*80)
    hdr = f"{'Kernel':<14} {'h_cv':>8} {'h_scott':>8} {'ratio':>6} "  \
          f"{'mass_cv':>8} {'KS_cv':>7} {'mass_Sc':>8} {'KS_Sc':>7} "  \
          f"{'tGPU':>7} {'AMISE':>6} {'OK?':>4}"
    print(hdr)
    print("-"*80)
    for kernel in KERNELS:
        d = compendium[kernel]
        ok = "✓" if d["gridsize_ok"] else "✗"
        print(
            f"  {kernel:<12} {d['h_cv']:>8.3g} {d['h_scott']:>8.2f} "
            f"{d['ratio_cv_scott']:>6.3f} "
            f"{d['mass_cv']:>8.4f} {d['ks_cv']:>7.4f} "
            f"{d['mass_scott']:>8.4f} {d['ks_scott']:>7.4f} "
            f"{d['gpu_t_ref_s']:>7.1f} {d['amise_pct']:>6.1f} {ok:>4}"
        )
    print("="*80)
    print(f"\nArchivos guardados en {OUT_DIR}/")
    print("  kernel_cv_all.csv")
    print("  kernel_cv_curves.png")
    print("  kernel_final_compendium.json")


if __name__ == "__main__":
    main()
