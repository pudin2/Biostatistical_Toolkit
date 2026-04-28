from __future__ import annotations

import json
import sys
import time
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
from kernels.bandwidth import scott_h
from kernels.data import load_otu_positives
from kernels.metadata import AMISE_PROPS, GPU_TIMING, H_EQ_FACTORS

OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(exist_ok=True)

ACTIVE_KERNELS = {"gaussian": "Scott", "epanechnikov": "CV notebook"}


def fit_conditional_density(
    train_values: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float,
    kernel: str,
) -> tuple[np.ndarray, float, float, float]:
    """Fit KDE and return (conditional_density, positive_mass, mode_x, fit_eval_time_s)."""
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    t0 = time.perf_counter()
    kde.fit(train_values.reshape(-1, 1))
    density = np.exp(kde.score_samples(x_grid.reshape(-1, 1)))
    elapsed = time.perf_counter() - t0
    positive_mass = float(np.trapezoid(density, x_grid))
    conditional_density = density / positive_mass
    mode_x = float(x_grid[np.argmax(conditional_density)])
    return conditional_density, positive_mass, mode_x, elapsed


def main() -> None:
    values = load_otu_positives()
    train_values, test_values = train_test_split(values, test_size=0.2, random_state=42)
    test_sample = test_values[:3000]

    std = float(np.std(values, ddof=1))
    bw_scott = scott_h(values)
    bw_active_epanechnikov = 4.42837

    scaled_bw = {kernel: bw_scott * H_EQ_FACTORS[kernel] for kernel in KERNELS}

    x_grid = np.logspace(np.log10(values.min() * 1e-3), np.log10(values.max() + 8 * max(scaled_bw.values())), 1500)
    dx = np.diff(x_grid, prepend=x_grid[0])

    common_rows: list[dict] = []
    common_densities: dict[str, np.ndarray] = {}

    for kernel in KERNELS:
        density, positive_mass, mode_x, fit_eval_time = fit_conditional_density(
            train_values, x_grid, bw_scott, kernel
        )
        cdf = np.cumsum(density * dx)
        cdf[-1] = 1.0
        cdf_fn = interp1d(
            x_grid,
            cdf,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 1.0),
            assume_sorted=True,
        )
        ks = kstest(test_sample, lambda z, f=cdf_fn: np.asarray(f(z), dtype=float), method="asymp")
        cvm = cramervonmises(test_sample, lambda z, f=cdf_fn: np.asarray(f(z), dtype=float))
        gpu_t = GPU_TIMING[kernel]
        amise = AMISE_PROPS[kernel]
        common_rows.append(
            {
                "kernel": kernel,
                "bandwidth_common_scott": bw_scott,
                "positive_mass_x_gt_0": positive_mass,
                "mode_x": mode_x,
                "ks_stat": float(ks.statistic),
                "cvm_stat": float(cvm.statistic),
                "cpu_fit_eval_time_s": round(fit_eval_time, 3),
                "gpu_t_ref_s": gpu_t["t_ref_s"],
                "gpu_t_sensitivity_s": gpu_t["t_sens_s"],
                "support": amise["support"],
                "ops_per_element": amise["ops"],
                "amise_efficiency_pct": amise["amise_pct"],
                "h_eq_factor_vs_gaussian": H_EQ_FACTORS[kernel],
            }
        )
        common_densities[kernel] = density

    common_df = pd.DataFrame(common_rows).sort_values("ks_stat").reset_index(drop=True)
    common_df.to_csv(OUT_DIR / "kernel_common_scott_metrics.csv", index=False)

    # Save simplified timing CSV for easy slide reference
    timing_rows = []
    for kernel in KERNELS:
        gpu_t = GPU_TIMING[kernel]
        amise = AMISE_PROPS[kernel]
        timing_rows.append({
            "kernel": kernel,
            "support": amise["support"],
            "ops_per_element": amise["ops"],
            "amise_efficiency_pct": amise["amise_pct"],
            "h_eq_factor": H_EQ_FACTORS[kernel],
            "gpu_t_ref_100k_s": gpu_t["t_ref_s"],
            "gpu_t_sensitivity_90evals_s": gpu_t["t_sens_s"],
            "gpu_throughput_pts_per_s": round(100_000 / gpu_t["t_ref_s"], 0),
        })
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(OUT_DIR / "kernel_timing_metrics.csv", index=False)

    pairwise_rows: list[dict] = []
    js_matrix = pd.DataFrame(np.zeros((len(KERNELS), len(KERNELS))), index=list(KERNELS), columns=list(KERNELS))

    for i, kernel_a in enumerate(KERNELS):
        density_a = common_densities[kernel_a]
        cdf_a = np.cumsum(density_a * dx)
        cdf_a[-1] = 1.0
        for j, kernel_b in enumerate(KERNELS):
            if i == j:
                continue
            density_b = common_densities[kernel_b]
            mix = 0.5 * (density_a + density_b)
            eps = 1e-15
            js_value = float(
                0.5 * np.trapezoid(density_a * np.log((density_a + eps) / (mix + eps)), x_grid)
                + 0.5 * np.trapezoid(density_b * np.log((density_b + eps) / (mix + eps)), x_grid)
            )
            js_matrix.loc[kernel_a, kernel_b] = js_value
            if j > i:
                cdf_b = np.cumsum(density_b * dx)
                cdf_b[-1] = 1.0
                pairwise_rows.append(
                    {
                        "kernel_a": kernel_a,
                        "kernel_b": kernel_b,
                        "js_divergence": js_value,
                        "ks_distance": float(np.max(np.abs(cdf_a - cdf_b))),
                        "l1_distance": float(np.trapezoid(np.abs(density_a - density_b), x_grid)),
                    }
                )

    pairwise_df = pd.DataFrame(pairwise_rows).sort_values("js_divergence").reset_index(drop=True)
    pairwise_df.to_csv(OUT_DIR / "kernel_pairwise_common_scott.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(js_matrix.loc[list(KERNELS), list(KERNELS)].to_numpy(), cmap="YlOrRd")
    ax.set_xticks(range(len(KERNELS)), list(KERNELS), rotation=45, ha="right")
    ax.set_yticks(range(len(KERNELS)), list(KERNELS))
    ax.set_title("Divergencia Jensen-Shannon entre kernels\nScott comun = 88.57")
    for r in range(len(KERNELS)):
        for c in range(len(KERNELS)):
            ax.text(c, r, f"{js_matrix.iloc[r, c]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="JS")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kernel_pairwise_js_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for kernel in KERNELS:
        ax.plot(x_grid, common_densities[kernel], label=kernel, linewidth=1.8, color=COLOR_MAP[kernel])
    ax.set_xscale("log")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad condicional en x > 0")
    ax.set_title("Distribuciones KDE por kernel\nScott comun = 88.57")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kernel_common_scott_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    active_specs = {"gaussian": bw_scott, "epanechnikov": bw_active_epanechnikov}
    active_rows: list[dict] = []
    active_densities: dict[str, np.ndarray] = {}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for kernel, bandwidth in active_specs.items():
        density, positive_mass, mode_x, fit_eval_time = fit_conditional_density(
            train_values, x_grid, bandwidth, kernel
        )
        active_densities[kernel] = density
        cdf = np.cumsum(density * dx)
        cdf[-1] = 1.0
        cdf_fn = interp1d(
            x_grid,
            cdf,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 1.0),
            assume_sorted=True,
        )
        ks = kstest(test_sample, lambda z, f=cdf_fn: np.asarray(f(z), dtype=float), method="asymp")
        cvm = cramervonmises(test_sample, lambda z, f=cdf_fn: np.asarray(f(z), dtype=float))
        active_rows.append(
            {
                "kernel": kernel,
                "bandwidth": bandwidth,
                "source": ACTIVE_KERNELS[kernel],
                "positive_mass_x_gt_0": positive_mass,
                "mode_x": mode_x,
                "ks_stat": float(ks.statistic),
                "cvm_stat": float(cvm.statistic),
                "cpu_fit_eval_time_s": round(fit_eval_time, 3),
                "gpu_t_ref_s": GPU_TIMING[kernel]["t_ref_s"],
            }
        )
        ax.plot(
            x_grid,
            density,
            label=f"{kernel} (bw={bandwidth:.3f})",
            linewidth=2.0,
            color=COLOR_MAP[kernel],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad condicional en x > 0")
    ax.set_title("Kernels actualmente usados en el flujo")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kernel_active_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    active_df = pd.DataFrame(active_rows)
    active_df.to_csv(OUT_DIR / "kernel_active_metrics.csv", index=False)

    density_a = active_densities["gaussian"]
    density_b = active_densities["epanechnikov"]
    cdf_a = np.cumsum(density_a * dx)
    cdf_b = np.cumsum(density_b * dx)
    cdf_a[-1] = 1.0
    cdf_b[-1] = 1.0
    mix = 0.5 * (density_a + density_b)
    eps = 1e-15
    active_pair = {
        "js_divergence": float(
            0.5 * np.trapezoid(density_a * np.log((density_a + eps) / (mix + eps)), x_grid)
            + 0.5 * np.trapezoid(density_b * np.log((density_b + eps) / (mix + eps)), x_grid)
        ),
        "ks_distance": float(np.max(np.abs(cdf_a - cdf_b))),
        "l1_distance": float(np.trapezoid(np.abs(density_a - density_b), x_grid)),
    }

    # Reconstruir kernel_complexity y kernel_gpu_timing en el formato exacto
    # del JSON original (combina AMISE_PROPS, H_EQ_FACTORS y GPU_TIMING).
    kernel_complexity_payload = {
        kernel: {
            "ops_per_element": AMISE_PROPS[kernel]["ops"],
            "support":         AMISE_PROPS[kernel]["support"],
            "amise_eff_pct":   AMISE_PROPS[kernel]["amise_pct"],
            "h_eq_factor":     H_EQ_FACTORS[kernel],
        }
        for kernel in KERNELS
    }
    kernel_gpu_timing_payload = {
        kernel: {
            "t_ref_s":                GPU_TIMING[kernel]["t_ref_s"],
            "t_total_sensitivity_s":  GPU_TIMING[kernel]["t_sens_s"],
        }
        for kernel in KERNELS
    }

    summary = {
        "data": {
            "n_positive": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": std,
            "bw_scott": bw_scott,
        },
        "hardware": {
            "backend": "GPU (CuPy) / CPU (sklearn fallback)",
            "gpu_vram_total_gb": 8.6,
            "gpu_vram_free_gb": 7.4,
            "cuda_version": "13.1",
            "n_data_points": int(len(values)),
            "gridsize_ref": 100_000,
            "chunk_eval_points": 1755,
            "note": "Tiempos GPU medidos en KDE_Gridsize_Sensitivity_MultiKernel.ipynb",
        },
        "scaled_bandwidths": scaled_bw,
        "kernel_complexity": kernel_complexity_payload,
        "kernel_gpu_timing": kernel_gpu_timing_payload,
        "common_scott_metrics": common_rows,
        "active_metrics": active_rows,
        "active_pair": active_pair,
    }
    (OUT_DIR / "kernel_report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Assets generados:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
