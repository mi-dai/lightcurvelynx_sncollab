"""
LSST SN Ia simulation pipeline.

Forward-models SN Ia light curves with the Rubin/LSST baseline cadence using
lightcurvelynx + SALT3 via sncosmo.

Usage:
    python lsst_snia_pipeline.py [--config sim_params.yaml] [--output path.parquet] [--plots]
                                 [--parallel-executor {loky,dask,none}]
"""

# --- 0. Environment setup ---
# LIGHTCURVELYNX_DATA_DIR must be set before lightcurvelynx imports;
# the download/cache path is resolved at import time.
import os
from pathlib import Path

_data_dir = Path(__file__).parent / "data"
_data_dir.mkdir(exist_ok=True)
os.environ["LIGHTCURVELYNX_DATA_DIR"] = str(_data_dir)

# --- 1. Imports ---
import argparse

import matplotlib
matplotlib.use("Agg")  # headless; no display required
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import interp1d

from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.astro_utils.snia_utils import (
    DistModFromRedshift,
    X0FromDistMod,
    num_snia_per_redshift_bin,
    snia_volumetric_rates,
)
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.ra_dec_sampler import ApproximateMOCSampler
from lightcurvelynx.math_nodes.scipy_random import SamplePDF
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.utils.extrapolate import LinearDecayOnMag, ZeroPadding


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def estimate_nsn(cfg: dict) -> int:
    """Integrate Frohmaier+2019 volumetric rate to get total expected SN Ia count."""
    solid_angle = cfg["sky_coverage"] * (np.pi / 180.0) ** 2
    survey_length = (cfg["tmax"] - cfg["tmin"]) / 365.25
    nsntotal, _ = num_snia_per_redshift_bin(
        cfg["zmin"],
        cfg["zmax"],
        znbins=1,
        solid_angle=solid_angle,
        vol_rate_function=snia_volumetric_rates,
        H0=cfg["H0"],
        Omega_m=cfg["Omega_m"],
    )
    return int(nsntotal[0] * survey_length)


def save_diagnostic_plots(results, cfg: dict, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Redshift distribution
    fig, ax = plt.subplots()
    results["source_redshift"].hist(bins=50, ax=ax)
    ax.set(xlabel="Redshift", ylabel="Count", title="Simulated redshift distribution")
    fig.savefig(plot_dir / "redshift_dist.png", dpi=150)
    plt.close(fig)

    # x1 and c distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    results["source_x1"].hist(bins=50, ax=axes[0])
    axes[0].set(xlabel="x1", ylabel="Count")
    results["source_c"].hist(bins=50, ax=axes[1])
    axes[1].set(xlabel="c", ylabel="Count")
    plt.tight_layout()
    fig.savefig(plot_dir / "x1_c_dist.png", dpi=150)
    plt.close(fig)

    # Example light curve for one random SN
    rng = np.random.default_rng()
    sn = results.iloc[rng.integers(len(results))]
    lc = sn["lightcurve"]
    fig, ax = plt.subplots()
    for band in cfg["filters"]:
        mask = lc["filter"] == band
        rest_phase = (lc["mjd"] - sn["source_t0"]) / (1.0 + sn["source_redshift"])
        mask &= (rest_phase > -20) & (rest_phase < 100)
        if mask.any():
            ax.errorbar(lc["mjd"][mask], lc["flux"][mask], lc["fluxerr"][mask],
                        fmt="o", label=band, capsize=3)
    ax.axvline(sn["source_t0"], ls="--", color="k", label="t0")
    ax.legend()
    ax.set(xlabel="MJD", ylabel="Flux (nJy)",
           title=f'Example SN Ia  z={sn["source_redshift"]:.3f}')
    fig.savefig(plot_dir / "example_lightcurve.png", dpi=150)
    plt.close(fig)
    print(f"Diagnostic plots saved to {plot_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="LSST SN Ia simulation pipeline")
    parser.add_argument("--config", default="sim_params.yaml",
                        help="Path to YAML config file (default: sim_params.yaml)")
    parser.add_argument("--output", default=None,
                        help="Override output parquet path from config")
    parser.add_argument("--plots", action="store_true",
                        help="Save diagnostic plots to output/plots/")
    parser.add_argument("--parallel-executor", default="dask",
                        choices=["loky", "dask", "none"],
                        help="Parallel executor backend (default: dask)")
    args = parser.parse_args()

    # --- 2. Load config ---
    cfg = load_config(args.config)
    if args.output:
        cfg["output_path"] = args.output

    rng = np.random.default_rng(cfg["seed"])

    # --- 3. Load LSST OpSim schedule ---
    opsim = OpSim.from_db(cfg["opsim_db"])
    print(f"OpSim loaded: {len(opsim):,} observations")
    print(f"MJD range: {opsim['time'].min():.1f} – {opsim['time'].max():.1f}")

    # --- 4. Sky coverage ---
    # -1 = estimate from OpSim unique pointing footprint; otherwise use config value directly
    if cfg["sky_coverage"] == -1:
        cfg["sky_coverage"] = opsim.estimate_coverage()
        print(f"Estimated sky coverage: {cfg['sky_coverage']:.0f} deg²")
    else:
        print(f"Using configured sky coverage: {cfg['sky_coverage']:.0f} deg²")

    # --- 5. Number of SNe Ia to simulate ---
    # -1 = integrate Frohmaier+2019 volumetric rate; positive int = use directly
    if cfg["nsn"] == -1:
        nsn = estimate_nsn(cfg)
        print(f"Expected SNe Ia (from volumetric rate): {nsn:,}")
    else:
        nsn = int(cfg["nsn"])
        print(f"Using configured nsn: {nsn:,}")

    # --- 6. Load LSST passbands ---
    passbands = PassbandGroup.from_preset("LSST", filters=cfg["filters"])
    print(passbands)

    # --- 7. Build redshift PDF ---
    # Expected SNe per redshift bin → interpolated PDF for sampling
    nsn_per_bin, z_mean = num_snia_per_redshift_bin(
        cfg["zmin"],
        cfg["zmax"],
        cfg["znbins"],
        H0=cfg["H0"],
        Omega_m=cfg["Omega_m"],
    )
    zpdf = interp1d(z_mean, nsn_per_bin, bounds_error=False, fill_value=0)

    # --- 8. Build SN Ia source model ---

    # RA/Dec: approximate MOC sampler over the observed OpSim footprint
    moc = opsim.build_moc(max_depth=12)
    radec = ApproximateMOCSampler(moc, node_label="radec")

    # Redshift drawn from volumetric rate PDF
    z_func = SamplePDF(zpdf, node_label="redshift")

    # Asymmetric Gaussian SALT3 priors — plain closures work with both loky and dask.
    def asymmetric_gaussian_pdf(x, mu, sigma_minus, sigma_plus):
        norm_factor = np.sqrt(2 / np.pi) / (sigma_minus + sigma_plus)
        return np.where(
            x < mu,
            norm_factor * np.exp(-0.5 * ((x - mu) / sigma_minus) ** 2),
            norm_factor * np.exp(-0.5 * ((x - mu) / sigma_plus) ** 2),
        )

    def x1_pdf(x):
        return asymmetric_gaussian_pdf(x, cfg["x1_mean"], cfg["x1_sigma_minus"], cfg["x1_sigma_plus"])

    def c_pdf(c):
        return asymmetric_gaussian_pdf(c, cfg["c_mean"], cfg["c_sigma_minus"], cfg["c_sigma_plus"])
    x1_func = SamplePDF(x1_pdf, node_label="x1")
    c_func = SamplePDF(c_pdf, node_label="c")
    m_abs_func = NumpyRandomFunc("normal", loc=cfg["m_abs_mean"], scale=cfg["m_abs_sigma"])

    # x0 via Tripp relation: μ(z) → distmod → x0
    distmod_func = DistModFromRedshift(z_func, H0=cfg["H0"], Omega_m=cfg["Omega_m"])
    x0_func = X0FromDistMod(
        distmod=distmod_func,
        x1=x1_func,
        c=c_func,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        m_abs=m_abs_func,
        node_label="x0_func",
    )

    # Wavelength/time boundary conditions: zero-pad before, linear mag decay after
    source = SncosmoWrapperModel(
        "salt3",
        t0=NumpyRandomFunc("uniform", low=cfg["tmin"], high=cfg["tmax"]),
        x0=x0_func,
        x1=x1_func,
        c=c_func,
        ra=radec.ra,
        dec=radec.dec,
        redshift=z_func,
        node_label="source",
        time_extrapolation=(ZeroPadding(), LinearDecayOnMag(decay_rate=0.02, mag_thres=30.0)),
        wave_extrapolation=(ZeroPadding(), ZeroPadding()),
    )

    # --- 9. Run simulation ---
    param_cols = [
        "source.t0",
        "source.x0",
        "source.x1",
        "source.c",
        "source.redshift",
        "source.ra",
        "source.dec",
        "x0_func.distmod",
    ]
    sim_kwargs = dict(
        model=source,
        num_samples=nsn,
        obstable=opsim,
        passbands=passbands,
        param_cols=param_cols,
        obstable_save_cols=["zp"],
        rng=rng,
        batch_size=cfg["batch_size"],
    )
    parallel_executor = args.parallel_executor
    if parallel_executor == "dask":
        import dask.distributed
        with dask.distributed.Client() as client:
            print(f"Dask dashboard: {client.dashboard_link}")
            results = simulate_lightcurves(**sim_kwargs, executor=client)
    elif parallel_executor == "loky":
        import loky
        executor = loky.get_reusable_executor(max_workers=cfg["num_jobs"])
        results = simulate_lightcurves(**sim_kwargs, executor=executor)
    else:
        results = simulate_lightcurves(**sim_kwargs, executor=None)
    print(f"Simulated {len(results):,} SNe Ia")

    # --- 10. Save results ---
    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path)
    print(f"Saved to {output_path}")

    # --- 11. Diagnostic plots (--plots only) ---
    if args.plots:
        plot_dir = output_path.parent / "plots"
        save_diagnostic_plots(results, cfg, plot_dir)


if __name__ == "__main__":
    main()
