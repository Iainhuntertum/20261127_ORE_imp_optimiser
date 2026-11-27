"""Refit the impedance mapping from a MATLAB file containing fit inputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def phi(x, Apos, mu, s):
    return Apos * (sigmoid((x - mu) / s) - sigmoid(-mu / s))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fit_data", nargs="?", type=Path,
        default=Path(__file__).resolve().with_name("20261127_ORE_imp_opt_fit_data.mat"),
        help="MATLAB fit input (defaults to the file beside this script)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("fit_results"))
    args = parser.parse_args()
    if not args.fit_data.is_file():
        parser.error(f"Fit input does not exist: {args.fit_data}")

    data = loadmat(args.fit_data)
    adapt_start = int(data["adapt_start"].item())
    adapt_end = int(data["adapt_end"].item())

    # The publication export stores bounds 951 and 1011 for the Python slice.
    # This selects MATLAB columns 952 through 1011 with same-trial pairing.
    x = data["con_norm_delta_error"][:, adapt_start:adapt_end].reshape(-1)
    y = data["con_norm_delta_cci"][:, adapt_start:adapt_end].reshape(-1)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid], y[valid]

    params, _ = curve_fit(
        phi, x, y,
        p0=[0.5, 0.08, 0.25],
        bounds=([0.0, 0.0, 1e-3], [2.0, 2.0, 2.0]),
        maxfev=20000,
    )
    print("Fit parameters [A_plus, mu, s]:", params)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    params_path = args.out_dir / "refitted_imp_opt_params.npy"
    np.save(params_path, params)

    x_grid = np.linspace(x.min(), x.max(), 500)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=10, alpha=0.15, label="Data")
    ax.plot(x_grid, phi(x_grid, *params), linewidth=2, label=r"Fitted $\phi$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set(xlabel=r"$\Delta$MPE", ylabel=r"$\Delta$CCI")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    figure_path = args.out_dir / "human_impedance_fit.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"Saved {params_path} and {figure_path}")


if __name__ == "__main__":
    main()
