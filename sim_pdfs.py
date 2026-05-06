"""
PDF functions for SN Ia parameter sampling.

Defined in a module (not notebook cells) so they can be pickled
by multiprocessing worker processes on macOS (spawn start method).
"""
import numpy as np
from functools import partial


def asymmetric_gaussian_pdf(x, mu, sigma_minus, sigma_plus):
    norm_factor = np.sqrt(2 / np.pi) / (sigma_minus + sigma_plus)
    return np.where(
        x < mu,
        norm_factor * np.exp(-0.5 * ((x - mu) / sigma_minus) ** 2),
        norm_factor * np.exp(-0.5 * ((x - mu) / sigma_plus) ** 2),
    )


def make_asymmetric_gaussian_pdf(mu, sigma_minus, sigma_plus):
    """Return a picklable partial for the asymmetric Gaussian PDF."""
    return partial(asymmetric_gaussian_pdf, mu=mu, sigma_minus=sigma_minus, sigma_plus=sigma_plus)
