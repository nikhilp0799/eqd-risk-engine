import numpy as np
import pytest

from eqdrisk.vol.ssvi import SSVIParams, fit_ssvi


def test_fit_ssvi_recovers_known_parameters():
    true = SSVIParams(rho=-0.5, eta=1.0)
    slices = []
    for theta in [0.02, 0.05, 0.10]:
        k = np.linspace(-0.4, 0.4, 20)
        w = true.total_variance(k, theta)
        slices.append((theta, k, w, np.ones_like(k)))

    fit = fit_ssvi(slices)

    assert fit.rho == pytest.approx(true.rho, rel=1e-4)
    assert fit.eta == pytest.approx(true.eta, rel=1e-4)
    for theta, k, w, _weights in slices:
        assert np.max(np.abs(fit.total_variance(k, theta) - w)) < 1e-6


def test_fit_ssvi_respects_no_arbitrage_feasibility():
    # Even with data that would "want" an infeasible eta, the fit must stay within
    # the sufficient no-static-arbitrage condition eta*(1+|rho|) <= 2.
    rng = np.random.default_rng(0)
    slices = []
    for theta in [0.01, 0.03, 0.06, 0.12]:
        k = np.linspace(-0.6, 0.6, 15)
        w = theta * (1 + 0.9 * np.abs(k)) + rng.normal(0, 0.001, size=k.shape)
        slices.append((theta, k, np.clip(w, 1e-6, None), np.ones_like(k)))

    fit = fit_ssvi(slices)

    assert fit.eta * (1 + abs(fit.rho)) <= 2.0 + 1e-9
