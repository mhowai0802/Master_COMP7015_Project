from typing import Optional

import numpy as np

from domain.stocks import StockPriceSeries


def _daily_returns(series: StockPriceSeries) -> np.ndarray:
    """
    Compute simple daily returns from a StockPriceSeries.
    """

    closes = np.array([p.close for p in series.prices], dtype=np.float64)
    if len(closes) < 2:
        return np.array([], dtype=np.float64)
    returns = closes[1:] / closes[:-1] - 1.0
    return returns


def compute_scenario_params(
    series: StockPriceSeries, sentiment: Optional[float] = None
) -> dict:
    """
    Compute the basic parameters used in the simple scenario model:

    - mu: historical average daily return
    - sigma: historical daily return standard deviation
    - mu_tilted: sentiment-adjusted mean (what the simulator actually uses)
    - n_obs: number of daily return observations
    """

    rets = _daily_returns(series)
    if rets.size == 0:
        return {"mu": 0.0, "sigma": 0.0, "mu_tilted": 0.0, "n_obs": 0}

    mu = float(np.mean(rets))
    sigma = float(np.std(rets) + 1e-8)

    if sentiment is not None:
        mu_tilted = mu + 0.5 * abs(mu) * float(sentiment)
    else:
        mu_tilted = mu

    return {
        "mu": mu,
        "sigma": sigma,
        "mu_tilted": mu_tilted,
        "n_obs": int(rets.size),
    }


def simulate_paths(
    series: StockPriceSeries,
    horizon_days: int = 20,
    n_paths: int = 1000,
    sentiment: Optional[float] = None,
) -> np.ndarray:
    """
    Simple generative scenario model:

    - Estimate historical daily return mean and std from the past period.
    - Optionally tilt the mean slightly based on sentiment:
        positive sentiment → slightly higher mean return
        negative sentiment → slightly lower mean return
    - Sample i.i.d. Gaussian returns and build price paths.

    Returns:
        paths: np.ndarray of shape (n_paths, horizon_days+1)
               containing simulated price levels, normalised so that
               paths[:, 0] == 1.0
    """

    params = compute_scenario_params(series, sentiment=sentiment)
    if params["n_obs"] == 0:
        return np.ones((n_paths, horizon_days + 1), dtype=np.float64)

    mu = params["mu_tilted"]
    sigma = float(params["sigma"])

    # Sample daily returns
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma, size=(n_paths, horizon_days))

    # Convert to price paths starting at 1.0
    paths = np.ones((n_paths, horizon_days + 1), dtype=np.float64)
    for t in range(1, horizon_days + 1):
        paths[:, t] = paths[:, t - 1] * (1.0 + samples[:, t - 1])

    return paths


def summarize_paths(paths: np.ndarray) -> dict:
    """
    Summarise Monte Carlo paths into useful risk / scenario metrics.

    Returns a dict with:
        - up_prob: probability final return > 0
        - median_return
        - p10_return, p90_return  (10% / 90% quantiles)
        - worst_return, best_return
    """

    if paths.size == 0:
        return {}

    final = paths[:, -1]
    total_returns = final - 1.0  # relative to start

    up_prob = float(np.mean(total_returns > 0.0))
    median = float(np.median(total_returns))
    p10 = float(np.percentile(total_returns, 10))
    p90 = float(np.percentile(total_returns, 90))
    worst = float(np.min(total_returns))
    best = float(np.max(total_returns))

    return {
        "up_prob": up_prob,
        "median_return": median,
        "p10_return": p10,
        "p90_return": p90,
        "worst_return": worst,
        "best_return": best,
    }


