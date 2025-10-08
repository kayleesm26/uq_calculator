import numpy as np
from scipy.stats import norm

def get_ece(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_levels: int = 1000) -> float:
    """
    Regression Expected Calibration Error (ECE).

    Parameters
    y_true : np.ndarray
        True target values (N,)
    mu : np.ndarray
        Predicted means (N,)
    sigma : np.ndarray
        Predicted standard deviations (N,)
    n_levels : int
        Number of confidence levels to evaluate (default = 1000, higher = smoother).

    Returns
    float
        ECE in percentage points. Smaller is better (0 = perfect calibration).
    """
    exp_cis = np.linspace(1e-10, 1 - 1e-10, n_levels)
    pred_cis = []

    for ci in exp_cis:
        lower, upper = norm.interval(ci, loc=mu, scale=sigma)
        coverage = ((y_true > lower) & (y_true < upper)).mean()
        pred_cis.append(coverage)

    exp_cis = np.array(exp_cis)
    pred_cis = np.array(pred_cis)

    return 100 * np.mean(np.abs(exp_cis - pred_cis))
