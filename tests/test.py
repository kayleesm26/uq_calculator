from ece_calculator import get_ece
import numpy as np

print("Testing Good vs Bad Calibration")
print("=" * 40)

# Well-calibrated UQ (should have low ECE)
np.random.seed(42)
n_samples = 1000
y_true = np.random.normal(0, 1, n_samples)
mu = np.zeros(n_samples)  # Predict mean = 0
sigma_good = np.ones(n_samples) * 1.0  # Correct uncertainty for N(0,1)

ece_good = get_ece(y_true, mu, sigma_good)
print(f"Well-calibrated ECE: {ece_good:.2f}% ✅")

# Bad calibration tests:

# 1. Overconfident (uncertainty too small)
sigma_overconfident = np.ones(n_samples) * 0.1  # Severely underestimated
ece_over = get_ece(y_true, mu, sigma_overconfident)
print(f"Overconfident ECE: {ece_over:.2f}% ❌")

# 2. Underconfident (uncertainty too large)
sigma_underconfident = np.ones(n_samples) * 3.0  # Overestimated
ece_under = get_ece(y_true, mu, sigma_underconfident)
print(f"Underconfident ECE: {ece_under:.2f}% ❌")

# 3. Wrong means + wrong uncertainty
mu_bad = np.ones(n_samples) * 2.0  # Biased predictions
sigma_bad = np.ones(n_samples) * 0.5  # Wrong uncertainty
ece_bad = get_ece(y_true, mu_bad, sigma_bad)
print(f"Bad means + bad uncertainty ECE: {ece_bad:.2f}% ❌")

print("\nInterpretation:")
print("0-5%: Excellent calibration")
print("5-10%: Good calibration")
print("10-20%: Poor calibration")
print("20%+: Very poor calibration")