import numpy as np
from uq_calculator import get_mcd

# Simple fake model (PyTorch-like)
class DummyTorchModel:
    def __call__(self, x):
        # add random noise to simulate stochastic dropout
        return x + np.random.normal(0, 0.1, x.shape)

# Generate synthetic input data
x_data = np.random.rand(5, 3)  # 5 samples, 3 features

# Instantiate model and run MC sampling
model = DummyTorchModel()
mean_pred, var_pred, preds = get_mcd(model, x_data, n_samples=10, framework="torch")

print("Mean predictions:\n", mean_pred)
print("Variance estimates:\n", var_pred)
print("All samples shape:", preds.shape)
