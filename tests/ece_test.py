import numpy as np
from sklearn.linear_model import LinearRegression
from ece_calculator import get_ece


def test_low_ece():
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = 2 * X.flatten() + 1 + np.random.normal(0, 1, 100)

    model = LinearRegression()
    model.fit(X, y_true)
    mu = model.predict(X)

    residuals = y_true - mu
    sigma = np.ones_like(mu) * np.std(residuals)
    ece = get_ece(y_true, mu, sigma)

    print(f"Good uncertainty ECE: {ece:.2f}%")
    return ece


def test_high_ece():
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = 2 * X.flatten() + 1 + np.random.normal(0, 1, 100)

    model = LinearRegression()
    model.fit(X, y_true)
    mu = model.predict(X)

    sigma = np.ones_like(mu) * 0.1
    ece = get_ece(y_true, mu, sigma)

    print(f"Overconfident ECE: {ece:.2f}%")
    return ece


if __name__ == "__main__":
    low_ece = test_low_ece()
    high_ece = test_high_ece()

    print(f"Low ECE: {low_ece:.2f}%")
    print(f"High ECE: {high_ece:.2f}%")

#test with tf and pt