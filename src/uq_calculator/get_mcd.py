import numpy as np


def get_mc_predictions(model, x_data, n_samples=100, framework="torch"):
    """
    Perform Monte Carlo (MC) sampling with dropout enabled during training.

    Parameters:
    model : torch.nn.Module or tf.keras.Model
        The model to sample from.
    x_data : array-like
        Input data for prediction.
    n_samples : int
        Number of MC forward passes.
    framework : str
        Either 'torch' or 'tf'.

    Returns
    mean_pred : np.ndarray
        Mean prediction across MC samples.
    var_pred : np.ndarray
        Variance.
    all_preds : np.ndarray
        All sampled predictions.
    """
    preds = []

    if framework == "torch":
        import torch
        model.train()
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(model(x_data).detach().cpu().numpy())

    elif framework == "tf":
        preds = [model(x_data, training=True).numpy() for _ in range(n_samples)]

    preds = np.stack(preds)
    mean_pred = preds.mean(axis=0)
    var_pred = preds.var(axis=0)

    return mean_pred, var_pred, preds
