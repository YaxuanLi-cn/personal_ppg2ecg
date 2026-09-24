import numpy as np
import torch
from scipy.linalg import eigh

from metrics import zscore_numpy, zscore_tensor


def matrix_power_psd(matrix, exponent):
    values, vectors = eigh((matrix + matrix.T) * 0.5, check_finite=True)
    return (vectors * np.maximum(values, 1e-10) ** exponent) @ vectors.T


def apply_transport(prediction, layers, strength=1.0):
    if not 0 <= strength <= 1:
        raise ValueError('Transport strength must be between zero and one')
    x = zscore_numpy(prediction)[:, :, 0]
    for layer in layers:
        matrix = np.asarray(layer['matrix'], dtype=np.float64)
        source = np.asarray(layer['source_mean'], dtype=np.float64)
        target = np.asarray(layer['target_mean'], dtype=np.float64)
        corrected = (x - source) @ matrix + target
        x = zscore_numpy((x + strength * (corrected - x))[..., None])[:, :, 0]
    return x[..., None].astype(np.float32)


def transport_tensor(prediction, layers, strength):
    x = zscore_tensor(prediction).squeeze(-1)
    for matrix, source, target in layers:
        corrected = (x - source) @ matrix + target
        x = zscore_tensor((x + strength * (corrected - x)).unsqueeze(-1)).squeeze(-1)
    return x.unsqueeze(-1)


def fit_transport(real_training, predicted_training, iterations=2, regularization=0.01, max_gain=3.0):
    if real_training.shape != predicted_training.shape or real_training.ndim != 3 or real_training.shape[-1] != 1:
        raise ValueError('Expected matching [N,L,1] training arrays')
    if not np.isfinite(real_training).all() or not np.isfinite(predicted_training).all() or len(real_training) < 2:
        raise ValueError('Expected finite nonempty calibration data')
    if regularization <= 0 or max_gain < 1 or iterations < 1:
        raise ValueError('Invalid covariance regularization, gain bound or iterations')
    real = zscore_numpy(real_training)[:, :, 0]
    target_mean = real.mean(axis=0)
    cov_real = np.cov(real, rowvar=False) + regularization * np.eye(real.shape[1])
    layers = []
    current = np.asarray(predicted_training)
    for _ in range(iterations):
        fake = zscore_numpy(current)[:, :, 0]
        source_mean = fake.mean(axis=0)
        cov_fake = np.cov(fake, rowvar=False) + regularization * np.eye(fake.shape[1])
        root = matrix_power_psd(cov_fake, 0.5)
        inverse = matrix_power_psd(cov_fake, -0.5)
        matrix = inverse @ matrix_power_psd(root @ cov_real @ root, 0.5) @ inverse
        values, vectors = eigh((matrix + matrix.T) * 0.5)
        matrix = (vectors * values.clip(1 / max_gain, max_gain)) @ vectors.T
        layer = {'matrix': matrix.astype(np.float32), 'source_mean': source_mean.astype(np.float32),
                 'target_mean': target_mean.astype(np.float32)}
        layers.append(layer)
        current = apply_transport(current, [layer])
    return layers
