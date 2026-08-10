"""Test-time input transforms used as training-free adversarial defences.

Pure numpy/scipy (NO TensorFlow) so the transforms are unit-testable off-GPU. A
defence takes a (possibly adversarial) CT and returns a transformed CT to feed
the *unchanged* model at inference. No retraining: this isolates whether
augmentation applied at TEST time blunts an attack, as opposed to during
training (the retraining study). It is the on-the-fly, label-free analogue of
the geometric/intensity ops in `augmentation.py`.

Volume layout is OpenKBP BDHWC: (batch, D, H, W, channels). These functions
operate on a SINGLE sample (D, H, W, C); the orchestrator (`adversarial_defense`)
handles the batch axis. Anatomical axes (verified on real data via parotid/PTV
landmarks): D=A-P, H=L-R, W=S-I, so the only anatomically valid flip (left-right)
is H = axis 1 of a (D, H, W, C) sample.

Two mechanism classes, deliberately distinct (and the distinction matters when
reading a null result; cf. Athalye et al. 2018 on obfuscated gradients — these
defend only a NON-adaptive attacker):

  * signal-destroying (smooth, noise): a low-pass filter / additive noise that
    erodes the high-frequency adversarial perturbation itself. These can
    genuinely remove the attack, at some cost to clean accuracy.
  * signal-preserving (flip, intensity): mirror / rescale the input. These do
    NOT remove the perturbation; they only help as an *averaged ensemble* of
    predictions against a non-adaptive attacker. A null result here means
    "this transform does not remove the adversarial signal", not "augmentation
    cannot help".
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

# (D, H, W, C) sample: left-right (the only anatomically valid flip) is axis 1.
LR_AXIS_SAMPLE = 1


def smooth_ct(ct: NDArray, sigma: float) -> NDArray:
    """Gaussian low-pass of each channel over (D, H, W).

    Erodes the high-frequency adversarial perturbation. `sigma` is in voxels;
    sigma <= 0 returns a copy unchanged. Border mode is 'nearest' (the volume
    is surrounded by air, so edge replication is harmless).
    """
    ct = np.asarray(ct, dtype=np.float32)
    if sigma <= 0:
        return ct.copy()
    out = np.empty_like(ct)
    for c in range(ct.shape[-1]):
        out[..., c] = gaussian_filter(ct[..., c], sigma=sigma, mode="nearest")
    return out


def add_noise(ct: NDArray, std: float, rng: np.random.Generator) -> NDArray:
    """Additive Gaussian noise, clipped to the normalised CT range [0, 1].

    Stochastic: the orchestrator averages the model prediction over several
    draws (randomised-smoothing style). std <= 0 returns a copy unchanged.
    """
    ct = np.asarray(ct, dtype=np.float32)
    if std <= 0:
        return ct.copy()
    out = ct + rng.normal(0.0, std, size=ct.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def scale_intensity(ct: NDArray, factor: float) -> NDArray:
    """Multiplicative intensity rescale, clipped to [0, 1]. factor == 1 is a no-op."""
    ct = np.asarray(ct, dtype=np.float32)
    return np.clip(ct * np.float32(factor), 0.0, 1.0)


def flip_lr(x: NDArray, axis: int = LR_AXIS_SAMPLE) -> NDArray:
    """Mirror left<->right. Its own EXACT inverse (no interpolation), so the
    orchestrator can flip the input, predict, and flip the output dose back with
    zero round-trip error."""
    return np.ascontiguousarray(np.flip(x, axis=axis))
