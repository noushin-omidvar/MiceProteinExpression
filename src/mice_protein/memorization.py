"""Obfuscation protocol for testing benchmark contamination of tabular models.

Large pretrained tabular models may have encountered public datasets such as this
one. A contaminated model could exploit memorized column names, column order, or
exact values. The protocol re-encodes the data in an information-preserving way:

1. column names are replaced by uninformative tokens (``f00``, ``f01``, ...);
2. column order is permuted; and
3. each column receives a random positive affine transformation ``a * x + b``.

Class information is unchanged, so a model that learns from the data should score
the same on both versions (up to its own sensitivity to scale), whereas a model
relying on memorized identifiers should degrade. Models that standardize their
inputs (e.g. scaled logistic regression) are exactly invariant and serve as a
positive control for the protocol itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def obfuscate(
    features: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return obfuscated features and the key needed to audit the transformation."""
    order = rng.permutation(features.shape[1])
    scale = np.exp(rng.uniform(np.log(0.2), np.log(5.0), features.shape[1]))
    spread = features.std().to_numpy()
    offset = rng.uniform(-3, 3, features.shape[1]) * spread * scale
    transformed = features * scale + offset
    shuffled = transformed.iloc[:, order]
    tokens = [f"f{k:02d}" for k in range(features.shape[1])]
    key = pd.DataFrame(
        {"token": tokens, "original": features.columns[order],
         "scale": scale[order], "offset": offset[order]}
    )
    return shuffled.set_axis(tokens, axis=1), key
