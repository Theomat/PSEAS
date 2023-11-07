"""
Provides cdf and pdf methods for a truncated Cauchy distribution.
"""

from typing import Union
import numpy as np

ArrayLike = Union[np.ndarray, float]


def pdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
    a: ArrayLike = -np.inf,
    b: ArrayLike = np.inf,
) -> ArrayLike:
    c: ArrayLike = np.arctan2(b - loc, scale) - np.arctan2(a - loc, scale)
    if isinstance(x, np.ndarray):
        mask: np.ndarray = np.logical_and(a <= x, x <= b)
        out: np.ndarray = np.zeros_like(x)
        if isinstance(loc, np.ndarray):
            loc = loc[mask]
        if isinstance(c, np.ndarray):
            c = c[mask]
        if isinstance(scale, np.ndarray):
            scale = scale[mask]
        out[mask] = 1 / ((1 + np.square((x[mask] - loc) / scale)) * scale * c)
        return out
    else:
        if x < a or x > b:
            return 0
        return 1 / ((1 + np.square((x - loc) / scale)) * scale * c)


def cdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
    a: ArrayLike = -np.inf,
    b: ArrayLike = np.inf,
) -> ArrayLike:
    if scale <= 0:
        return np.ones_like(x)
    const: ArrayLike = np.arctan2(a - loc, scale)
    c: ArrayLike = np.arctan2(b - loc, scale) - const
    return (np.arctan2(np.clip(x, a, b) - loc, scale) - const) / c
