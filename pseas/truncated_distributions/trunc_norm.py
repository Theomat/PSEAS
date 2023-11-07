"""
Provides cdf and pdf methods for a truncated normal distribution.
"""
from typing import Union
import numpy as np

import scipy.stats as st

ArrayLike = Union[np.ndarray, float]


def pdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
    a: ArrayLike = -np.inf,
    b: ArrayLike = np.inf,
) -> ArrayLike:
    a = (a - loc) / scale
    b = (b - loc) / scale
    return st.truncnorm.pdf(x, loc=loc, scale=scale, a=a, b=b)


def cdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
    a: ArrayLike = -np.inf,
    b: ArrayLike = np.inf,
) -> ArrayLike:
    a = (a - loc) / scale
    b = (b - loc) / scale
    return st.truncnorm.cdf(x, loc=loc, scale=scale, a=a, b=b)
