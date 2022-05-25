import numpy as np
from typing import Union, Tuple


def diff(
    a: Union[np.ndarray, float], b: Union[np.ndarray, float], cv_type: list
) -> Union[np.ndarray, float]:
    """get difference of elements of numbers or arrays
    in range(-inf, inf) if is_angle is False or in range(-pi, pi) if is_angle is True

    Args:
        a: number or array
        b: number or array

    Returns:
        diff: element-wise difference (a-b)
    """
    diff = a - b

    # wrap to range(-pi,pi) for angle
    if hasattr(diff, "__len__") and cv_type == "angle":

        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi

    elif cv_type == "angle":

        if diff < -np.pi:
            diff += 2 * np.pi
        elif diff > np.pi:
            diff -= 2 * np.pi

    return diff


def welford_var(
    count: float, mean: float, M2: float, newValue: float
) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm

    args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample

    returns:
        mean: sample mean,
        M2: sum of powers of differences from the mean
        var: sample variance
    """
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var


def cond_avg(a: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """get hist conditioned average of a, elements with 0 counts set to 0,

    Args:
        a: input array
        hist: histogram along cv (biased probability density)

    Returns:
        cond_avg: conditional average
    """
    return np.divide(a, hist, out=np.zeros_like(a), where=(hist != 0))
