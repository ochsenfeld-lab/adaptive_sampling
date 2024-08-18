import numpy as np
from typing import Union, Tuple


def diff(
    a: Union[np.ndarray, float],
    b: Union[np.ndarray, float],
    periodicity: list,
) -> Union[np.ndarray, float]:
    """get (periodic) difference of elements of numbers or arrays

    Args:
        a: number or array
        b: number or array
        periodicity: periodic boundary conditions [lower, upper]

    Returns:
        diff: element-wise difference (a-b)
    """
    diff_ab = a - b
    diff_ab = correct_periodicity(diff_ab, periodicity)
    return diff_ab


def sum(
    a: Union[np.ndarray, float],
    b: Union[np.ndarray, float],
    periodicity: list,
) -> Union[np.ndarray, float]:
    """get (periodic) sum of elements of numbers or arrays

    Args:
        a: number or array
        b: number or array
        periodicity: periodic boundary conditions [lower, upper]

    Returns:
        diff: element-wise difference (a-b)
    """
    sum_ab = a + b
    sum_ab = correct_periodicity(sum_ab, periodicity)
    return sum_ab


def correct_periodicity(
    x: Union[np.ndarray, float],
    periodicity: list,
) -> Union[np.ndarray, float]:
    """Wrap x to periodic range

    Args:
        x: float or array to correct
        periodicity: periodic boundary conditions ([lower, upper]),
                     if None, returns x

    Returns:
        x: x in periodic range defined by periodicity
    """
    if not periodicity:
        return x

    if len(periodicity) != 2:
        raise ValueError("Invalid periodicity")

    period = periodicity[1] - periodicity[0]
    if isinstance(x, np.ndarray):
        x[x > periodicity[1]] -= period
        x[x < periodicity[0]] += period
    else:
        if x > periodicity[1]:
            x -= period
        elif x < periodicity[0]:
            x += period
    return x


def welford_var(
    count: float,
    mean: float,
    M2: float,
    newValue: float,
    tau: float = None,
) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm


    Args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample
        tau: decay time for exponentially decaying running average

    Returns:
        mean: sample mean,
        M2: sum of powers of differences from the mean
        var: sample variance
    """
    if tau != None and tau < 1:
        raise ValueError(
            " >>> Decay time for Welford's variance has to be bigger than 1."
        )

    if count == 0:
        return 0.0, 0.0, 0.0

    delta = newValue - mean
    if tau == None:
        mean += delta / count
    else:
        mean += delta / tau
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var


def combine_welford_stats(
    count_a, mean_a, M2_a, count_b, mean_b, M2_b
) -> Tuple[float, float, float, float]:
    """Combines running sample stats of welford's algorithm using Chan et al.'s algorithm.

    args:
        count_a, mean_a, M2_a: stats of frist subsample
        count_a, mean_a, M2_a: stats of second subsample

    returns:
        mean, M2 and sample variance of combined samples
    """
    count = count_a + count_b
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    delta = mean_b - mean_a
    mean = mean_a + delta * count_b / count
    if count_b == 0:
        M2 = M2_a
    else:
        M2 = M2_a + M2_b + (delta * delta) * ((count_a / count_b) / count)
    var = M2 / count if count > 2 else 0.0
    return count, mean, M2, var


def cond_avg(a: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """get hist conditioned average of a, elements with 0 counts set to 0

    Args:
        a: input array
        hist: histogram along cv (biased probability density)

    Returns:
        cond_avg: conditional average
    """
    return np.divide(a, hist, out=np.zeros_like(a), where=(hist != 0))
