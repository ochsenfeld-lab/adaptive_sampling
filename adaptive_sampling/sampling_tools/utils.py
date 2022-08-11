import numpy as np
from typing import Union, Tuple


def diff(
    a: Union[np.ndarray, float], b: Union[np.ndarray, float], cv_type: str
) -> Union[np.ndarray, float]:
    """get difference of elements of numbers or arrays
    in range(-pi, pi) if cv_type='angle' else range(-inf, inf) 

    Args:
        a: number or array
        b: number or array

    Returns:
        diff: element-wise difference (a-b)
    """
    diff = a - b

    # wrap to range(-pi,pi) for angle
    if isinstance(diff, np.ndarray) and cv_type == "angle":

        diff[diff > np.pi] -= 2 * np.pi
        diff[diff < -np.pi] += 2 * np.pi

    elif cv_type == "angle":

        if diff < -np.pi:
            diff += 2 * np.pi
        elif diff > np.pi:
            diff -= 2 * np.pi

    return diff


def sum(
    a: Union[np.ndarray, float], b: Union[np.ndarray, float], cv_type: str
) -> Union[np.ndarray, float]:
    """get sum of elements of numbers or arrays
    in range(-pi, pi) if cv_type='angle' else range(-inf, inf) 

    Args:
        a: number or array
        b: number or array

    Returns:
        diff: element-wise difference (a-b)
    """
    sum = a + b

    # wrap to range(-pi,pi) for angle
    if isinstance(diff, np.ndarray) and cv_type == "angle":

        sum[sum > np.pi] -= 2 * np.pi
        sum[sum < -np.pi] += 2 * np.pi

    elif cv_type == "angle":

        if sum < -np.pi:
            sum += 2 * np.pi
        elif sum > np.pi:
            sum -= 2 * np.pi

    return sum


def welford_var(
    count: float, mean: float, M2: float, newValue: float
) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm
    
    Args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample

    Returns:
        mean: sample mean,
        M2: sum of powers of differences from the mean
        var: sample variance
    """
    if count == 0:
        return 0.0, 0.0, 0.0
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var


def combine_welford_stats(
    count_a, 
    mean_a, 
    M2_a, 
    count_b, 
    mean_b, 
    M2_b
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
    """get hist conditioned average of a, elements with 0 counts set to 0,

    Args:
        a: input array
        hist: histogram along cv (biased probability density)

    Returns:
        cond_avg: conditional average
    """
    return np.divide(a, hist, out=np.zeros_like(a), where=(hist != 0))
    