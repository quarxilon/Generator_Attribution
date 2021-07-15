"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import numpy as np


def log_scale(array, epsilon=1e-12):
    """
	Abs + log (ln) scale input array.
    """
    array = np.abs(array)
    array += epsilon # logarithm of zero is undefined
    array = np.log(array)
    return array


def welford(sample):
    """
	Calculates the mean, variance and sample variance along the first axis of an array.
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    existing_aggregate = (None, None, None)
    for data in sample:
        existing_aggregate = _welford_update(existing_aggregate, data)

    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]


def _welford_update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    if count is None:
        count, mean, M2 = 0, np.zeros_like(new_value), np.zeros_like(new_value)

    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    return (count, mean, M2)


def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return (float("nan"), float("nan"), float("nan"))
    else:
        return (mean, variance, sample_variance)


def welford_multidimensional(sample):
    """
	Same as welford() but for multidimensional data; computes along the last axis.
    """
    aggregates = {}

    for data in sample:
        # for each sample, update each axis separately
        for i, d in enumerate(data):
            existing_aggregate = aggregates.get(i, (None, None, None))
            existing_aggregate = _welford_update(existing_aggregate, d)
            aggregates[i] = existing_aggregate

    means, variances = list(), list()

    # in newer python versions dicts would keep their insert order
    for i in range(len(aggregates)):
        aggregate = aggregates[i]
        mean, variance = _welford_finalize(aggregate)[:-1]
        means.append(mean)
        variances.append(variance)

    return np.asarray(means), np.asarray(variances)