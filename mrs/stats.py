"""Small statistical utilities used when SciPy is unavailable on workers."""

import math
from typing import Iterable, Tuple


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    value = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        value *= d * c

        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) < epsilon:
            return value
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if math.isnan(value):
        return math.nan
    if math.isinf(value):
        return 0.0 if value < 0 else 1.0
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        degrees_of_freedom / 2.0, 0.5, x
    )
    return tail if value < 0 else 1.0 - tail


def paired_ttest_less(candidate: Iterable[float], baseline: Iterable[float]) -> Tuple[float, float]:
    """One-sided paired t-test for ``mean(candidate - baseline) < 0``."""
    differences = [float(a) - float(b) for a, b in zip(candidate, baseline)]
    if len(differences) < 2:
        raise ValueError("paired t-test requires at least two pairs")
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    if variance == 0.0:
        if mean < 0.0:
            return -math.inf, 0.0
        if mean > 0.0:
            return math.inf, 1.0
        return 0.0, 0.5
    statistic = mean / math.sqrt(variance / len(differences))
    return statistic, student_t_cdf(statistic, len(differences) - 1)
