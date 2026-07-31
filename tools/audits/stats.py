"""Interval estimates shared by the blind-listening audits.

Both are deliberately simple: at the counts these audits run at (n=25..60 per
stratum) the honest answer is "does this clear zero", not a third decimal place,
and each function names its own approximation so a reader cannot mistake it for
something tighter than it is.
"""

from __future__ import annotations

from typing import Any


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a single proportion.

    Preferred over the normal approximation because these audits routinely land
    on 0/25 or 25/25, where the normal interval collapses to a point.
    """
    if total <= 0:
        return None
    rate = hits / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5)
    half /= denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_difference(
    rate_a: float | None, total_a: int, rate_b: float, total_b: int
) -> dict[str, Any] | None:
    """Is this rate actually different from the one it is being compared to?

    A normal-approximation interval on the difference. It is the right shape of
    answer here - the question is whether the difference clears zero, not what
    the difference is to three places - and it is labelled approximate because at
    these counts it is.
    """
    if rate_a is None or total_a <= 0 or total_b <= 0:
        return None
    variance = rate_a * (1 - rate_a) / total_a + rate_b * (1 - rate_b) / total_b
    half = 1.96 * (variance**0.5)
    difference = rate_a - rate_b
    return {
        "difference": round(difference, 4),
        "ci95_approx": [round(difference - half, 4), round(difference + half, 4)],
        "separated_from_reference": bool(abs(difference) > half),
        "method": "normal approximation on the difference of two proportions",
    }
