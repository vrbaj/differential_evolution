"""Backward-compatible benchmark function imports."""

from differential_evolution.benchmarks import (
    ackley_function,
    beale_function,
    booth_function,
    bukin_function,
    easom_function,
    eggholder_function,
    goldstein_price_function,
    gpd_ll_function,
    himmelblau_function,
    levi_function,
    mccormick_function,
    matyas_function,
    rastrigin_function,
    schaffer_n2_function,
    sphere_function,
    three_hump_camel_function,
)

himmelblaus_function = himmelblau_function

__all__ = [
    "sphere_function",
    "rastrigin_function",
    "beale_function",
    "booth_function",
    "matyas_function",
    "himmelblau_function",
    "himmelblaus_function",
    "bukin_function",
    "mccormick_function",
    "three_hump_camel_function",
    "ackley_function",
    "goldstein_price_function",
    "levi_function",
    "easom_function",
    "eggholder_function",
    "schaffer_n2_function",
    "gpd_ll_function",
]
