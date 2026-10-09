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
    "ackley_function",
    "beale_function",
    "booth_function",
    "bukin_function",
    "easom_function",
    "eggholder_function",
    "goldstein_price_function",
    "gpd_ll_function",
    "himmelblau_function",
    "himmelblaus_function",
    "levi_function",
    "matyas_function",
    "mccormick_function",
    "rastrigin_function",
    "schaffer_n2_function",
    "sphere_function",
    "three_hump_camel_function",
]
