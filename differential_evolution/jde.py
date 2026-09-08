"""Convenience helpers for the jDE control-parameter scheme."""

from __future__ import annotations

from dataclasses import dataclass

from .crossover_rates import AdaptiveCrossoverRate
from .crossovers import BinomialCrossover
from .mutation import Rand1
from .scales import AdaptiveScaleFactor


@dataclass(frozen=True)
class JDEComponents:
    """The mutation and crossover components for Brest et al.'s jDE.

    The helper returns ``Rand1`` mutation plus binomial crossover with
    per-individual self-adapting ``F`` and ``CR`` controllers.
    """

    mutation: Rand1
    crossover: BinomialCrossover


def jde_rand_1_bin(
    *,
    initial_f: float = 0.5,
    tau_f: float = 0.1,
    f_lower: float = 0.1,
    f_upper: float = 1.0,
    initial_cr: float = 0.9,
    tau_cr: float = 0.1,
    cr_lower: float = 0.0,
    cr_upper: float = 1.0,
) -> JDEComponents:
    """Return mutation and crossover components matching Brest et al.'s jDE.

    Returns
    -------
    JDEComponents
        The ``DE/rand/1/bin`` components with jDE-style parameter adaptation.
    """

    return JDEComponents(
        mutation=Rand1(
            scale=AdaptiveScaleFactor(
                initial=initial_f,
                tau=tau_f,
                lower=f_lower,
                upper=f_upper,
            )
        ),
        crossover=BinomialCrossover(
            crossover_rate=AdaptiveCrossoverRate(
                initial=initial_cr,
                tau=tau_cr,
                lower=cr_lower,
                upper=cr_upper,
            )
        ),
    )
