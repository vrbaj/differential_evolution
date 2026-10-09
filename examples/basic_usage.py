from differential_evolution import (
    Best1,
    BinomialCrossover,
    DifferentialEvolution,
    ExponentialCrossover,
    Rand1,
    sphere_function,
)


def main() -> None:
    optimizer = DifferentialEvolution(
        objective=sphere_function,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        population_size=20,
        mutation=Rand1(scale=0.8),
        crossover=BinomialCrossover(crossover_rate=0.9),
        max_generations=50,
        seed=123,
    )
    result = optimizer.run()
    print("rand1/bin result:", result.x, result.fun)

    optimizer_alt = DifferentialEvolution(
        objective=sphere_function,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        population_size=20,
        mutation=Best1(scale=0.5),
        crossover=ExponentialCrossover(crossover_rate=0.8),
        max_generations=50,
        seed=123,
    )
    result_alt = optimizer_alt.run()
    print("best1/exp result:", result_alt.x, result_alt.fun)


if __name__ == "__main__":
    main()
