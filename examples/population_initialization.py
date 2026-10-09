"""Choose an initializer and inspect its population before optimization."""

from differential_evolution import (
    BinomialCrossover,
    DifferentialEvolution,
    OppositionInitializer,
    QuasiOppositionInitializer,
    Rand1,
    RandomInitializer,
    SobolInitializer,
    TentInitializer,
    sphere_function,
)


def main() -> None:
    initializers = (
        ("Random (the default)", RandomInitializer()),
        ("Tent map", TentInitializer()),
        ("Opposition", OppositionInitializer()),
        ("Quasi-opposition", QuasiOppositionInitializer()),
        ("Sobol with seeded digital shift", SobolInitializer()),
        ("Sobol without digital shift", SobolInitializer(scramble=False)),
    )

    for name, initializer in initializers:
        # Construct a fresh optimizer for each run; seed controls its RNG.
        optimizer = DifferentialEvolution(
            objective=sphere_function,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            population_size=16,
            initializer=initializer,
            mutation=Rand1(scale=0.8),
            crossover=BinomialCrossover(crossover_rate=0.9),
            max_evaluations=416,
            seed=123,
        )

        # initialize() lets you inspect starting points and their fitness.
        # run() then continues from this population without initializing again.
        optimizer.initialize()
        print(f"\n{name}")
        print("  First initial vector:", optimizer.population[0])
        print("  Initial best fitness:", min(optimizer.fitness))
        print("  Initialization evaluations:", optimizer.nfev)

        result = optimizer.run()
        print("  Final best fitness:", result.fun)
        print("  Total evaluations:", result.nfev, "generations:", result.nit)


if __name__ == "__main__":
    main()
