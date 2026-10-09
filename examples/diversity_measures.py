"""Track population spread and movement alongside optimization progress."""

from differential_evolution import (
    AveragePairwiseDistance,
    BinomialCrossover,
    DifferentialEvolution,
    PopulationDiameter,
    Rand1,
    sphere_function,
)


def main() -> None:
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    optimizer = DifferentialEvolution(
        objective=sphere_function,
        bounds=bounds,
        population_size=20,
        mutation=Rand1(scale=0.8),
        crossover=BinomialCrossover(crossover_rate=0.9),
        max_generations=50,
        # Mix component objects and string aliases; each measure must be unique.
        diversity_measures=[
            PopulationDiameter(),
            AveragePairwiseDistance(),
            "radius",
            "coherence",
        ],
        record_snapshots=True,
        snapshot_interval=10,
        seed=123,
    )
    result = optimizer.run()
    print("Best solution:", result.x, "fitness:", result.fun)

    # Histories use canonical names, even for measures supplied as aliases.
    # Entry 0 describes initialization; later entries describe each generation.
    for name, history in result.diversity_history.items():
        print(f"{name}: initial={history[0]:.6g}, final={history[-1]:.6g}")

    # Snapshots align diversity statistics with generation, budget, and fitness.
    for snapshot in result.snapshots:
        diameter = snapshot.diversity["population_diameter"]
        coherence = snapshot.diversity["population_coherence"]
        print(f"Generation {snapshot.generation:2d}, evaluations={snapshot.evaluations:4d}: "
              f"best={snapshot.best_fitness:.6g}, diameter={diameter:.6g}, "
              f"coherence={coherence:.6g}")

    # Measures can also be called directly on a population without tracking a run.
    diameter = PopulationDiameter()(result.population, bounds)
    print("Final diameter computed directly:", diameter)


if __name__ == "__main__":
    main()
