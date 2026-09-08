from differential_evolution import BinomialCrossover, DifferentialEvolution, sphere_function


class ShrinkMutation:
    def __call__(self, context):
        target = context.population[context.target_index]
        return [value * 0.5 for value in target]


class ReplaceFirstCoordinate:
    def __call__(self, target_vector, donor_vector, rng):
        trial = list(target_vector)
        trial[0] = donor_vector[0]
        return trial


def main() -> None:
    optimizer = DifferentialEvolution(
        objective=sphere_function,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        population_size=10,
        mutation=ShrinkMutation(),
        crossover=ReplaceFirstCoordinate(),
        max_generations=10,
        seed=99,
    )
    result = optimizer.run()
    print(result.x, result.fun)


if __name__ == "__main__":
    main()
