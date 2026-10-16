"""Ordered external evaluation without a runtime dependency on worker libraries."""
from concurrent.futures import ThreadPoolExecutor

from differential_evolution import LSHADE, sphere_function


def main():
    # For CPU-heavy Python objectives use ProcessPoolExecutor in a script with
    # this main guard. Threads suit I/O or native simulations releasing the GIL.
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = LSHADE(
            objective=sphere_function,
            batch_objective=lambda candidates: executor.map(sphere_function, candidates),
            bounds=[(-5., 5.)] * 3,
            population_size=54,
            max_evaluations=1000,
            seed=123,
        ).run()
    print(result.fun, result.nfev)


if __name__ == "__main__":
    main()
