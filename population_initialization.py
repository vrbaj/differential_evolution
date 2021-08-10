import numpy as np


def random_initialization(population_size, bounds):
    population = []
    for _ in range(population_size):
        individual = []
        for bound in bounds:
            individual.append(np.random.uniform(bound[0], bound[1]))
        population.append(individual)
    return population


def tent_initialization(population_size, bounds):
    x = np.random.uniform(0, 1)
    population = []
    for _ in range(population_size):
        individual = []
        for bound in bounds:
            if x < 0.5:
                x = 2 * x
            else:
                x = 2 * (1 - x)
            individual.append((bound[1] - bound[0]) * x + bound[0])
        population.append(individual)
    return population
