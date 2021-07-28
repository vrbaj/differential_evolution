import numpy as np
from random import sample


class DifferentialEvolution:
    def __init__(self, cost_function, bounds, max_iterations, population_size, mutation, crossover):
        self.cost_function = cost_function
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation = mutation
        self.crossover = crossover
        self.bounds = bounds
        self.population = []

    def initialize(self):
        for _ in range(self.population_size):
            individual = []
            for bound in self.bounds:
                individual.append(np.random.uniform(bound[0], bound[1]))
            self.population.append(individual)

    def evolve(self):
        generation = 0
        while generation < self.max_iterations:
            generation += 1
            for idx, individual in enumerate(self.population):
                crossover_candidates = [idx]
                new_individual = []
                while idx in crossover_candidates:
                    crossover_candidates = sample(range(self.population_size - 1), 3)
                # print(crossover_candidates)
                r1 = self.population[crossover_candidates[0]]
                r2 = self.population[crossover_candidates[1]]
                r3 = self.population[crossover_candidates[2]]
                for dimension in range(len(individual)):
                    cr = np.random.uniform(0, 1)
                    if cr > self.crossover:
                        new_individual.append(individual[dimension])
                    else:
                        new_individual.append(r1[dimension] + self.mutation * (r2[dimension] - r3[dimension]))
                if self.cost_function(new_individual) < self.cost_function(individual):
                    self.population[idx] = new_individual

    def get_best(self):
        best_cost = np.inf
        for individual in self.population:
            cost = self.cost_function(individual)
            if cost < best_cost:
                best = individual
                best_cost = cost
        return best


def function_to_minimize(x):
    return x[0] ** 2 + x[1] ** 2


if __name__ == '__main__':
    diff_evolution = DifferentialEvolution(function_to_minimize, bounds=[[-5, 5], [-5, 5]], max_iterations=100,
                                           population_size=50,  mutation=0.5, crossover=0.7)
    diff_evolution.initialize()
    diff_evolution.evolve()
    print("the best solution: ", diff_evolution.get_best())
