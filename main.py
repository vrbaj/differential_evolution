import numpy as np
from random import sample
from matplotlib import pyplot as plt
import numpy as np


class DifferentialEvolution:
    def __init__(self, cost_function, bounds, max_iterations, population_size, mutation, crossover, strategy):
        self.cost_function = cost_function
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation = mutation
        self.crossover = crossover
        self.bounds = bounds
        self.population = []
        self.strategy = strategy
        self.generation = 0
        self.best_individual = None
        self.best_individual_idx = None
        self.best_individual_history = []

    def initialize(self):
        for _ in range(self.population_size):
            individual = []
            for bound in self.bounds:
                individual.append(np.random.uniform(bound[0], bound[1]))
            self.population.append(individual)

    def evolve(self):
        self.generation = 0
        while self.generation < self.max_iterations:
            self.generation += 1
            self.get_best()
            self.best_individual_history.append(self.best_individual)
            for idx, individual in enumerate(self.population):
                crossover_candidates = self.generate_crossover_candidates(idx)
                new_individual = []
                for dimension in range(len(individual)):
                    cr = np.random.uniform(0, 1)
                    if cr > self.crossover:
                        new_individual.append(individual[dimension])
                    else:
                        new_individual.append(self.mutate(crossover_candidates, dimension))
                if self.cost_function(new_individual) < self.cost_function(individual):
                    self.population[idx] = new_individual

    def generate_crossover_candidates(self, idx):
        crossover_candidates = [idx]
        while idx in crossover_candidates:
            crossover_candidates = sample(range(self.population_size - 1), 5)
        if self.strategy == "DE/rand/1":
            return crossover_candidates[:3]
        elif self.strategy == "DE/rand/2":
            return crossover_candidates
        elif self.strategy == "DE/best/1":
            crossover_candidates[0] = self.best_individual_idx
            return crossover_candidates[:3]

    def mutate(self, crossover_candidates, dimension):
        if self.strategy == "DE/rand/1" or self.strategy == "DE/best/1":
            return self.population[crossover_candidates[0]][dimension] + \
                   self.mutation * (self.population[crossover_candidates[1]][dimension] -
                                    self.population[crossover_candidates[2]][dimension])
        elif self.strategy == "DE/rand/2":
            return self.population[crossover_candidates[0]][dimension] + \
                   self.mutation * (self.population[crossover_candidates[1]][dimension] -
                                    self.population[crossover_candidates[2]][dimension]) + \
                   self.mutation * (self.population[crossover_candidates[3]][dimension] -
                                    self.population[crossover_candidates[4]][dimension])

    def get_best(self):
        best_cost = np.inf
        for idx, individual in enumerate(self.population):
            cost = self.cost_function(individual)
            if cost < best_cost:
                best = individual
                best_cost = cost
                best_idx = idx
        self.best_individual = best
        self.best_individual_idx = best_idx
        return best


def function_to_minimize(x):
    return x[0] ** 2 + x[1] ** 2


if __name__ == '__main__':
    diff_evolution = DifferentialEvolution(function_to_minimize, bounds=[[-5, 5], [-5, 5]], max_iterations=100,
                                           population_size=50,  mutation=0.5, crossover=0.7, strategy="DE/best/1")
    diff_evolution.initialize()
    diff_evolution.evolve()
    print("the best solution: ", diff_evolution.get_best())
    print(diff_evolution.best_individual_history)
    plt.plot(np.log10(diff_evolution.best_individual_history))
    plt.show()