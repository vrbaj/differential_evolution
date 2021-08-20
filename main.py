from random import sample
from matplotlib import pyplot as plt
import numpy as np
import population_initialization


class DifferentialEvolution:
    def __init__(self, cost_function, bounds, max_iterations,
                 population_size, mutation, crossover, strategy, population_initialization_algorithm):
        self.cost_function = cost_function
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation = mutation
        self.crossover = crossover
        self.bounds = bounds
        self.population = []
        self.strategy = strategy
        self.generation = 0
        self.generation_best_individual = None
        self.generation_best_individual_idx = None
        self.best_individual_history = []
        self.generation_fitness = []
        self.initial_population = population_initialization_algorithm

    def initialize(self):
        if self.initial_population == "random":
            self.population = population_initialization.random_initialization(self.population_size, self.bounds)
        elif self.initial_population == "obl":
            self.population = population_initialization.obl_initialization(self.population_size,
                                                                           self.bounds, self.cost_function)
        elif self.initial_population == "tent":
            self.population = population_initialization.tent_initialization(self.population_size, self.bounds)
        elif self.initial_population == "qobl":
            self.population = population_initialization.qobl_initialization(self.population_size,
                                                                            self.bounds, self.cost_function)
        elif self.initial_population == "sobol":
            # # TODO various polynomials, various m numbers
            self.population = population_initialization.sobol_initialization(self.population_size, self.bounds)

    def evolve(self):
        self.generation = 0
        while self.generation < self.max_iterations:
            self.generation_fitness = []
            self.generation += 1
            self.get_best()
            self.best_individual_history.append(self.generation_best_individual)
            for idx, individual in enumerate(self.population):
                crossover_candidates = self.generate_crossover_candidates(idx)
                new_individual = []
                random_dimension = np.random.randint(0, len(individual))
                for dimension in range(len(individual)):
                    cr = np.random.uniform(0, 1)
                    if cr > self.crossover or random_dimension != dimension:
                        new_individual.append(individual[dimension])
                    else:
                        # check boundaries, if violation, random generate new one
                        new_individual.append(self.mutate(crossover_candidates, dimension))

                if self.cost_function(new_individual) < self.cost_function(individual):
                    self.population[idx] = new_individual
                self.generation_fitness.append(self.cost_function(self.population[idx]))
                # measure diversity here?

    def generate_crossover_candidates(self, idx):
        crossover_candidates = [idx]
        while idx in crossover_candidates:
            crossover_candidates = sample(range(self.population_size - 1), 6)
        if self.strategy == "DE/rand/1":
            return crossover_candidates[:3]
        elif self.strategy == "DE/rand/2":
            return crossover_candidates
        elif self.strategy == "DE/best/1":
            crossover_candidates[0] = self.generation_best_individual_idx
            return crossover_candidates[:3]
        elif self.strategy == "DE/best/2":
            crossover_candidates[0] = self.generation_best_individual_idx
            return crossover_candidates
        elif self.strategy == "DE/current-to-best/1":
            crossover_candidates[0] = self.generation_best_individual_idx
            crossover_candidates[1] = idx
            return crossover_candidates[:4]
        elif self.strategy == "DE/current-to-best/2":
            crossover_candidates[0] = self.generation_best_individual_idx
            crossover_candidates[1] = idx
            return crossover_candidates
        elif self.strategy == "DE/current-to-rand/1":
            crossover_candidates[0] = idx
            return crossover_candidates[:4]
        elif self.strategy == "DE/current-to-rand/2":
            crossover_candidates[0] = idx
            return crossover_candidates[:6]

    def mutate(self, crossover_candidates, dimension):
        if self.strategy == "DE/rand/1" or self.strategy == "DE/best/1":
            return self.population[crossover_candidates[0]][dimension] + \
                   self.mutation[0] * (self.population[crossover_candidates[1]][dimension] -
                                       self.population[crossover_candidates[2]][dimension])
        elif self.strategy == "DE/rand/2" or self.strategy == "DE/best/2":
            return self.population[crossover_candidates[0]][dimension] + \
                   self.mutation[0] * (self.population[crossover_candidates[1]][dimension] -
                                       self.population[crossover_candidates[2]][dimension]) + \
                   self.mutation[0] * (self.population[crossover_candidates[3]][dimension] -
                                       self.population[crossover_candidates[4]][dimension])
        elif self.strategy == "DE/current-to-best/1":
            return self.population[crossover_candidates[1]][dimension] + \
                   self.mutation[0] * (self.population[crossover_candidates[0]][dimension] -
                                       self.population[crossover_candidates[1]][dimension]) +\
                   self.mutation[1] * (self.population[crossover_candidates[2]][dimension] -
                                       self.population[crossover_candidates[3]][dimension])
        elif self.strategy == "DE/current-to-best/2":
            return self.population[crossover_candidates[1]][dimension] + \
                   self.mutation[0] * (self.population[crossover_candidates[0]][dimension] -
                                       self.population[crossover_candidates[1]][dimension]) + \
                   self.mutation[1] * (self.population[crossover_candidates[2]][dimension] -
                                       self.population[crossover_candidates[3]][dimension]) + \
                   self.mutation[1] * (self.population[crossover_candidates[4]][dimension] -
                                       self.population[crossover_candidates[5]][dimension])
        elif self.strategy == "DE/current-to-rand/1":
            return self.population[crossover_candidates[0]][dimension] + \
                   self.mutation[0] * (self.population[crossover_candidates[1]][dimension] -
                                       self.population[crossover_candidates[0]][dimension]) + \
                   self.mutation[1] * (self.population[crossover_candidates[2]][dimension] -
                                       self.population[crossover_candidates[3]][dimension])
        elif self.strategy == "DE/current-to-rand/2":
            return self.population[crossover_candidates[0]][dimension] + \
                   self.mutation[0] * (self.population[crossover_candidates[1]][dimension] -
                                       self.population[crossover_candidates[0]][dimension]) + \
                   self.mutation[1] * (self.population[crossover_candidates[2]][dimension] -
                                       self.population[crossover_candidates[3]][dimension]) + \
                   self.mutation[1] * (self.population[crossover_candidates[4]][dimension] -
                                       self.population[crossover_candidates[5]][dimension])

    def get_best(self):
        best_cost = np.inf
        best = np.inf
        best_idx = np.inf
        for idx, individual in enumerate(self.population):
            cost = self.cost_function(individual)
            if cost < best_cost:
                best = individual
                best_cost = cost
                best_idx = idx
        self.generation_best_individual = best
        self.generation_best_individual_idx = best_idx
        return best

    def filter_history(self, dimension):
        solutions = [solution[dimension] for solution in self.best_individual_history]
        return solutions

    def measure_diversity(self, measure):
        if measure == "std-fitness":
            return np.std(self.generation_fitness)


def function_to_minimize(x):
    return x[0] ** 2 + x[1] ** 2


if __name__ == '__main__':
    from testing_functions import sphere_function as sphere_function
    diff_evolution = DifferentialEvolution(sphere_function, bounds=[[-5, 5.1], [-5.1, 5]], max_iterations=100,
                                           population_size=17,  mutation=[0.7, 0.7], crossover=0.7,
                                           strategy="DE/current-to-rand/2", population_initialization_algorithm="sobol")
    diff_evolution.initialize()
    diff_evolution.evolve()
    print("the best solution: ", diff_evolution.get_best())
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Error")
    optimal_value = [0, 0]
    ax1.plot(np.log10(np.abs(np.asarray(diff_evolution.filter_history(0)) - optimal_value[0])))
    ax2.plot(np.log10(np.abs(np.asarray(diff_evolution.filter_history(1)) - optimal_value[1])))
    plt.show()
