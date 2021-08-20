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


def obl_initialization(population_size, bounds, cost_function):
    # oppositional based learning
    population_ext = []
    for _ in range(population_size):
        individual = []
        opposite_individual = []
        for bound in bounds:
            x_i = np.random.uniform(bound[0], bound[1])
            individual.append(x_i)
            opposite_individual.append(min(bound) + max(bound) - x_i)

        population_ext.append(individual)
        population_ext.append(opposite_individual)
    population_ext.sort(key=cost_function)
    population = population_ext[:population_size]
    return population


def qobl_initialization(population_size, bounds, cost_function):
    # quasi-oppositional differential evolution
    population_ext = []
    for _ in range(population_size):
        individual = []
        quasi_opposite_individual = []
        for bound in bounds:
            x_i = np.random.uniform(bound[0], bound[1])
            individual.append(x_i)
            opposite_individual = min(bound) + max(bound) - x_i
            m = (min(bound) + max(bound)) / 2
            if x_i < m:
                quasi_opposite_individual.append(m + (opposite_individual - m) * np.random.uniform(0, 1))
            else:
                quasi_opposite_individual.append(
                    opposite_individual + (m - opposite_individual) * np.random.uniform(0, 1))

        population_ext.append(individual)
        population_ext.append(quasi_opposite_individual)
    population_ext.sort(key=cost_function)
    population = population_ext[:population_size]
    return population


def sobol_initialization(population_size, bounds):
    # TODO various polynomials, various m numbers
    polynomial_coefficients = [0, 1]
    # generate m_i
    m = [1, 3, 7]
    for i in range(3, population_size + 1):
        m_i = 2 * polynomial_coefficients[0] * m[i - 1] ^ 2 ** 2 * polynomial_coefficients[1] * m[i - 2] ^ \
              2 ** 3 * m[i - 3] ^ m[i - 3]
        m.append(m_i)
    # generate v_i
    v = []
    for idx, m_i in enumerate(m):
        bin_repr = bin(m_i)[2:]
        while len(bin_repr) < idx + 1:
            bin_repr = "0" + bin_repr
        v.append(bin_repr)
    # add zeros to v_i and 0.
    w = []
    for v_i in v:
        v_i = "0." + v_i
        w.append(v_i)
    zero_string = "0." + (len(w[0]) - 2) * "0"
    population = []
    for individual in range(1, population_size + 1):
        bv = []
        for idx, bit in enumerate(reversed(bin(individual)[2:])):
            if bit == "0":
                bv.append(zero_string)
            else:
                bv.append(w[idx])
        normalized_individual = [my_xor(bv)] * len(bounds)
        for idx, dimension in enumerate(normalized_individual):
            normalized_individual[idx] = bounds[idx][0] + normalized_individual[idx] * (bounds[idx][1] - bounds[idx][0])
        population.append(normalized_individual)
    return population


def my_xor(bv):
    binary_reps = []
    max_len = max(len(x) for x in bv)
    bv = [x + (max_len - len(x)) * "0" for x in bv]
    for item in bv:
        binary_reps.append(int(item[2:], 2))
    # xor
    for index in range(1, len(bv)):
        new_bv = []
        for letter_idx, letter in enumerate(bv[index]):
            if bv[index][letter_idx] == ".":
                new_bv.append(".")
            else:
                if bv[index][letter_idx] == bv[index - 1][letter_idx]:
                    new_bv.append("0")
                else:
                    new_bv.append("1")
        bv[index] = new_bv
    individual = 0
    for exponent, item in enumerate("".join(bv[-1]).replace(".", "")):
        individual = individual + int(item) * 1 / (2 ** exponent)
    return individual
