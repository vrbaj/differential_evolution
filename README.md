# Differential evolution algorithm
This library aims to simplify the usage of differential evolution algorithm and enables effective comparison of various versions.
Various standard testing and population diversity measurement functions are also implemented.
Only binomial crossover is implemented so far, exponential crossover will be implemented soon.

## Implemented mutation strategies
- DE/rand/1
- DE/rand/2
- DE/best/1
- DE/best/2
- DE/current-to-best/1
- DE/current-to-best/2
- DE/current-to-rand/1

## Implemented crossover 
- binomial

## Implemented testing functions
- sphere function
- Rastrigin function
- Beale function
- Booth function
- Matyas function
- Himmelblau's function
- Bukin function
- McCormick function
- Three-hump camel function
- Ackley function
- Goldstein-Price function
- Lévi function n.13
- Easom function

## Implemented functions for population diversity estimation
- standard deviation of fitness function

## Implemented initial generation
- random
- oppositional based learning 
- tent map
- quasi-oppositional based learning