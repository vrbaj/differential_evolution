"""A penalty trades objective improvement against small constraint violations."""
from differential_evolution import DifferentialEvolution, Rand1


def violation(x):
    return max(0., 1.-x[0]-x[1])


def penalized(x):
    return sum(value*value for value in x) + 1000.*violation(x)**2


def main():
    result = DifferentialEvolution(objective=penalized, bounds=[(0.,1.)]*2,
        population_size=40, mutation=Rand1(.8), max_evaluations=4040, seed=123).run()
    print('Penalized objective:',result.fun,'constraint violation:',violation(result.x))
    # Analytic penalty minimizer has x1=x2=rho/(1+2*rho) and is infeasible.
    exact = 1000./2001.
    print('Exact penalty minimizer:',exact,'exact constrained minimizer:',.5)
    expected = 2.*exact**2 + 1000.*(1.-2.*exact)**2
    assert abs(result.fun-expected) < 1e-5


if __name__ == '__main__':
    main()
