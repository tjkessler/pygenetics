from pygenetics import Population

def minimize_integers(integers):

    return sum(integers)

pop = Population(10, minimize_integers)
pop.add_param(0, 10)
pop.add_param(0, 10)
pop.add_param(0, 10)
pop.initialize()
for _ in range(10):
    pop.next_generation()
    print('Average fitness: {}'.format(pop.average_fitness))
    print('Average obj. fn. return value: {}'.format(pop.average_ret_val))
    print('Best fitness score: {}'.format(pop.best_fitness))
    print('Best obj. fn. return value: {}'.format(pop.best_ret_val))
    print('Best parameters: {}\n'.format(pop.best_params))
