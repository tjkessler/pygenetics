from random import randint, uniform


def calc_cdf_vals(members: list) -> list:
    ''' calc_cdf_vals: calculates the cumulative distribution of population
    member selection probabilities (i.e. members w/ higher fitness more likely
    to be chosen for next generation)

    Args:
        members (list): list of pygenetics.Member objects

    Returns:
        list: CDF values w/ parallel indices to supplied Members
    '''

    fitness_sum = sum(m._fitness_score for m in members)
    selection_probs = [m._fitness_score / fitness_sum for m in members]
    cdf_vals = []
    cumsum = 0
    for p in selection_probs:
        cumsum += p
        cdf_vals.append(cumsum)
    return cdf_vals


def call_obj_fn(params: list, obj_fn: callable, obj_fn_args: dict) -> tuple:
    ''' call_obj_fn: calls supplied objective function, evaluating using
    supplied parameters; callable in single- and multi-processed configurations

    Args:
        params (list): list of ints or floats corresponding to current bee
            parameter values
        obj_fn (callable): function to accept list of paramters, returns a
            quantitative measurement of fitness
        obj_fn_args (dict): non-tunable kwargs to pass to objective function

    Returns:
        tuple: (params, objective function return value)
    '''

    return (params, obj_fn(params, **obj_fn_args))


def determine_best_member(members: list) -> tuple:
    ''' determine_best_member: returns the fitness score, objective function
    return value, and paramters for the best-performing population member

    Args:
        members (list): list of pygenetics.Member objects

    Returns:
        tuple: (best fitness, best return value, best parameters)
    '''

    best_fitness = members[0]._fitness_score
    best_ret_val = members[0]._obj_fn_val
    best_params = members[0]._params
    for m in members[1:]:
        if m._fitness_score > best_fitness:
            best_fitness = m._fitness_score
            best_ret_val = m._obj_fn_val
            best_params = m._params
    return (best_fitness, best_ret_val, best_params)


def mutate_params(curr_params: list, params: list, p_mutation: float) -> list:
    ''' mutate_parameters: based on supplied mutation rate, mutates parameters
    supplied by the user

    Args:
        curr_params (list): current parameter values, int or float
        params (list): list of pygenetics.Parameter objects
        p_mutation (float): [0, 1], probability of mutation on any parameter

    Returns:
        list: mutated parameters
    '''

    new_params = []
    for idx, param in enumerate(curr_params):
        if uniform(0, 1) < p_mutation:
            new_params.append(params[idx].mutate(param))
        else:
            new_params.append(param)
    return new_params


def perform_crossover(params_1: list, params_2: list) -> tuple:
    ''' perform_crossover: performs a crossover of two parameter lists, using
    a random crossover point, and returns the resulting parameter lists

    Args:
        params_1 (list): first set of parameter values
        params_2 (list): second set of paramter values

    Returns:
        tuple: (crossed 1, crossed 2)
    '''

    cross_pt = randint(1, len(params_1) - 1)
    first_params = params_1[:cross_pt]
    first_params.extend(params_2[cross_pt:])
    second_params = params_2[:cross_pt]
    second_params.extend(params_1[cross_pt:])
    return (first_params, second_params)
