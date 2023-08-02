import pytest

from pygenetics import Member, Parameter, Population
from pygenetics.utils import calc_cdf_vals, call_obj_fn,\
    determine_best_member, mutate_params, perform_crossover


# member.py


def test_member():
    m = Member([0, 0, 0], 0.0)
    assert m._params == [0, 0, 0]
    assert m._obj_fn_val == 0.0
    assert m._fitness_score == 1.0
    assert m.calc_fitness(1.0) == 0.5
    assert m.calc_fitness(-1.0) == 2.0


# parameter.py


def test_param_init():
    p = Parameter(0, 10)
    assert p._min_val == 0
    assert p._max_val == 10
    assert p._dtype == int
    assert p._restrict is True
    p = Parameter(0.0, 10.0)
    assert p._dtype == float
    p = Parameter(0, 10, restrict=False)
    assert p._restrict is False
    with pytest.raises(ValueError):
        p = Parameter(0, 10.0)
    with pytest.raises(ValueError):
        p = Parameter('a', 'b')


def test_param_rand_val():
    p = Parameter(0.0, 10.0)
    for _ in range(100):
        rv = p.rand_val
        assert rv >= 0.0
        assert rv <= 10.0


def test_param_mutate():
    p = Parameter(0.0, 10.0)
    curr_val = 5.0
    mut_val = p.mutate(curr_val)
    assert curr_val != mut_val
    p = Parameter(0.0, 10.0, restrict=False)
    curr_val = 5.0
    outside_bounds = False
    for _ in range(10000):
        curr_val = p.mutate(curr_val)
        if curr_val > 10.0:
            outside_bounds = True
            break
        if curr_val < 0.0:
            outside_bounds = True
            break
    assert outside_bounds is True


# population.py


def _objective_function(params):
    return sum(params)


def _objective_function_kwargs(params, my_kwarg):
    return sum(params) + my_kwarg


def test_pop_init():
    p = Population(10, _objective_function)
    assert p._pop_size == 10
    assert p._obj_fn == _objective_function
    kwargs = {'my_kwarg': 2}
    p = Population(10, _objective_function_kwargs, kwargs)
    assert p._obj_fn_args == kwargs
    p = Population(10, _objective_function, num_processes=8)
    assert p._num_processes == 8


def test_pop_exceptions():
    with pytest.raises(ReferenceError):
        p = Population(10, None)
    with pytest.raises(ValueError):
        p = Population(1, _objective_function)
    p = Population(10, _objective_function)
    with pytest.raises(RuntimeError):
        p.initialize()
    with pytest.raises(RuntimeError):
        p.next_generation()
    p.add_param(0, 10)
    p.initialize()
    with pytest.raises(RuntimeError):
        p.add_param(0, 10)
    with pytest.raises(ValueError):
        p.next_generation(p_crossover=1.1)
    with pytest.raises(ValueError):
        p.next_generation(p_crossover=-0.1)
    with pytest.raises(ValueError):
        p.next_generation(p_mutation=1.1)
    with pytest.raises(ValueError):
        p.next_generation(p_mutation=-0.1)


def test_pop_no_stats():
    p = Population(10, _objective_function)
    assert p.best_fitness is None
    assert p.best_ret_val is None
    assert p.best_params is None
    assert p.average_fitness is None
    assert p.average_ret_val is None


def test_pop_add_parameter():
    p = Population(10, _objective_function)
    p.add_param(0, 1)
    p.add_param(2, 3)
    p.add_param(4, 5)
    assert p._params[0]._min_val == 0
    assert p._params[0]._max_val == 1
    assert p._params[1]._min_val == 2
    assert p._params[1]._max_val == 3
    assert p._params[2]._min_val == 4
    assert p._params[2]._max_val == 5
    assert p._params[0]._dtype == int
    p.add_param(0.0, 10.0)
    assert p._params[3]._dtype == float
    assert p._params[3]._restrict is True
    p.add_param(0.0, 10.0, restrict=False)
    assert p._params[4]._restrict is False


def test_pop_initialize():
    p = Population(20, _objective_function)
    p.add_param(0, 10)
    p.add_param(0, 10)
    p.initialize()
    assert len(p._members) == 20


def test_pop_kwargs():
    p = Population(10, _objective_function_kwargs, {'my_kwarg': 2})
    p.add_param(0, 0)
    p.add_param(0, 0)
    p.initialize()
    assert p.best_ret_val == 2


def test_pop_next_generation():
    p = Population(100, _objective_function)
    p.add_param(0, 10)
    p.add_param(0, 10)
    p.add_param(0, 10)
    p.initialize()
    for _ in range(10000):
        p.next_generation()
    assert p.best_fitness == 1
    assert p.best_ret_val == 0
    assert p.best_params == [0, 0, 0]


def test_pop_multiprocessing():
    p = Population(10, _objective_function, num_processes=4)
    assert p._num_processes == 4
    p.add_param(0, 10)
    p.add_param(0, 10)
    p.initialize()
    p.next_generation()


# utils.py


def test_utils_calc_cdf_vals():
    members = [
        Member([0, 0, 0], 1.0),
        Member([0, 0, 0], 2.0),
        Member([0, 0, 0], 3.0)
    ]
    cdf_vals = calc_cdf_vals(members)
    assert len(cdf_vals) == 3
    assert cdf_vals[2] == 1.0


def test_utils_call_obj_fn():
    param_vals = [2, 2, 2]
    ret_params, result = call_obj_fn(param_vals, _objective_function, {})
    assert result == 6
    assert param_vals == ret_params
    ret_params, result = call_obj_fn(param_vals, _objective_function_kwargs,
                                     {'my_kwarg': 2})
    assert result == 8
    assert param_vals == ret_params


def test_utils_determine_best_member():
    members = [
            Member([0, 0, 0], 0.0),
            Member([1, 1, 1], 1.0),
            Member([2, 2, 2], 2.0)
        ]
    b_fitness, b_ret_val, b_params = determine_best_member(members)
    assert b_fitness == 1.0
    assert b_ret_val == 0.0
    assert b_params == [0, 0, 0]


def test_utils_mutate_params():
    param_vals = [5, 5, 5]
    params = [
        Parameter(0, 10),
        Parameter(0, 10),
        Parameter(0, 10)
    ]
    new_vals = mutate_params(param_vals, params, 0.0)
    assert param_vals == new_vals
    new_vals = mutate_params(param_vals, params, 1.0)
    assert param_vals != new_vals


def test_utils_perform_crossover():
    p1 = [0, 0]
    p2 = [1, 1]
    p1_new, p2_new = perform_crossover(p1, p2)
    assert p1_new == [0, 1]
    assert p2_new == [1, 0]
    p1 = [0, 0, 0, 0]
    p2 = [1, 1, 1, 1]
    p1_new, p2_new = perform_crossover(p1, p2)
    possible_combos = [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    assert p1_new in possible_combos
    assert p2_new in possible_combos
