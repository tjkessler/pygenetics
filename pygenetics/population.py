from bisect import bisect
from multiprocessing import Pool
from random import random, uniform
from typing import Callable, Union
from warnings import warn

from pygenetics.member import Member
from pygenetics.parameter import Parameter
from pygenetics.utils import calc_cdf_vals, call_obj_fn,\
    determine_best_member, mutate_params, perform_crossover


class Population:

    def __init__(self, pop_size: int,
                 objective_fn: Callable[[list], Union[int, float]],
                 obj_fn_args: dict = {}, num_processes: int = 1):
        ''' Population object: initializes a genetic algorithm population with
        user-specified population size, objective function and any additional
        immutable objects to pass to the objective function

        Args:
            pop_size (int): size of the population
            objective_fn (callable): function for optimization
            obj_fn_args (dict): immutable arguments to pass to objective_fn
            num_processes (int): number of concurrent processes to utilize
        '''

        if not callable(objective_fn):
            raise ReferenceError('Supplied objective function is not callable')

        if pop_size <= 1:
            raise ValueError('`pop_size` must be >= 2: {}'.format(pop_size))

        self._obj_fn = objective_fn
        self._obj_fn_args = obj_fn_args
        self._pop_size = pop_size
        self._num_processes = num_processes
        self._members = []
        self._params = []

    @property
    def best_fitness(self) -> float:
        ''' Returns fitness score from best-performing member '''

        if len(self._members) == 0:
            return None
        return determine_best_member(self._members)[0]

    @property
    def best_ret_val(self) -> Union[int, float]:
        ''' Returns objective_fn return value from best-performing member '''

        if len(self._members) == 0:
            return None
        return determine_best_member(self._members)[1]

    @property
    def best_params(self) -> list:
        ''' Returns parameters from best-performing member '''

        if len(self._members) == 0:
            return None
        return determine_best_member(self._members)[2]

    @property
    def average_fitness(self) -> float:
        ''' Returns average fitness score for population '''

        if len(self._members) == 0:
            return None
        return (sum(m._fitness_score for m in self._members) /
                len(self._members))

    @property
    def average_ret_val(self) -> float:
        ''' Returns average objective_fn return value for population '''

        if len(self._members) == 0:
            return None
        return (sum(m._obj_fn_val for m in self._members) / len(self._members))

    def add_param(self, min_val: Union[int, float], max_val: Union[int, float],
                  restrict: bool = True):
        ''' Population.add_param: adds a parameter to be processed by the user-
        supplied objective function

        Args:
            min_val (int, float): minimum value allowed for the parameter's
                initialization
            max_val (int, float): maximum value allowed for the parameter's
                initialization
            restrict (bool): if `True`, parameter mutations must be within
                [min_val, max_val], `False` allows out-of-bounds mutation
        '''

        if len(self._members) > 0:
            raise RuntimeError(
                'Cannot add another parameter after population is created'
            )

        self._params.append(Parameter(min_val, max_val, restrict))

    def initialize(self):
        ''' Population.initialize: generates random paramter values for each
        population member, evaluates fitness for each
        '''

        if len(self._params) == 0:
            raise RuntimeError(
                'No parameters have been added, cannot initialize'
            )

        if len(self._members) > 0:
            warn('initialize() called again: overwriting current population',
                 RuntimeWarning)

        self._members = []

        if self._num_processes > 1:
            mp_pool = Pool(processes=self._num_processes)
        member_results = []

        for _ in range(self._pop_size):

            params = [p.rand_val for p in self._params]
            if self._num_processes > 1:
                member_results.append(mp_pool.apply_async(
                    call_obj_fn, [params, self._obj_fn, self._obj_fn_args]
                ))
            else:
                member_results.append(call_obj_fn(
                    params, self._obj_fn, self._obj_fn_args
                ))

        if self._num_processes > 1:

            mp_pool.close()
            mp_pool.join()
            member_results = [r.get() for r in member_results]

        for result in member_results:

            self._members.append(Member(result[0], result[1]))

    def next_generation(self, p_crossover: float = 0.5,
                        p_mutation: float = 0.01):
        ''' Population.next_generation: generates the next generation of
        population members; members are chosen proportionally based on their
        fitness, where a higher fitness results in a higher chance to be
        chosen for the next generation; included is a chance for crossover
        between two members and a chance for mutation within a member's
        parameter/chromosome

        Args:
            p_crossover (float): [0, 1], probability a member is subjected to
                crossover after selection for the next generation; default
                value of 0.5 (50%)
            p_mutation (float): [0, 1], probability a chosen member's
                parameters are subject to mutation; default value of 0.01 (1%)
        '''

        if len(self._members) == 0:
            raise RuntimeError(
                'initilize() must be called before next_generation()'
            )

        if p_crossover < 0 or p_crossover > 1:
            raise ValueError('`p_crossover` must be within [0, 1]: {}'.format(
                p_crossover
            ))

        if p_mutation < 0 or p_mutation > 1:
            raise ValueError('`p_mutation` must be within [0, 1]: {}'.format(
                p_mutation
            ))

        new_param_vals = []
        cdf_vals = calc_cdf_vals(self._members)

        while len(new_param_vals) < self._pop_size:

            chosen_member = self._members[bisect(cdf_vals, random())]

            if len(self._params) > 1 and uniform(0, 1) < p_crossover:

                mate = self._members[bisect(cdf_vals, random())]
                while chosen_member == mate:
                    mate = self._members[bisect(cdf_vals, random())]
                new_params_1, new_params_2 = perform_crossover(
                    chosen_member._params, mate._params
                )
                new_param_vals.append(mutate_params(
                    new_params_1, self._params, p_mutation
                ))
                new_param_vals.append(mutate_params(
                    new_params_2, self._params, p_mutation
                ))

            else:

                new_param_vals.append(mutate_params(
                    chosen_member._params, self._params, p_mutation
                ))

        if self._num_processes > 1:
            mp_pool = Pool(processes=self._num_processes)
        new_member_results = []

        for vals in new_param_vals:

            if self._num_processes > 1:
                new_member_results.append(mp_pool.apply_async(
                    call_obj_fn, [vals, self._obj_fn, self._obj_fn_args]
                ))
            else:
                new_member_results.append(call_obj_fn(
                    vals, self._obj_fn, self._obj_fn_args
                ))

        if self._num_processes > 1:

            mp_pool.close()
            mp_pool.join()
            new_member_results = [r.get() for r in new_member_results]

        self._members = []
        for result in new_member_results:

            self._members.append(Member(result[0], result[1]))
