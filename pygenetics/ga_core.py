#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ga_core.py (0.5.2)
#
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from random import choice, randint, uniform
from operator import add, sub
from multiprocessing import Pool

# Third party open source packages
from numpy import logspace, random as nrandom

# PyGenetics library imports
from pygenetics.selection_functions import minimize_best_n

# Supported parameter types and functions to generate random initial values
SUPPORTED_DTYPES = {
    int: randint,
    float: uniform
}


class Parameter:

    def __init__(self, name, min_val, max_val):
        '''Parameter object

        Args:
            name (str): name of the parameter
            min_val (int or float): minimum allowed value for the parameter
            max_val (int or float): maximum allowed value for the parameter
        '''

        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        if type(min_val) != type(max_val):
            raise ValueError('Supplied min_val is not the same type as\
                             supplied max_val: {}, {}'.format(
                                 type(min_val),
                                 type(max_val))
                             )
        self.dtype = type(min_val + max_val)
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError('Unsupported data type: use {}'
                             .format(SUPPORTED_DTYPES))


class Member:

    def __init__(self, parameters, cost_fn_val):
        '''Member object

        Args:
            parameters (dictionary): dictionary of parameter names and values
            cost_fn_val (float): value returned by cost function using params
        '''

        self.parameters = parameters
        self.cost_fn_val = cost_fn_val
        self.fitness_score = self.__calc_fitness_score(cost_fn_val)

    @staticmethod
    def __calc_fitness_score(cost_fn_val):
        '''Derives fitness score from cost function return value

        Args:
            cost_fn_val (float): value returned by cost function

        Returns:
            float: standardized fitness score
        '''

        if cost_fn_val >= 0:
            return 1 / (cost_fn_val + 1)
        else:
            return 1 + abs(cost_fn_val)


class Population:

    def __init__(self, size, cost_fn, cost_fn_args=None, num_processes=1,
                 select_fn=minimize_best_n):
        '''Population object: tunes specified parameters by measuring the
        performance of population members

        Args:
            size (int): number of Members in the population
            cost_fn (callable): function used to evaluate member fitness
            cost_fn_args (iterable or dict): additional user-specified
                arguments to pass to cost_fn
            num_processes (int): number of concurrent processes to run for
                Member generation/evaluation
            select_fn (callable): function used to sort members for parent
                based on member fitness score
        '''

        if size <= 0:
            raise ValueError('Population *size* cannot be <= 0')
        self.__pop_size = size
        if not callable(cost_fn):
            raise ValueError('Supplied *cost_fn* is not callable')
        self.__cost_fn = cost_fn
        if not callable(select_fn):
            raise ValueError('Supplied *select_fn* is not callable')
        self.__cost_fn_args = cost_fn_args
        self.__select_fn = select_fn
        self.__parameters = []
        self.__members = []
        self.__num_processes = num_processes

        self.__best_fitness = None
        self.__best_cost_fn_val = None
        self.__best_parameters = None

    def __len__(self):
        '''len(Population) == population size'''

        return self.__pop_size

    @property
    def fitness(self):
        '''Population fitness == average member fitness score'''

        if len(self.__members) != 0:
            if self.__num_processes > 1:
                members = [m.get() for m in self.__members]
            else:
                members = self.__members
            return sum(m.fitness_score for m in members) / len(members)
        else:
            return None

    @property
    def best_fitness(self):
        '''Fitness score of best performing population member so far'''

        return self.__best_fitness

    @property
    def ave_cost_fn_val(self):
        '''Returns average cost function return value for all members'''

        if len(self.__members) != 0:
            if self.__num_processes > 1:
                members = [m.get() for m in self.__members]
            else:
                members = self.__members
            return sum(m.cost_fn_val for m in members) / len(members)
        else:
            return None

    @property
    def best_cost_fn_val(self):

        return self.__best_cost_fn_val

    @property
    def parameters(self):
        '''Population parameter vals == average member parameter vals'''

        if len(self.__members) != 0:
            if self.__num_processes > 1:
                members = [m.get() for m in self.__members]
            else:
                members = self.__members
            params = {}
            for p in self.__parameters:
                params[p.name] = sum(
                    m.parameters[p.name] for m in members
                ) / len(members)
            return params
        else:
            return None

    @property
    def best_parameters(self):
        '''Parameter values of best performing population member so far'''

        return self.__best_parameters

    @property
    def members(self):
        '''Returns Member objects of population'''

        if self.__num_processes > 1:
            return [m.get() for m in self.__members]
        else:
            return self.__members

    def add_parameter(self, name, min_val, max_val):
        '''Adds a paramber to the Population

        Args:
            name (str): name of the parameter
            min_val (int or float): minimum value for the parameter
            max_val (int or float): maximum value for the parameter
        '''

        self.__parameters.append(Parameter(name, min_val, max_val))

    def generate_population(self):
        '''Generates self.__pop_size Members with randomly initialized values
        for each parameter added with add_parameter(), evaluates their fitness
        '''

        if self.__num_processes > 1:
            process_pool = Pool(processes=self.__num_processes)
        self.__members = []

        for _ in range(self.__pop_size):
            feed_dict = {}
            for param in self.__parameters:
                feed_dict[param.name] = self.__random_param_val(
                    param.min_val,
                    param.max_val,
                    param.dtype
                )
            if self.__num_processes > 1:
                self.__members.append(process_pool.apply_async(
                    self._start_process,
                    [self.__cost_fn, feed_dict, self.__cost_fn_args])
                )
            else:
                self.__members.append(
                    Member(
                        feed_dict,
                        self.__cost_fn(feed_dict, self.__cost_fn_args)
                    )
                )

        if self.__num_processes > 1:
            process_pool.close()
            process_pool.join()

        self.__determine_best_member()

    def next_generation(self, mut_rate=0, max_mut_amt=0, log_base=10):
        '''Generates the next population from a previously evaluated generation

        Args:
            mut_rate (float): mutation rate for new members (0.0 - 1.0)
            max_mut_amt (float): how much the member is allowed to mutate
                (0.0 - 1.0, proportion change of mutated parameter)
            log_base (int): the higher this number, the more likely the first
                Members (chosen with supplied selection function) are chosen
                as parents for the next generation
        '''

        if self.__num_processes > 1:
            process_pool = Pool(processes=self.__num_processes)
            members = [m.get() for m in self.__members]
        else:
            members = self.__members

        if len(members) == 0:
            raise Exception(
                'Generation 0 not found: use generate_population() first'
            )

        selected_members = self.__select_fn(members)
        reproduction_probs = list(reversed(logspace(0.0, 1.0,
                                  num=len(selected_members), base=log_base)))
        reproduction_probs = reproduction_probs / sum(reproduction_probs)

        self.__members = []

        for _ in range(self.__pop_size):
            parent_1 = nrandom.choice(selected_members, p=reproduction_probs)
            parent_2 = nrandom.choice(selected_members, p=reproduction_probs)

            feed_dict = {}
            for param in self.__parameters:
                which_parent = uniform(0, 1)
                if which_parent < 0.5:
                    feed_dict[param.name] = parent_1.parameters[param.name]
                else:
                    feed_dict[param.name] = parent_2.parameters[param.name]
                feed_dict[param.name] = self.__mutate_parameter(
                    feed_dict[param.name], param, mut_rate, max_mut_amt
                )

            if self.__num_processes > 1:
                self.__members.append(process_pool.apply_async(
                    self._start_process,
                    [self.__cost_fn, feed_dict, self.__cost_fn_args])
                )
            else:
                self.__members.append(
                    Member(
                        feed_dict,
                        self.__cost_fn(feed_dict, self.__cost_fn_args)
                    )
                )

        if self.__num_processes > 1:
            process_pool.close()
            process_pool.join()

        self.__determine_best_member()

    @staticmethod
    def _start_process(cost_fn, feed_dict, cost_fn_args):
        '''Static method: starts a process to generate (evaluate) a new Member

        Args:
            cost_fn (function): cost function supplied to the Population
            feed_dict (dictionary): dictionary of Parameter objects
            cost_fn_args (iterable or dict): user-supplied args for cost_fn

        Returns:
            Evaluated Member object
        '''

        return Member(feed_dict, cost_fn(feed_dict, cost_fn_args))

    @staticmethod
    def __random_param_val(min_val, max_val, dtype):
        '''
        Private, static method: returns a random value

        Args:
            min_val (int or float): minimum value for random value
            max_Val (int or float): maximum value for random vlaue
            dtype (type): type of random value, int or float

        Returns:
            int or float: random value
        '''

        return SUPPORTED_DTYPES[dtype](min_val, max_val)

    @staticmethod
    def __mutate_parameter(value, param, mut_rate, max_mut_amt):
        '''Private, static method: mutates parameter

        Args:
            value (int or float): current value for Member's parameter
            param (Parameter): parameter object
            mut_rate (float): mutation rate of the value
            max_mut_amt (float): maximum mutation amount of the value

        Returns:
            int or float: mutated value
        '''

        if uniform(0, 1) < mut_rate:
            mut_amt = uniform(0, max_mut_amt)
            op = choice((add, sub))
            new_val = op(value, param.dtype(
                (param.max_val - param.min_val) * mut_amt
            ))
            if new_val > param.max_val:
                return param.max_val
            elif new_val < param.min_val:
                return param.min_val
            else:
                return new_val
        else:
            return value

    def __determine_best_member(self):
        '''Private method: determines if any current population members have a
        fitness score better than the current best
        '''

        if self.__num_processes > 1:
            members = [m.get() for m in self.__members]
        else:
            members = self.__members

        if self.__best_fitness is None:
            self.__best_fitness = members[0].fitness_score
            self.__best_cost_fn_val = members[0].cost_fn_val
            self.__best_parameters = {}
            for p in self.__parameters:
                self.__best_parameters[p.name] = members[0].parameters[p.name]

        for m_id, member in enumerate(members):
            if member.fitness_score > self.__best_fitness:
                self.__best_fitness = member.fitness_score
                self.__best_cost_fn_val = member.cost_fn_val
                self.__best_parameters = {}
                for p in self.__parameters:
                    self.__best_parameters[p.name] = member.parameters[p.name]
