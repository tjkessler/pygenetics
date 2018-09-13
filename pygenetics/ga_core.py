#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ga_core.py (0.4.1)
#
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Third party open source packages
import random
import numpy as np
import types
from operator import add, sub
import multiprocessing as mp

# PyGenetics library imports
from pygenetics import selection_functions

# Supported parameter types and functions to generate random initial values
SUPPORTED_DTYPES = {
    int: random.randint,
    float: random.uniform
}


class Parameter:
    '''
    Parameter object; contains *name* of the parameter to tune, its minimum
    value *min_val* and its maximum value *max_val*
    '''

    def __init__(self, name, min_val, max_val):
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
    '''
    Member object: contains a *feed_dict* (dictionary of parameter names and
    unique values for the member) and a *fitness_score* calcutated using the
    population's cost function
    '''

    def __init__(self, feed_dict, fitness_score):
        self.feed_dict = feed_dict
        self.fitness_score = fitness_score

    @property
    def param_vals(self):
        return self.feed_dict


class Population:
    '''
    Population object: tunes specified parameters by measuring the performance
    of population members
    '''

    def __init__(self, size, cost_fn, cost_fn_args=None, num_processes=4,
                 select_fn=selection_functions.minimize_best_n):
        '''
        Initialize the population

        *size*          -   population size (number of members)
        *cost_fn*       -   function used to evaluate population member fitness
        *cost_fn_args*  -   Additional arbitrary arguments for supplied cost
                            function; passed to cost function after feed_dict
        *num_processes* -   if > 0, will utilize multiprocessing for member
                            generation
        *select_fn*     -   function used to order population members for next
                            generation
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

    def __len__(self):
        '''
        len(Population) == population size
        '''

        return self.__pop_size

    @property
    def fitness(self):
        '''
        Population fitness == average member fitness score
        '''

        if len(self.__members) != 0:
            if self.__num_processes > 0:
                members = [self.__members.get() for p in self.__processes]
            else:
                members = self.__members
            return sum(m.fitness_score for m in members)/len(members)
        else:
            return None

    @property
    def param_vals(self):
        '''
        Population parameter vals == average member parameter vals
        '''

        if len(self.__members) != 0:
            if self.__num_processes > 0:
                members = [self.__members.get() for p in self.__processes]
            else:
                members = self.__members
            params = {}
            for p in self.__parameters:
                params[p.name] = sum(
                    m.feed_dict[p.name] for m in members
                )/len(members)
            return params
        else:
            return None

    @property
    def members(self):
        '''
        Returns Member objects from population
        '''

        if self.__num_processes > 0:
            return [m.get() for m in self.__members]
        else:
            return self.__members

    def add_parameter(self, name, min_val, max_val):
        '''
        Adds a paramber to the list of population parameters to tune

        *name*      -   name of the parameter
        *min_val*   -   minimum allowed value of the parameter
        *max_val*   -   maximum allowed value of the parameter
        '''

        self.__parameters.append(Parameter(name, min_val, max_val))

    def generate_population(self):
        '''
        Generates self.__pop_size Members with randomly initialized values
        for each parameter added with add_parameter(), evaluates their fitness
        '''

        if self.__num_processes > 0:
            process_pool = mp.Pool(processes=self.__num_processes)
        self.__members = []

        for _ in range(self.__pop_size):
            feed_dict = {}
            for param in self.__parameters:
                feed_dict[param.name] = self.__random_param_val(
                    param.min_val,
                    param.max_val,
                    param.dtype
                )
            if self.__num_processes > 0:
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

        if self.__num_processes > 0:
            process_pool.close()
            process_pool.join()

    def next_generation(self, num_survivors, mut_rate=0, max_mut_amt=0):
        '''
        Generates the next population from a previously evaluated generation

        *num_survivors*     -   number of top performers (from self.select_fn)
                                to generate the next population from
        *mut_rate*          -   chance for each new population member to mutate
                                (between 0 and 1)
        *max_mut_amt*       -   if new member is mutating, maximum possible
                                change amount (between 0 and 1, is multiplied
                                by max param val - min param val)
        '''

        if self.__num_processes > 0:
            process_pool = mp.Pool(processes=self.__num_processes)
            members = [m.get() for m in self.__members]
        else:
            members = self.__members

        if len(members) == 0:
            raise Exception(
                'Generation 0 not found: use generate_population() first'
            )

        selected_members = self.__select_fn(members, num_survivors)
        reproduction_probs = list(reversed(np.logspace(0.0, 1.0,
                                  num=num_survivors, base=10)))
        reproduction_probs = reproduction_probs / sum(reproduction_probs)

        self.__members = []

        for _ in range(self.__pop_size):
            parent_1 = np.random.choice(selected_members, p=reproduction_probs)
            parent_2 = np.random.choice(selected_members, p=reproduction_probs)

            feed_dict = {}
            for param in self.__parameters:
                which_parent = random.uniform(0, 1)
                if which_parent < 0.5:
                    feed_dict[param.name] = parent_1.feed_dict[param.name]
                else:
                    feed_dict[param.name] = parent_2.feed_dict[param.name]
                feed_dict[param.name] = self.__mutate_parameter(
                    feed_dict[param.name], param, mut_rate, max_mut_amt
                )

            if self.__num_processes > 0:
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

        if self.__num_processes > 0:
            process_pool.close()
            process_pool.join()

    @staticmethod
    def _start_process(cost_fn, feed_dict, cost_fn_args):
        '''
        Static method: starts a process to generate (evaluate) a new Member
        with parameter values in *feed_dict*
        '''

        return Member(feed_dict, cost_fn(feed_dict, cost_fn_args))

    @staticmethod
    def __random_param_val(min_val, max_val, dtype):
        '''
        Private, static method: returns a random value between *min_val* and
        *max_val* of type *dtype*
        '''

        return SUPPORTED_DTYPES[dtype](min_val, max_val)

    @staticmethod
    def __mutate_parameter(value, param, mut_rate, max_mut_amt):
        '''
        Private, static method: mutates a *param*'s *value*; chance of mutation
        depends on *mut_rate*, maximum change amount depends on *max_mut_amt*
        '''

        if random.uniform(0, 1) < mut_rate:
            mut_amt = random.uniform(0, max_mut_amt)
            op = random.choice((add, sub))
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
