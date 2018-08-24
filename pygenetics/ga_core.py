#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ga_core.py
#
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>

# Third party open source packages
import random
import numpy as np
import types
from operator import add, sub

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


class Population:
    '''
    Population object: tunes specified parameters by measuring the performance
    of population members
    '''

    def __init__(self, size, cost_fn,
                 select_fn=selection_functions.minimize_best_n):
        '''
        Initialize the population of size *size*, a *cost_fn* to run for each
        member, and a *select_fn* to select/order members based on performance
        '''

        if size <= 0:
            raise ValueError('Population *size* cannot be <= 0')
        self.pop_size = size
        if not callable(cost_fn):
            raise ValueError('Supplied *cost_fn* is not callable')
        self.cost_fn = cost_fn
        if not callable(select_fn):
            raise ValueError('Supplied *select_fn* is not callable')
        self.select_fn = select_fn
        self.parameters = []
        self.members = []

    def __len__(self):
        '''
        len(Population) == population size
        '''

        return self.pop_size

    @property
    def fitness(self):
        '''
        Population fitness == average member fitness score
        '''

        if len(self.members) != 0:
            return sum(m.fitness_score for m in self.members)/len(self.members)
        else:
            return None

    def add_parameter(self, name, min_val, max_val):
        '''
        Adds a paramber to the list of population parameters to tune, with name
        *name*, minimum value *min_val*, maximum value *max_val*
        '''

        self.parameters.append(Parameter(name, min_val, max_val))

    def generate_population(self):
        '''
        Generates self.pop_size Members with randomly initialized values
        for each parameter added with add_parameter()
        '''

        for _ in range(self.pop_size):
            feed_dict = {}
            for param in self.parameters:
                feed_dict[param.name] = self.__random_param_val(param.min_val,
                                                                param.max_val,
                                                                param.dtype)
            fitness_score = self.cost_fn(feed_dict)
            self.members.append(Member(feed_dict, fitness_score))

    def next_generation(self, num_survivors, mut_rate=0.1, max_mut_amt=0.1):
        '''
        Generates self.pop_size new Members using *num_survivors* Members
        from the previous population selected using self.select_fn (selection
        function returns an ordered list of Members, e.g. by lowest fitness
        score); Optional arguments for mutation rate *mut_rate* and maximum
        mutation amount (proportion of parameter range) *max_mut_amt*
        '''

        selected_members = self.select_fn(self.members, num_survivors)
        reproduction_probs = list(reversed(np.logspace(0.0, 1.0,
                                  num=num_survivors, base=10)))
        reproduction_probs = reproduction_probs / sum(reproduction_probs)

        self.members = []

        for _ in range(self.pop_size):
            parent_1 = np.random.choice(selected_members, p=reproduction_probs)
            parent_2 = np.random.choice(selected_members, p=reproduction_probs)

            feed_dict = {}
            for param in self.parameters:
                which_parent = random.uniform(0, 1)
                if which_parent < 0.5:
                    feed_dict[param.name] = parent_1.feed_dict[param.name]
                else:
                    feed_dict[param.name] = parent_2.feed_dict[param.name]
                feed_dict[param.name] = self.__mutate_parameter(
                    feed_dict[param.name], param, mut_rate, max_mut_amt
                )

            fitness_score = self.cost_fn(feed_dict)
            self.members.append(Member(feed_dict, fitness_score))

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
