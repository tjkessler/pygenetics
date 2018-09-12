#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# selection_functions.py (0.4.0)
#
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#


def minimize_best_n(Members, n):
    '''
    Select *n* *Members* with the lowest fitness score
    '''

    return(sorted(Members, key=lambda Member: Member.fitness_score)[0:n])


def maximize_best_n(Members, n):
    '''
    Select *n* *Members* with the largest fitness score
    '''

    return(sorted(Members, key=lambda Member: Member.fitness_score,
                  reverse=True)[0:n])
