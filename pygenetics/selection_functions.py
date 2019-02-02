#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# selection_functions.py (0.6.0)
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#


def minimize_best_n(Members):
    '''
    Orders population members from lowest fitness to highest fitness

    Args:
        Members (list): list of PyGenetics Member objects

    Returns:
        lsit: ordered lsit of Members, from highest fitness to lowest fitness
    '''

    return(list(reversed(sorted(
        Members, key=lambda Member: Member.fitness_score
    ))))
