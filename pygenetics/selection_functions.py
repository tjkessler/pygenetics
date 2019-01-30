#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# selection_functions.py (0.5.2)
#
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#


def minimize_best_n(Members):
    '''
    Orders population members from lowest fitness to highest fitness

    Args:
        Members (list): list of Pygenetics Member objects

    Returns:
        lsit: ordered lsit of Members, from lowest to highest
    '''

    return(list(reversed(sorted(
        Members, key=lambda Member: Member.fitness_score
    ))))
