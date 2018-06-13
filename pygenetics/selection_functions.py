#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#	selection_functions.py
#  
#	Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
#   Contains pre-built selection functions
#

def minimize_best_n(Members, n):
    return(sorted(Members, key = lambda Member: Member.fitness_score)[0:n])