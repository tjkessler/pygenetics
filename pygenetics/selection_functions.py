#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#	selection_functions.py
#  
#	Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
#   Contains pre-built selection functions
#

# Select *n* *members* with the lowest fitness score
def minimize_best_n(Members, n):
    return(sorted(Members, key = lambda Member: Member.fitness_score)[0:n])

# Select *n* *members* with the largest fitness score
def maximize_best_n(Members, n):
    return(sorted(Members, key = lambda Member: Member.fitness_score)[0:n])