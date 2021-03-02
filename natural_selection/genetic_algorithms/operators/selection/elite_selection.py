# -*- coding: utf-8 -*-
"""Basic methods for selection operations.
"""
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2020, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"

from numpy import random

def selection_elites_tournament(individuals : list, n : int, tournament_size : int = 5, island=None) -> list:
    elites = []
    for i in range(n):
        selection = selection_elites_random(individuals, tournament_size)
        elites.extend(selection_elites_top_n(selection, 1))
    return elites

def selection_elites_random(individuals : list, n : int, island=None) -> list:
    return random.choice(individuals, size=n).tolist()

def selection_elites_top_n(individuals : list, n : int, desc : bool = True, island=None) -> list:
    """
    A Classic top N selection function, sorted on fitness.

    Args:
        individuals (list): A list of Individuals.
        n (int): Number to select.
        desc (bool): In descending order (default = True).
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        list: Top N Individuals.
    """
    def sortFitness(val):
        return val.fitness

    individuals.sort(key=sortFitness, reverse=desc)

    return individuals[0:n]