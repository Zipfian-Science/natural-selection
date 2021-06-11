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
    """
    Classic tournament selection. Given a number of selection rounds (`n`), select a random list of individuals of `tournament_size` and select the top individual from the random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): The number of tournaments to run, effectively the number of selected individuals to return.
        tournament_size (int): The number of random individuals to select during each tournament (default = 5).
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        list: Top n Individuals from tournaments.
    """
    elites = []
    for i in range(n):
        selection = selection_elites_random(individuals, tournament_size)
        elites.extend(selection_elites_top_n(selection, 1))
    return elites

def selection_elites_random(individuals : list, n : int, island=None) -> list:
    """
    Completely random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): Number to select.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        list: Random n Individuals.
    """
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
        list: Top n Individuals.
    """
    def sortFitness(val):
        return val.fitness

    individuals.sort(key=sortFitness, reverse=desc)

    return individuals[0:n]