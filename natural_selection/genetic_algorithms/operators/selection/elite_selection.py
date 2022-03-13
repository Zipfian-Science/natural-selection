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

def selection_elites_tournament(individuals : list, n : int = 4, tournament_size : int = 5, island=None) -> list:
    """
    Classic tournament selection. Given a number of selection rounds (`n`), select a random list of individuals of `tournament_size` and select the top individual from the random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): The number of tournaments to run, effectively the number of selected individuals to return (default = 4).
        tournament_size (int): The number of random individuals to select during each tournament (default = 5).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Top n Individuals from tournaments.
    """
    elites = []
    for i in range(n):
        selection = selection_elites_random(individuals, tournament_size, island=island)
        elites.extend(selection_elites_top_n(selection, 1, island=island))
    return elites

def selection_elites_tournament_unique(individuals : list, n : int = 4, tournament_size : int = 5, max_step : int = 100, island=None) -> list:
    """
    Classic tournament selection but ensures a unique list of selected individuals. Given a number of selection rounds (`n`), select a random list of individuals of `tournament_size` and select the top individual from the random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): The number of tournaments to run, effectively the number of selected individuals to return (default = 4).
        tournament_size (int): The number of random individuals to select during each tournament (default = 5).
        max_step (int): In the unlikely event that a unique list of size `n` can not be achieved, break out of the loop after this amount of steps (default = 100).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Top n Individuals from tournaments.
    """
    elites = []
    i = 0
    steps = 0
    while i < n:
        selection = selection_elites_random(individuals, tournament_size, island=island)
        fittest = selection_elites_top_n(selection, 1, island=island)[0]
        steps += 1
        if not fittest in elites:
            elites.append(fittest)
            i += 1
            continue
        if steps >= max_step:
            break
    return elites

def selection_elites_random(individuals : list, n : int = 4, island=None) -> list:
    """
    Completely random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): Number to select (default = 4).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Random n Individuals.
    """
    return random.choice(individuals, size=n).tolist()

def selection_elites_top_n(individuals : list, n : int = 4, desc : bool = True, island=None) -> list:
    """
    A Classic top N selection function, sorted on fitness.

    Args:
        individuals (list): A list of Individuals.
        n (int): Number to select (default = 4).
        desc (bool): In descending order, only used if Island is None, else `maximise_function` overrides (default = True).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Top n Individuals.
    """
    def sortFitness(val):
        return val.fitness

    if island:
        individuals.sort(key=sortFitness, reverse=island.maximise_function)
    else:
        individuals.sort(key=sortFitness, reverse=desc)

    return individuals[0:n]