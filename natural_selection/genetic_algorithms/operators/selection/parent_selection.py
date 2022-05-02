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

def selection_roulette(individuals : list, n : int = 4, with_replacement : bool = True, island=None) -> list:
    """
    Classic roulette wheel selection, or also known as Fitness proportionate selection.
    This method selects individuals based on their fitness in proportion to the whole population.

    Note:
        Can only be used for function maximisation.

    Args:
        individuals (list): A list of Individuals.
        n (int): The number of selected individuals to return (default = 4).
        with_replacement (bool) : Whether the sample is with or without replacement (default = True).
        island (Island): The Island calling the method (default = None).

    Raises:
        ValueError : If the island is set for function minimisation.

    Returns:
        list:  n Individuals from roulette selection.
    """
    if island and not island.maximise_function:
        raise ValueError('Roulette selection not allowed for function minimisation')

    population_fitness_sum = sum([i.fitness for i in individuals])

    probabilities = [i.fitness / population_fitness_sum for i in individuals]

    # Selects one chromosome based on the computed probabilities
    return random.choice(individuals, size=n, replace=with_replacement, p=probabilities).tolist()

def selection_almost_roulette_minimisation(individuals : list, n : int = 4, with_replacement : bool = True, island=None) -> list:
    """
    Almost roulette wheel selection selection but for function minimisation.
    This method selects individuals based on their fitness in proportion to the whole population.

    Note:
        Can only be used for function minimisation.

    Args:
        individuals (list): A list of Individuals.
        n (int): The number of selected individuals to return (default = 4).
        with_replacement (bool) : Whether the sample is with or without replacement (default = True).
        island (Island): The Island calling the method (default = None).

    Raises:
        ValueError : If the island is set for function maximisation.

    Returns:
        list:  n Individuals from roulette selection.
    """
    if island and island.maximise_function:
        raise ValueError('Almost Roulette Minimisation selection not allowed for function maximisation')

    population_fitness_sum = sum([i.fitness for i in individuals])

    probabilities = [i.fitness / population_fitness_sum for i in individuals]

    probabilities = [(1 - p) / (len(individuals) - 1) for p in probabilities]

    # Selects one chromosome based on the computed probabilities
    return random.choice(individuals, size=n, replace=with_replacement, p=probabilities).tolist()

def selection_tournament(individuals : list, n : int = 4, tournament_size : int = 5, with_replacement : bool = True, island=None) -> list:
    """
    Classic tournament selection. Given a number of selection rounds (`n`), select a random list of individuals of `tournament_size` and select the top individual from the random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): The number of tournaments to run, effectively the number of selected individuals to return (default = 4).
        tournament_size (int): The number of random individuals to select during each tournament (default = 5).
        with_replacement (bool) : Whether the sample is with or without replacement (default = True).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Top n Individuals from tournaments.
    """
    parents = []
    for i in range(n):
        selection = selection_random(individuals, tournament_size, island=island, with_replacement=with_replacement)
        parents.extend(selection_elites_top_n(selection, 1, island=island))
    return parents

def selection_tournament_unique(individuals : list, n : int = 4, tournament_size : int = 5, with_replacement : bool = True, max_step : int = 100, island=None) -> list:
    """
    Classic tournament selection but ensures a unique list of selected individuals. Given a number of selection rounds (`n`), select a random list of individuals of `tournament_size` and select the top individual from the random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): The number of tournaments to run, effectively the number of selected individuals to return (default = 4).
        tournament_size (int): The number of random individuals to select during each tournament (default = 5).
        max_step (int): In the unlikely event that a unique list of size `n` can not be achieved, break out of the loop after this amount of steps (default = 100).
        with_replacement (bool) : Whether the sample is with or without replacement (default = True).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Top n Individuals from tournaments.
    """
    parents = []
    i = 0
    steps = 0
    while i < n:
        selection = selection_random(individuals, tournament_size, island=island, with_replacement=with_replacement)
        fittest = selection_elites_top_n(selection, 1, island=island)[0]
        steps += 1
        if not fittest in parents:
            parents.append(fittest)
            i += 1
            continue
        if steps >= max_step:
            break
    return parents

def selection_random(individuals : list, n : int = 4, with_replacement : bool = True, island=None) -> list:
    """
    Completely random selection.

    Args:
        individuals (list): A list of Individuals.
        n (int): Number to select (default = 4).
        with_replacement (bool) : Whether the sample is with or without replacement (default = True).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Random n Individuals.
    """
    return random.choice(individuals, size=n, replace=with_replacement).tolist()

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