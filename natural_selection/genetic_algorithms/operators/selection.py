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

def selection_elites_tournament(individuals : list, n : int, desc : bool = True, island=None) -> list:
    raise NotImplemented("Not yet there, but soon!")

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

def selection_parents_two(individuals : list, island=None):
    """
    Simple function to select two parents at a time, sequentially.

    Args:
        individuals (list): A list of Individuals, specifically the selected "elites".
        island (Island): The Island calling the method (optional, default = None).

    Yields:
        list: Containing the two individuals selected for crossover.
    """
    for parent_1, parent_2 in zip(individuals[::2], individuals[1::2]):
        yield [parent_1, parent_2]

def selection_survivors(individuals : list, n : int, desc : bool = True, island=None) -> list:
    raise NotImplemented("Not yet there, but soon!")