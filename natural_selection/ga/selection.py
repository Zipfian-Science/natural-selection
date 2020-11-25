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


def selection_function_classic(population : list, n : int, desc : bool = True, island=None) -> list:
    """
    A Classic top N selection function, sorted on fitness.

    Args:
        population (list): A list of Individuals.
        n (int): Number to select.
        desc (bool): In descending order (default = True).
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        list: Top N Individuals.
    """
    def sortFitness(val):
        return val.fitness

    population.sort(key=sortFitness, reverse=desc)

    return population[0:n]