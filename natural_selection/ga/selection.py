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

def top_n_selection(island, population, n):
    """
    A Classic top N selection function, sorted on fitness.

    Args:
        island (Island): The Island calling the method.
        population (list): A list of Individuals.
        n (int): Number to select.

    Returns:
        list: Top N Individuals.
    """
    def sortFitness(val):
        return val.fitness

    population.sort(key=sortFitness, reverse=True)

    return population[0:n]