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

def selection_parents_two(individuals : list, island=None):
    """
    Simple function to select two parents at a time, sequentially. Parental selection always yields.

    Args:
        individuals (list): A list of Individuals, specifically the selected "elites".
        island (Island): The Island calling the method (optional, default = None).

    Yields:
        list: Containing the two individuals selected for crossover.
    """
    for parent_1, parent_2 in zip(individuals[::2], individuals[1::2]):
        yield [parent_1, parent_2]

def selection_parents_two_shuffled(individuals : list, island=None):
    raise NotImplementedError("Future implementation")

def selection_parents_n_gram(individuals : list, n : int = 2,island=None):
    raise NotImplementedError("Future implementation")