# -*- coding: utf-8 -*-
"""Basic methods for mutation operations.
"""
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2020, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"

import random

def mutatation_function_classic(individual, prob : float, island=None):
    """
    A Classic mutation function.

    Args:
        individual: Individual object containing a Genome.
        prob: The probability of swapping genes.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        Individual: The newly mutated individual.
    """
    for i in range(len(individual.genome)):
        if random.random() < prob:
            individual.genome[i] = individual.genome[i].randomize()
            individual.reset_fitness()

    return individual