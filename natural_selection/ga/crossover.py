# -*- coding: utf-8 -*-
"""Basic methods for crossover operations.
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
from natural_selection.ga import Island, Individual

def crossover_function_classic(mother : Individual, father : Individual, prob : float, island : Island = None) -> tuple:
    """
    A Classic crossover function.

    Args:
        mother (Individual): Individual object containing a Genome.
        father (Individual): Individual object containing a Genome.
        prob (float): The probability of swapping genes.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        tuple: Two new Individuals.
    """
    size = min(len(mother.genome), len(father.genome))
    for i in range(size):
        if random.random() < prob:
            mother.genome[i], father.genome[i] = father.genome[i], mother.genome[i]
            mother.reset_fitness()
            father.reset_fitness()

    return mother, father