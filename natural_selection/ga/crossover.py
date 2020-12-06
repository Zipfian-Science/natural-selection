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

def crossover_function_classic(individuals : list, prob : float, island=None) -> list:
    """
    A Classic crossover function, taking a list of 2 individuals and swapping positional genes based on the `prob` strength of crossover.
    Returns these two individuals with modified genomes.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        prob (float): The probability of swapping genes.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]
    size = min(len(mother.genome), len(father.genome))
    for i in range(size):
        if random.random() < prob:
            mother.genome[i], father.genome[i] = father.genome[i], mother.genome[i]

    return [mother, father]