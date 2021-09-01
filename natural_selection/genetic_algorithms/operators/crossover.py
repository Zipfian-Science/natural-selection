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

def crossover_two_uniform(individuals : list, prob : float = 0.5, island=None) -> list:
    """
    A Classic crossover function, taking a list of 2 individuals and swapping positional genes based on the `prob` strength of crossover.
    Returns these two individuals with modified genomes.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        prob (float): The probability of swapping genes.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]
    size = min(len(mother.chromosome), len(father.chromosome))
    for i in range(size):
        if random.random() < prob:
            mother.chromosome[i], father.chromosome[i] = father.chromosome[i], mother.chromosome[i]

    return [mother, father]

def crossover_two_n_point(individuals : list, n_points : int = 1, prob : float = 0.5, island=None) -> list:
    """
    Classic crossover method to randomly select N points for crossover.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        n_points (int): The amount of random points to split at (default = 1).
        prob (float): The probability of swapping genes.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]
    size = min(len(mother.chromosome), len(father.chromosome))

    point_cut_list = random.sample(range(1,size-1), n_points)
    point_cut_list.sort()
    point_cut_list.insert(0,0)
    point_cut_list.append(size)
    for i in range(len(point_cut_list)-1):
        if random.random() < prob:
            b = point_cut_list[i]
            e = point_cut_list[i+1]
            mother.chromosome[b:e], father.chromosome[b:e] = father.chromosome[b:e], mother.chromosome[b:e]

    return [mother, father]