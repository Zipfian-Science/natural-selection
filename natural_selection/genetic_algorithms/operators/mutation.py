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

def mutation_randomize(individual, prob : float = 0.2, island=None):
    """
    A Classic mutation function, changes a gene of the given individual based on the `prob` strength of mutation.

    Args:
        individual (Individual): Individual object containing a Genome.
        prob (float): The probability of randomizing genes (default = 0.2).
        island (Island): The Island calling the method (default = None).

    Returns:
        Individual: The newly mutated individual.
    """
    for i in range(len(individual.chromosome)):
        if random.random() < prob:
            island.verbose_logging(f"mutate: gene_before {repr(individual.chromosome[i])}")
            individual.chromosome[i].randomise()
            island.verbose_logging(f"mutate: gene_after {repr(individual.chromosome[i])}")

    return individual

def mutation_randomize_n_point(individual, n_points : int = 1, prob : float = 0.2, island=None) -> list:
    """
    Much like n_point crossover, random slices of the chromosome are selected and then all genes in those selected slices are randomised.
    This method might be a little more aggressive, as it can mutate longer strings at a time.

    Args:
        individual (Individual): Individual object containing a Genome.
        n_points (int): The amount of random points to split at (default = 1).
        prob (float): The probability of randomizing genes (default = 0.2).
        island (Island): The Island calling the method (default = None).

    Returns:
        Individual: The newly mutated individual.
    """

    size = len(individual.chromosome)

    point_cut_list = random.sample(range(1,size-1), n_points)
    point_cut_list.sort()
    point_cut_list.insert(0,0)
    point_cut_list.append(size)
    for i in range(len(point_cut_list)-1):
        if random.random() < prob:
            b = point_cut_list[i]
            e = point_cut_list[i+1]
            for g_id in range(b,e):
                island.verbose_logging(f"mutate: gene_before {repr(individual.chromosome[i])}")
                individual.chromosome[g_id].randomise()
                island.verbose_logging(f"mutate: gene_after {repr(individual.chromosome[i])}")

    return individual