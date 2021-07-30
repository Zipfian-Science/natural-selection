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
            island._verbose_logging(f"mutate: gene_before {repr(individual.chromosome[i])}")
            individual.chromosome[i].randomise()
            island._verbose_logging(f"mutate: gene_after {repr(individual.chromosome[i])}")

    return individual