# -*- coding: utf-8 -*-
'''
    Base functions for running population mutation
    @author Justin Hocking
    @version 0.0.1
'''
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2020, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"

import random

def classic_mutate_function(island, individual, prob):
    for i in range(len(individual.genome)):
        if random.random() < prob:
            individual.genome[i] = individual.genome[i].randomize()
            individual.reset_fitness()

    return individual