# -*- coding: utf-8 -*-
'''
    Base functions for running population crossover (mating)
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

def classic_mate_function(island, mother, father, prob):
    size = min(len(mother.genome), len(father.genome))
    for i in range(size):
        if random.random() < prob:
            mother.genome[i], father.genome[i] = father.genome[i], mother.genome[i]
            mother.reset_fitness()
            father.reset_fitness()

    return mother, father