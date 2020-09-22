# -*- coding: utf-8 -*-
'''
    Base functions for running population selection
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

def top_n_selection(island, population, n):
    def sortFitness(val):
        return val.fitness

    population.sort(key=sortFitness, reverse=True)

    return population[0:n]