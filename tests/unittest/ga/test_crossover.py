import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection.genetic_algorithms.utils.random_functions import random_int
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_n_point

class TestNpointCx(unittest.TestCase):

    def test_crossover_two_n_point(self):

        gen = Chromosome([
            Gene(name="first", value=1,randomise_function=random_int) for i in range(50)
        ])
        gen_2 = Chromosome([
            Gene(name="second", value=2, randomise_function=random_int) for i in range(50)
        ])
        ind_1 = Individual(print, name="Adam", chromosome=gen)
        ind_2 = Individual(print, name="Eve", chromosome=gen_2)

        offspring = crossover_two_n_point([ind_1, ind_2], n_points=15)

        self.assertNotEquals(offspring[0], offspring[1])

