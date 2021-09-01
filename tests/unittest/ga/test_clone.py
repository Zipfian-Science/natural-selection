import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection.genetic_algorithms.utils.random_functions import random_int
from natural_selection.genetic_algorithms.utils import clone_classic

class TestNpointCx(unittest.TestCase):

    def test_crossover_two_n_point(self):

        gen = Chromosome([
            Gene(name="first", value=1,randomise_function=random_int, gene_min=0, gene_max=100) for i in range(5)
        ])

        ind_1 = Individual(print, name="Adam", chromosome=gen)

        new = clone_classic([ind_1])[0]

        ind_1.name = 'eve'
        ind_1.chromosome.genes[0].randomise()

        self.assertNotEquals(new.chromosome.genes[0].value, ind_1.chromosome.genes[0].value)

        new.chromosome.genes[1].randomise()

        self.assertNotEquals(new.chromosome.genes[1].value, ind_1.chromosome.genes[1].value)
