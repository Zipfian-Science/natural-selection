import unittest
from natural_selection.genetic_algorithms import Gene
from natural_selection.genetic_algorithms.utils.random_functions import random_int

class TestGene(unittest.TestCase):

    def test_init(self):
        g = Gene(name="test", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
        self.assertEquals(str(g), '(test:3)')

        g_2 = g.randomise_new()

        self.assertNotEqual(g, g_2)