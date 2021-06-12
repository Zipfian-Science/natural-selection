import unittest
import random
from natural_selection.genetic_algorithms import Gene, Chromosome
from natural_selection.genetic_algorithms.utils.random_functions import random_int

class TestGenome(unittest.TestCase):

    def test_init(self):
        g = Gene(name="test", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g])

        self.assertGreater(len(gen.genes), 0)

    def test_append(self):
        g = Gene(name="test", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
        gen = Chromosome()
        gen.append(g)

        with self.assertRaises(AssertionError) as ae:
            gen.append(1)

    def test_indexing(self):
        g = Gene(name="test", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g])

        self.assertEquals(gen[0], g)

        with self.assertRaises(AssertionError) as ae:
            gen[2]

        with self.assertRaises(AssertionError) as ae:
            gen[0] = 1

        gen[0] = Gene("test_other", 3, 10, 1, random.randint)

        self.assertNotEqual(gen[0], g)

    def test_len_and_str_and_iter(self):
        g_1 = Gene(name="test", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="test_other", value=4, gene_max=10, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])

        self.assertEquals(len(gen), 2)
        self.assertEquals(str(gen), 'Chromosome(Gene(test:3)-Gene(test_other:4))')

        for g in gen:
            self.assertIsInstance(g, Gene)

        r = repr(gen)
        s = str(gen)

        self.assertNotEqual(r,s)