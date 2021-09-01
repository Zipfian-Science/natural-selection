import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection.genetic_algorithms.utils.random_functions import random_int

class TestIndividual(unittest.TestCase):

    def setUp(self) -> None:
        self.fitness = lambda individual, island, x, y: individual.chromosome[0].value * x + individual.chromosome[0].value * y
        g_1 = Gene(name="test", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="test_other", value=4, gene_max=10, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])
        self.ind = Individual(self.fitness, name="Adam", chromosome=gen)

    def test_birthday(self):
        self.assertEquals(self.ind.age, 0)
        self.ind.birthday()
        self.assertEquals(self.ind.age, 1)
        self.ind.birthday(3)
        self.assertEquals(self.ind.age, 4)

    def test_reset_fitness(self):
        self.ind.fitness = 5
        self.ind.reset_fitness()
        self.assertIsNone(self.ind.fitness)
        self.ind.reset_fitness(fitness=4)
        self.assertEquals(self.ind.fitness, 4)

        self.ind.genetic_code = "true"
        self.ind.reset_fitness(reset_genetic_code=False)
        self.assertEquals(self.ind.genetic_code, "true")
        self.ind.reset_fitness(reset_genetic_code=True)
        self.assertIsNone(self.ind.genetic_code)

    def test_add_gene(self):
        self.ind.add_gene(Gene(name="test", value=3, gene_max=10, gene_min=1, randomise_function=random_int))
        self.assertEquals(len(self.ind.chromosome), 3)

    def test_evaluate(self):
        f = self.ind.evaluate({'x' : 2, 'y' : 5})
        self.assertEquals(f, 21)
        self.assertNotEquals(f, 10)

    def test_unique_genetic_code(self):
        code = self.ind.unique_genetic_code()
        self.assertIsNotNone(code)