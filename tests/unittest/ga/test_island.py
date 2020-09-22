import unittest
import random
from natural_selection.ga import Gene, Genome, Individual, Island

class TestSimpleIsland(unittest.TestCase):

    def setUp(self) -> None:
        self.fitness = lambda gen, x, y: (gen[0].value * x) + (gen[1].value *y)
        g_1 = Gene("first", 1, 10, 1, random.randint)
        g_2 = Gene("second", 1, 100, 1, random.randint)
        gen = Genome([g_1, g_2])
        self.ind = Individual(self.fitness, name="Adam", genome=gen)

        self.life = Island({'x': 0.5, 'y' : 0.2})

    def test_create(self):
        self.life.create(self.ind, population_size=5)
        self.assertEquals(len(self.life.population), 5)

    def test_import_migrants(self):
        self.life.create(self.ind, population_size=5)

        fitness = lambda gen, x, y: (gen[0].value * x) + (gen[1].value * y)
        g_1 = Gene("first", 1, 10, 1, random.randint)
        g_2 = Gene("second", 1, 100, 1, random.randint)
        gen = Genome([g_1, g_2])

        aliens = [Individual(fitness, name="AlsoAdam", genome=gen)]

        self.life.import_migrants(aliens)

        self.assertEquals(len(self.life.population), 5)

        fitness = lambda gen, x, y: (gen[0].value * x) + (gen[1].value * y)
        g_1 = Gene("first", 5, 10, 1, random.randint)
        g_2 = Gene("second", 99, 100, 1, random.randint)
        gen = Genome([g_1, g_2])

        aliens = [Individual(fitness, name="Eve", genome=gen)]

        self.life.import_migrants(aliens)

        self.assertEquals(len(self.life.population), 6)

    def test_import_evolution(self):
        self.life.create(self.ind, population_size=5)
        self.life.evolve()