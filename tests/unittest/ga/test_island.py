import os.path
import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual, Island
from natural_selection.genetic_algorithms.utils.random_functions import random_int
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_n_point

class TestSimpleIsland(unittest.TestCase):

    def setUp(self) -> None:
        self.fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        self.ind = Individual(self.fitness, name="Adam", chromosome=gen)
        self.ind.add_new_property('some_property', 10)

        self.life = Island({'x': 0.6, 'y' : 0.2}, crossover_function=crossover_two_n_point)

    def test_write_report(self):
        self.life.initialise(self.ind, population_size=5)
        self.life.evolve(crossover_params={'n_points': 2})

        self.life.write_report('test.csv')
        self.life.write_report('test.json', output_json=True)

        self.assertTrue(os.path.isfile('test.csv'))
        self.assertTrue(os.path.isfile('test.json'))

        os.remove('test.csv')
        os.remove('test.json')

    def test_lineage(self):
        self.life.initialise(self.ind, population_size=5)
        self.life.evolve(crossover_params={'n_points': 2}, n_generations=10, crossover_probability=0.6)

        self.assertTrue(len(self.life.lineage['edges']) > 1)

        self.life.write_lineage_json('test.json')

        self.assertTrue(os.path.isfile('test.json'))

        os.remove('test.json')

    def test_create(self):
        self.life.initialise(self.ind, population_size=5)
        self.assertEquals(len(self.life.population), 5)

    def test_import_migrants(self):
        self.life.initialise(self.ind, population_size=5)

        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])

        aliens = [Individual(fitness, name="AlsoAdam", chromosome=gen)]

        self.life.import_migrants(aliens)

        self.assertEquals(len(self.life.population), 5)

        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=5, gene_max=10, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=99, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])

        aliens = [Individual(fitness, name="Eve", chromosome=gen)]

        self.life.import_migrants(aliens)

        self.assertEquals(len(self.life.population), 6)

    def test_evolve_generational(self):
        self.life.initialise(self.ind, population_size=5)
        self.life.evolve(crossover_params={'n_points' : 2})

class TestSimpleIslandMinimise(unittest.TestCase):

    def setUp(self) -> None:
        self.fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=1, gene_max=10, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=10, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])
        self.ind = Individual(self.fitness, name="Adam", chromosome=gen)
        self.ind.add_new_property('some_property', 10)

        self.life = Island({'x': 0.6, 'y' : 0.2}, maximise_function=False)

    def test_evolve_generational(self):
        self.life.initialise(self.ind, population_size=5)
        best = self.life.evolve()

        self.assertLessEqual(best.fitness, 8)

class TestSimpleIslandMaximise(unittest.TestCase):

    def setUp(self) -> None:
        self.fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=1, gene_max=10, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=10, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])
        self.ind = Individual(self.fitness, name="Adam", chromosome=gen)
        self.ind.add_new_property('some_property', 10)

        self.life = Island({'x': 0.6, 'y' : 0.2})

    def test_evolve_generational(self):
        self.life.initialise(self.ind, population_size=5)
        best = self.life.evolve()

        self.assertGreaterEqual(best.fitness, 0.8)

class TestOtherIsland(unittest.TestCase):

    def setUp(self) -> None:
        self.fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        self.ind = Individual(self.fitness, name="Adam", chromosome=gen)

        self.life = Island({'x': 0.6, 'y' : 0.2}, crossover_function=crossover_two_n_point)

    def test_create(self):
        self.life.initialise(self.ind, population_size=5)
        self.assertEquals(len(self.life.population), 5)

    def test_import_migrants(self):
        self.life.initialise(self.ind, population_size=5)

        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])

        aliens = [Individual(fitness, name="AlsoAdam", chromosome=gen)]

        self.life.import_migrants(aliens)

        self.assertEquals(len(self.life.population), 5)

        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)
        g_1 = Gene(name="first", value=5, gene_max=10, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=99, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])

        aliens = [Individual(fitness, name="Eve", chromosome=gen)]

        self.life.import_migrants(aliens)

        self.assertEquals(len(self.life.population), 6)

    def test_evolve_generational(self):
        self.life.initialise(self.ind, population_size=5)
        self.life.evolve()

def fitness(individual, island, x, y):
    return (individual.chromosome[0].value * x) + (individual.chromosome[1].value * y)

def fitness_raise_error(individual, island, x, y):
    return 1 / 0

class TestSaveLoad(unittest.TestCase):

    def test_save_load(self):
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind = Individual(fitness, name="Adam", chromosome=gen)

        life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point)

        life.initialise(ind, population_size=5)

        life.evolve()

        fp = 'testing_island.pkl'

        life.save_island(fp)

        self.assertTrue(os.path.isfile(fp))

        new_life = Island(filepath=fp)

        self.assertDictEqual(new_life.function_params, life.function_params)
        self.assertListEqual(new_life.population, life.population)
        self.assertEqual(new_life.name, life.name)

        new_life.evolve(starting_generation=new_life.generation_count)

        self.assertEqual(new_life.generation_count, life.generation_count*2)
        os.remove(fp)


class TestException(unittest.TestCase):

    def test_raise_exception(self):
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])
        ind = Individual(fitness_raise_error, name="Adam", chromosome=gen)

        life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, verbose=True)

        try:
            life.initialise(ind, population_size=5)
        except:
            self.assertTrue(True)

class TestCheckpoints(unittest.TestCase):

    def test_save_load(self):
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2])
        ind = Individual(fitness, name="Adam", chromosome=gen)

        life = Island({'x': 0.6, 'y': 0.2}, save_checkpoint_level=2)

        life.initialise(ind, population_size=5)

        life.evolve()
