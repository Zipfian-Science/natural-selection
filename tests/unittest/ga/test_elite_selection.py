import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection import Island
from natural_selection.genetic_algorithms.utils.random_functions import random_int
from natural_selection.genetic_algorithms.operators.selection import selection_tournament_unique, selection_roulette, selection_almost_roulette_minimisation

class TestRoulette(unittest.TestCase):

    def test_roultee_max(self):
        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (
                individual.chromosome[1].value * y)

        gen = Chromosome([])
        ind_1 = Individual(fitness, name="Adam1", chromosome=gen)

        gen = Chromosome([])
        ind_2 = Individual(fitness, name="Adam2", chromosome=gen)

        gen = Chromosome([])
        ind_3 = Individual(fitness, name="Adam3", chromosome=gen)

        gen = Chromosome([])
        ind_4 = Individual(fitness, name="Adam4", chromosome=gen)

        ind_1.fitness = 0.9
        ind_2.fitness = 0.6
        ind_3.fitness = 0.6
        ind_4.fitness = 0.1

        e = selection_roulette([ind_1, ind_2, ind_3, ind_4], n=2)
        self.assertEqual(len(e), 2)

        # See if the reverse is possible by redefining the fitness scale as a max problem

        MAX_VAL = 1

        ind_1.fitness = MAX_VAL - 0.9
        ind_2.fitness = MAX_VAL - 0.6
        ind_3.fitness = MAX_VAL - 0.6
        ind_4.fitness = MAX_VAL - 0.1

        e = selection_roulette([ind_1, ind_2, ind_3, ind_4], n=2)
        self.assertEqual(len(e), 2)

        island = Island(verbose=False, maximise_function=False)

        self.assertRaises(ValueError, selection_roulette, *[[ind_1, ind_2, ind_3, ind_4], 2, False, island])




    def test_roultee_min(self):
        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (
                individual.chromosome[1].value * y)

        gen = Chromosome([])
        ind_1 = Individual(fitness, name="Adam1", chromosome=gen)

        gen = Chromosome([])
        ind_2 = Individual(fitness, name="Adam2", chromosome=gen)

        gen = Chromosome([])
        ind_3 = Individual(fitness, name="Adam3", chromosome=gen)

        gen = Chromosome([])
        ind_4 = Individual(fitness, name="Adam4", chromosome=gen)

        ind_1.fitness = 0.9
        ind_2.fitness = 0.6
        ind_3.fitness = 0.6
        ind_4.fitness = 0.1

        island = Island(verbose=False, maximise_function=False)

        e = selection_almost_roulette_minimisation([ind_1, ind_2, ind_3, ind_4], n=2, island=island)
        self.assertEqual(len(e), 2)

        island = Island(verbose=False, maximise_function=True)
        self.assertRaises(ValueError, selection_almost_roulette_minimisation, *[[ind_1, ind_2, ind_3, ind_4], 2, False, island])

class TestUniqueTournament(unittest.TestCase):

    def test_no_unique(self):
        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (
                individual.chromosome[1].value * y)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_1 = Individual(fitness, name="Adam", chromosome=gen)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_2 = Individual(fitness, name="Adam", chromosome=gen)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_3 = Individual(fitness, name="Adam", chromosome=gen)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_4 = Individual(fitness, name="Adam", chromosome=gen)

        ind_1.fitness = 0.5
        ind_2.fitness = 0.5
        ind_3.fitness = 0.5
        ind_4.fitness = 0.5

        e = selection_tournament_unique([ind_1, ind_2, ind_3, ind_4], n=2, tournament_size=3, max_step=10)
        self.assertEqual(len(e), 1)

    def test_one_unique(self):
        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (
                individual.chromosome[1].value * y)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_1 = Individual(fitness, name="Adam", chromosome=gen)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_2 = Individual(fitness, name="Adam", chromosome=gen)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_3 = Individual(fitness, name="Adam", chromosome=gen)

        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=2, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind_4 = Individual(fitness, name="Adam", chromosome=gen)

        ind_1.fitness = 0.5
        ind_2.fitness = 0.5
        ind_3.fitness = 0.5
        ind_4.fitness = 0.5

        e = selection_tournament_unique([ind_1, ind_2, ind_3, ind_4], n=2, tournament_size=3, max_step=10)
        self.assertGreater(len(e), 1)
