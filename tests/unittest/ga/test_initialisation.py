import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection import Island
from natural_selection.genetic_algorithms.utils.random_functions import random_int
from natural_selection.genetic_algorithms.operators.mutation import mutation_randomize_n_point
from natural_selection.genetic_algorithms.operators.initialisation import initialise_population_mutation_function

class TestInitialisations(unittest.TestCase):

    def test_initialise_population_mutation_function(self):
        fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (
                    individual.chromosome[1].value * y)
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        ind = Individual(fitness, name="Adam", chromosome=gen)

        life = Island({'x': 0.6, 'y': 0.2},
                      initialisation_function=initialise_population_mutation_function,
                      mutation_function=mutation_randomize_n_point)

        init_params = {'mutation_params': {
            'n_points':  2, 'prob':  0.5
        }}

        life.initialise(ind, population_size=5, initialisation_params=init_params)

        life.initialise(ind, population_size=5)

        self.assertTrue(True)