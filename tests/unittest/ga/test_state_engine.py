import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection import Island
from natural_selection.genetic_algorithms.utils.random_functions import random_int
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_n_point
from natural_selection.utils.population_growth import population_generational, population_steady_state_remove_oldest, population_steady_state_remove_weakest

class TestSimpleIsland(unittest.TestCase):

    def setUp(self) -> None:
        self.fitness = lambda individual, island, x, y: (individual.chromosome[0].value * x) + (
                    individual.chromosome[1].value * y)
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        self.ind = Individual(self.fitness, name="Adam", chromosome=gen)

    def test_population_steady_state_remove_oldest(self):

        life = Island({'x': 0.6, 'y' : 0.2}, crossover_function=crossover_two_n_point,
                      population_growth_function=population_steady_state_remove_oldest,
                      allow_twins=True
                      )
        life.initialise(self.ind, population_size=10)

        best = life.evolve(crossover_params={'n_points' : 2})

        self.assertEqual(len(life.population), 10)

    def test_population_steady_state_remove_weakest(self):

        life = Island({'x': 0.6, 'y' : 0.2}, crossover_function=crossover_two_n_point,
                      population_growth_function=population_steady_state_remove_weakest,
                      allow_twins=True
                      )
        life.initialise(self.ind, population_size=10)

        best = life.evolve(crossover_params={'n_points' : 2})

        self.assertEqual(len(life.population), 10)

    def test_population_generational(self):

        life = Island({'x': 0.6, 'y' : 0.2},
                      crossover_function=crossover_two_n_point,
                      population_growth_function=population_generational,
                      allow_twins=True
                      )
        life.initialise(self.ind, population_size=10)

        best = life.evolve(crossover_params={'n_points' : 2},
                           parent_selection_params={'n' : 10},
                           crossover_probability=1.0,)

        self.assertEqual(len(life.population), 10)
