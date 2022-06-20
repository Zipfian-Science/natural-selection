import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection import Island
from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_choice
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_n_point, crossover_one_binary_union

class TestRandomPointCrossover(unittest.TestCase):

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

class TestMiscellaneousCrossover(unittest.TestCase):

    def test_crossover_one_binary_union(self):

        gen = Chromosome([
            Gene(name=f"gene_1", value=True ,randomise_function=random_choice, choices=[True, False]),
            Gene(name=f"gene_2", value=False, randomise_function=random_choice, choices=[True, False]),
            Gene(name=f"gene_3", value=True, randomise_function=random_choice, choices=[True, False]),
            Gene(name=f"gene_4", value=False, randomise_function=random_choice, choices=[True, False]),
        ])
        gen_2 = Chromosome([
            Gene(name=f"gene_1", value=False, randomise_function=random_choice, choices=[True, False]),
            Gene(name=f"gene_2", value=True, randomise_function=random_choice, choices=[True, False]),
            Gene(name=f"gene_3", value=True, randomise_function=random_choice, choices=[True, False]),
            Gene(name=f"gene_4", value=False, randomise_function=random_choice, choices=[True, False]),
        ])

        ind_1 = Individual(print, name="Adam", chromosome=gen)
        ind_2 = Individual(print, name="Eve", chromosome=gen_2)


        island = Island(verbose=False)

        offspring = crossover_one_binary_union([ind_1, ind_2], island=island)
        expected = [True, True, True, False]
        self.assertListEqual(offspring[0].chromosome.to_list(), expected)


