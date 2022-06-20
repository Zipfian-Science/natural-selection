from datetime import datetime
import unittest
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection import Island
from natural_selection.genetic_algorithms.utils.random_functions import random_int
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_n_point

import time

def fitness(individual, island, x, y):
    time.sleep(1)
    return (individual.chromosome[0].value * x) + (individual.chromosome[1].value *y)

class TestSimpleIsland(unittest.TestCase):

    def setUp(self) -> None:
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        self.ind = Individual(fitness, name="Adam", chromosome=gen)
        self.ind.add_new_property('some_property', 10)

    def test_raise_value(self):
        with self.assertRaises(ValueError):
            life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, core_count=1.1, verbose=False)

        with self.assertRaises(ValueError):
            life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, core_count=-0.5, verbose=False)

        with self.assertRaises(ValueError):
            life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, core_count="-0.5", verbose=False)

    def test_import_migrants(self):
        life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, core_count=1, verbose=False)
        life.initialise(self.ind, population_size=2)
        g_1 = Gene(name="first", value=1, gene_max=25, gene_min=1, randomise_function=random_int)
        g_2 = Gene(name="second", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_3 = Gene(name="third", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_4 = Gene(name="fourth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_5 = Gene(name="fifth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        g_6 = Gene(name="sixth", value=1, gene_max=100, gene_min=1, randomise_function=random_int)
        gen = Chromosome([g_1, g_2, g_3, g_4, g_5, g_6])
        migrants = [life.create_individual(fitness, name=f"alien_{i}", chromosome=gen) for i in range(10)]

        for m in migrants:
            m.chromosome.randomise_all_genes()

        start = datetime.utcnow()
        life.import_migrants(migrants)
        seq_end = datetime.utcnow() - start

        life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, core_count=-1, verbose=False)

        start = datetime.utcnow()
        life.import_migrants(migrants)
        para_end = datetime.utcnow() - start

        print(f'Migrants, Seq: {seq_end} Para: {para_end}')
        self.assertLess(para_end, seq_end)


    def test_evolve(self):
        life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, core_count=1, verbose=False)

        start = datetime.utcnow()
        life.initialise(self.ind, population_size=10)
        seq_end = datetime.utcnow() - start
        start = datetime.utcnow()
        life.evolve(parent_selection_params={'n':8}, crossover_probability=1.0)
        seq_end_2 = datetime.utcnow() - start

        life = Island({'x': 0.6, 'y': 0.2}, crossover_function=crossover_two_n_point, core_count=-1, verbose=False)

        start = datetime.utcnow()
        life.initialise(self.ind, population_size=10)
        para_end = datetime.utcnow() - start
        start = datetime.utcnow()
        life.evolve(parent_selection_params={'n': 8}, crossover_probability=1.0)
        para_end_2 = datetime.utcnow() - start

        self.assertLess(para_end, seq_end)
        self.assertLess(para_end_2, seq_end_2)

        print(f'Init, Seq: {seq_end} Para: {para_end}')
        print(f'Evo, Seq: {seq_end_2} Para: {para_end_2}')




