import unittest
import natural_selection.genetic_programs.node_operators as ops
from natural_selection.genetic_programs import GeneticProgram
from natural_selection.genetic_programs.operators.initialisation import initialise_population_grow_method
from natural_selection.genetic_programs.operators.crossover import crossover_two_one_point
from natural_selection.genetic_programs.operators.mutation import mutation_random_point_change
from natural_selection import Island

import numpy as np


def fitness_function(program, island, X, Y):
    try:
        Y_pred = [program(x=x) for x in X]

        f = np.square(np.subtract(Y, Y_pred)).mean()

        f = f if f > 0 else 999999
    except Exception as ex:
        print(ex)
        f = 999999

    return f

class TestGeneticProgram(unittest.TestCase):

    def test_island_symbolic_regression(self):

        # Let's try find a simple quadratic

        x = list(range(5))
        ground_truth = list(map(lambda x: x**2 + x, x))





        operators = [ops.OperatorPow, ops.OperatorAdd]
        terminals = ['x', 2]

        gp_symb = GeneticProgram(fitness_function=fitness_function, operators=operators, terminals=terminals, max_depth=3, min_depth=2, growth_mode='grow')


        island = Island(function_params={'X':x,'Y':ground_truth},
                        maximise_function=False,
                        initialisation_function=initialise_population_grow_method,
                        crossover_function=crossover_two_one_point,
                        mutation_function=mutation_random_point_change,
                        core_count=1,
                        verbose=False)
        island.initialise(gp_symb, population_size=4)

        best = island.evolve(mutation_probability=0.1, n_generations=50)

        self.assertIsInstance(best, GeneticProgram)

        Y_pred = [best(x=v) for v in x]

        self.assertEqual(len(Y_pred), len(ground_truth))



