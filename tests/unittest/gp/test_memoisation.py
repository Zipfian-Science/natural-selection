import unittest
import natural_selection.genetic_programs.node_operators as ops
from natural_selection.genetic_programs import GeneticProgram
from natural_selection.genetic_programs.operators.initialisation import initialise_population_grow_method
from natural_selection.genetic_programs.operators.crossover import crossover_two_one_point
from natural_selection.genetic_programs.operators.mutation import mutation_random_point_change
from natural_selection import Island

import numpy as np


class TestGeneticProgramMemoisation(unittest.TestCase):

    def test_is_deterministic(self):

        operators = [ops.OperatorAdd]
        terminals = ['x', 2]

        gp_symb_first = GeneticProgram(operators=operators,
                                 terminals=terminals,
                                 max_depth=3, min_depth=2, growth_mode='grow',
                                 is_deterministic=True)

        gp_symb_second = GeneticProgram(operators=operators,
                                       terminals=terminals,
                                       max_depth=3, min_depth=2, growth_mode='grow',
                                       is_deterministic=True)

        result = gp_symb_first(x=3)

        result = gp_symb_second(x=3)

        self.assertTrue(True)



