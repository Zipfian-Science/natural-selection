import unittest
import natural_selection.genetic_programs.node_operators as ops
from natural_selection.genetic_programs import GeneticProgram

class TestGeneticProgram(unittest.TestCase):

    def test_island_symbolic_regression(self):

        # Let's try find a simple quadratic

        x = list(range(5))
        ground_truth = list(map(lambda x: x**2, x))

        operators = [ops.OperatorPow, ops.OperatorAdd]
        terminals = ['X', 2]

        gp_symb = GeneticProgram(operators=operators, terminals=terminals, max_depth=3, min_depth=1, growth_mode='grow')




