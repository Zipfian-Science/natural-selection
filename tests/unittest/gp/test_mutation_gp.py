import unittest
from natural_selection.genetic_programs.node_operators import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv, Operator
from natural_selection.genetic_programs.operators.mutation import mutation_random_point_change
from natural_selection.genetic_programs import GeneticProgram

import copy

class TestGPCrossover(unittest.TestCase):

    def test_mutation_random_point_change(self):
        operators = [OperatorAdd(operator_label='+'), OperatorSub(operator_label='-')]
        terminals = ['X', 2]

        gp_full = GeneticProgram(operators=operators, terminals=terminals, max_depth=4, min_depth=2, growth_mode='full')

        gp_grow = GeneticProgram(operators=operators, terminals=terminals, max_depth=4, min_depth=2, growth_mode='grow')

        mother_repr = repr(gp_full.node_tree)
        father_repr = repr(gp_grow.node_tree)

        offspring = [mutation_random_point_change(copy.deepcopy(gp_full)), mutation_random_point_change(copy.deepcopy(gp_grow))]

        offspring_1_repr = repr(offspring[0].node_tree)
        offspring_2_repr = repr(offspring[1].node_tree)

        self.assertNotEqual(mother_repr, offspring_1_repr)
        self.assertNotEqual(father_repr, offspring_2_repr)

        mother_repr_after = repr(gp_full.node_tree)
        father_repr_after = repr(gp_grow.node_tree)

        self.assertEqual(mother_repr, mother_repr_after)
        self.assertEqual(father_repr, father_repr_after)

