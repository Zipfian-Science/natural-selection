import unittest
from natural_selection.genetic_programs.node_operators import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv, Operator
import natural_selection.genetic_programs.node_operators as op
from natural_selection.genetic_programs import Node, GeneticProgram

class TestGeneticProgram(unittest.TestCase):

    def test_empty_arg_call(self):
        node = Node(arity=1,operator=None,terminal_value=39, is_terminal=True)
        gp = GeneticProgram(node_tree=node)

        return_value = gp()

        self.assertEqual(return_value, 39)

        node = Node(arity=1, label='X', operator=None, is_terminal=True)

        gp.node_tree = node

        return_value = gp()

        self.assertIsNone(return_value)

        return_value = gp(X=42)

        self.assertEqual(return_value, 42)

        pass