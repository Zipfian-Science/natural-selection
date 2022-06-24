import unittest
from natural_selection.genetic_programs.node_operators import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv, Operator
import natural_selection.genetic_programs.node_operators as op
from natural_selection.genetic_programs import Node, GeneticProgram

class TestNodeOperators(unittest.TestCase):

    def test_custom_operator_with_function(self):
        def func(args):
            return abs(args[0] - args[1]) / (args[0] + args[1])

        custom = Operator(operator_label='custom',function=func,min_arity=2,max_arity=2)


        returned_value = custom.exec([15,20])
        expected_value = abs(15 - 20 ) / (15 + 20)

        self.assertEqual(returned_value,expected_value)

        with self.assertRaises(AssertionError):
            returned_value = custom.exec([15,20,50])

        with self.assertRaises(AssertionError):
            returned_value = custom.exec([15])

        n1 = Node(label='X', is_terminal=True)
        n2 = Node(label='Y', is_terminal=True)

        n = Node(arity=2,operator=custom, children=[n1, n2])

        X = 16
        Y = 19

        expected_value = abs(X - Y) / (X + Y)
        returned_value = n(X=X,Y=Y)

        self.assertEqual(returned_value, expected_value)

