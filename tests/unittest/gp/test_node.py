import unittest
from natural_selection.genetic_programs.functions import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv, Operator
import natural_selection.genetic_programs.functions as op
from natural_selection.genetic_programs import Node

class TestNode(unittest.TestCase):

    def test_simple_add(self):
        n_3 = Node(label='3',arity=1,operator=None,terminal_value=3, is_terminal=True)
        n_39 = Node(arity=1,operator=None,terminal_value=39, is_terminal=True)
        n = Node(arity=2,operator=OperatorAdd(),children=[n_3,n_39])

        value = n()

        self.assertEquals(value, 42)
        self.assertEquals(str(n), "add(3, 39)")

    def test_simple_sub(self):
        n_42 = Node(terminal_value=42, is_terminal=True)
        n_2 = Node(terminal_value=2, is_terminal=True)
        n = Node(label='N',arity=2,operator=OperatorSub(),children=[n_42,n_2])

        value = n()

        self.assertEquals(value, 40)
        self.assertEquals(str(n), "N(42, 2)")

        n = Node(arity=2, operator=OperatorSub(), children=[n_2, n_42])

        value = n()

        self.assertEquals(value, -40)
        self.assertEquals(str(n), "sub(2, 42)")

    def test_simple_mul(self):
        n_1 = Node(terminal_value=2, is_terminal=True)
        n_2 = Node(terminal_value=2, is_terminal=True)
        n_3 = Node(terminal_value=2, is_terminal=True)

        n = Node(label='N',arity=2,operator=OperatorMul(),children=[n_1,n_2,n_3])

        value = n()

        self.assertEquals(value, 8)

    def test_simple_div(self):
        n_1 = Node(terminal_value=8, is_terminal=True)
        n_2 = Node(terminal_value=2, is_terminal=True)
        n_3 = Node(terminal_value=2, is_terminal=True)

        n = Node(label='N',arity=2,operator=OperatorDiv(),children=[n_1,n_2,n_3])

        value = n()

        self.assertEquals(value, 2)

    def test_custom_operator(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node(arity=1, is_terminal=True, terminal_value=0.5)
        n_2 = Node(arity=1, is_terminal=True, terminal_value=1.5)

        n_3 = Node('Nc',arity=1,operator=operator_cos, children=[n_1])
        n_4 = Node(arity=1,operator=operator_cos, children=[n_2])

        n_5 = Node(label='N',arity=1,operator=OperatorAdd(),children=[n_3,n_4])

        value = n_5()

        expected = math.cos(0.5) + math.cos(1.5)

        self.assertEquals(value, expected)
        self.assertEquals(str(n_5), "N(Nc(0.5), cos(1.5))")

    def test_custom_operator_kwargs(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node('x', is_terminal=True)
        n_2 = Node('z', is_terminal=True)

        n_3 = Node(label='cos',arity=1,operator=operator_cos,children=[n_1])
        n_4 = Node(operator=operator_cos, children=[n_2])

        n_5 = Node(operator=OperatorAdd(),children=[n_3,n_4])

        n_6 = Node(terminal_value=2, is_terminal=True)

        n_7 = Node(operator=OperatorAdd(), children=[n_5,n_6])

        value = n_7(x=0.5,z=1.5)

        expected = math.cos(0.5) + math.cos(1.5) + 2

        self.assertEquals(value, expected)

    def test_str(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node(label='x', arity=1, is_terminal=True)
        n_2 = Node(label='z', arity=1, is_terminal=True)

        n_3 = Node(operator=operator_cos, children=[n_1])
        n_4 = Node(operator=operator_cos, children=[n_2])

        n_5 = Node(label='add', arity=2, operator=OperatorAdd(),children=[n_3, n_4])

        n_6 = Node(label='2', arity=1, is_terminal=True, terminal_value=2)

        n_7 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_5, n_6])

        value = str(n_7)

        self.assertEquals(value, "add(add(cos(x), cos(z)), 2)")

        n_8 = Node(terminal_value=42, is_terminal=True)

        self.assertEquals(str(n_8), "42")

    def test_repr_simple(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node(label='x', arity=1, is_terminal=True)
        n_2 = Node(label='z', arity=1, is_terminal=True)

        n_3 = Node(operator=operator_cos, children=[n_1])
        n_4 = Node(operator=operator_cos, children=[n_2])

        n_5 = Node(label='add', arity=2, operator=OperatorAdd(),children=[n_3, n_4])

        n_6 = Node(label='2', arity=1, is_terminal=True, terminal_value=2)

        n_7 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_5, n_6])

        value = repr(n_7)

        self.assertNotEqual(value, "add(add(cos(x), cos(z)), 2)")
        self.assertEquals(value, 'add(2, add(cos(x), cos(z)))')

        n_8 = Node(terminal_value=42, is_terminal=True)

        self.assertEquals(repr(n_8), "42")

    def test_repr_advance(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node(label='z', arity=1, is_terminal=True)
        n_2 = Node(label='x', arity=1, is_terminal=True)

        n_3 = Node(operator=operator_cos, children=[n_1])
        n_4 = Node(operator=operator_cos, children=[n_2])

        n_5 = Node(label='add', arity=2, operator=OperatorAdd(),children=[n_3, n_4])

        n_6 = Node(label='2', arity=1, is_terminal=True, terminal_value=2)

        n_7 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_5, n_6])

        value_1 = repr(n_7)

        n_1 = Node(label='x', arity=1, is_terminal=True)
        n_2 = Node(label='z', arity=1, is_terminal=True)

        n_3 = Node(operator=operator_cos, children=[n_1])
        n_4 = Node(operator=operator_cos, children=[n_2])

        n_5 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_3, n_4])

        n_6 = Node(label='2', arity=1, is_terminal=True, terminal_value=2)

        n_7 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_5, n_6])

        value_2 = repr(n_7)

        self.assertEquals(value_1, value_2)

        n_1 = Node(label='x', arity=1, is_terminal=True)
        n_2 = Node(arity=1, is_terminal=True, terminal_value=3)

        n_3 = Node(operator=operator_cos, children=[n_1])

        n_5 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_2, n_3])

        n_6 = Node(label='2', arity=1, is_terminal=True, terminal_value=2)

        n_7 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_5, n_6])

        value_1 = repr(n_7)
        str_1 = str(n_7)

        n_1 = Node(label='x', arity=1, is_terminal=True)
        n_2 = Node(arity=1, is_terminal=True, terminal_value=3)

        n_3 = Node(operator=operator_cos, children=[n_1])

        n_5 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_3, n_2])

        n_6 = Node(label='2', arity=1, is_terminal=True, terminal_value=2)

        n_7 = Node(label='add', arity=2, operator=OperatorAdd(), children=[n_6, n_5])

        value_2 = repr(n_7)
        str_2 = str(n_7)

        self.assertEquals(value_1, value_2)
        self.assertNotEqual(str_1, str_2)

    def test_return_node(self):
        x = Node(label='x', is_terminal=True)
        n = Node(operator=op.OperatorReturn(), children=[x])

        value = n(x=42)

        self.assertEquals(value, 42)
        self.assertEquals(str(n), "return(x)")

    def test_depth(self):
        n_1 = Node(terminal_value=2, is_terminal=True)
        n_2 = Node(terminal_value=2, is_terminal=True)

        n_mul = Node(label='N',arity=2,operator=OperatorMul(),children=[n_1,n_2])

        n_3 = Node(terminal_value=5, is_terminal=True)

        n_add = Node(operator=OperatorAdd(), children=[n_mul,n_3])

        value = n_add.depth()

        self.assertEquals(value, 3)

        n_x = Node(label='x', is_terminal=True)
        n_min = Node(operator=OperatorSub(), children=[n_add, n_x])

        value = n_min.depth()

        self.assertEquals(value, 4)

        value = n_1.depth()

        self.assertEquals(value, 1)

        value = n_mul.depth()

        self.assertEquals(value, 2)