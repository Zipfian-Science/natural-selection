import unittest
from natural_selection.genetic_programs.primitives import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv, Operator
import natural_selection.genetic_programs.primitives as op
from natural_selection.genetic_programs import Node

class TestNode(unittest.TestCase):

    def test_simple_add(self):
        n_3 = Node(label='3',arity=1,operator=None,terminal_value=3)
        n_39 = Node(arity=1,operator=None,terminal_value=39)
        n = Node(arity=2,operator=OperatorAdd(),children=[n_3,n_39])

        value = n()

        self.assertEquals(value, 42)
        self.assertEquals(str(n), "add(3, 39)")

    def test_simple_sub(self):
        n_42 = Node('42',1,None,42)
        n_2 = Node('2',1,None,2)
        n = Node('N',2,OperatorSub(),None,[n_42,n_2])

        value = n()

        self.assertEquals(value, 40)
        self.assertEquals(str(n), "N(42, 2)")

        n = Node(arity=2, operator=OperatorSub(), terminal_value=None, children=[n_2, n_42])

        value = n()

        self.assertEquals(value, -40)
        self.assertEquals(str(n), "sub(2, 42)")

    def test_simple_mul(self):
        n_1 = Node('2',1,None,2)
        n_2 = Node('2',1,None,2)
        n_3 = Node('2',1,None,2)

        n = Node('N',2,OperatorMul(),None,[n_1,n_2,n_3])

        value = n()

        self.assertEquals(value, 8)


    def test_simple_div(self):
        n_1 = Node('8',1,None,8)
        n_2 = Node('2',1,None,2)
        n_3 = Node('2',1,None,2)

        n = Node('N',2,OperatorDiv(),None,[n_1,n_2,n_3])

        value = n()

        self.assertEquals(value, 2)

    def test_custom_operator(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node('0.5',1,None,0.5)
        n_2 = Node('1.5',1,None,1.5)

        n_3 = Node('Nc',arity=1,operator=operator_cos, terminal_value=None, children=[n_1])
        n_4 = Node(arity=1,operator=operator_cos, terminal_value=None, children=[n_2])

        n_5 = Node('N',1,OperatorAdd(),None,[n_3,n_4])

        value = n_5()

        expected = math.cos(0.5) + math.cos(1.5)

        self.assertEquals(value, expected)
        self.assertEquals(str(n_5), "N(Nc(0.5), cos(1.5))")

    def test_custom_operator_kwargs(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node('x', terminal_value=True)
        n_2 = Node('z', terminal_value=True)

        n_3 = Node(label='cos',arity=1,operator=operator_cos,children=[n_1])
        n_4 = Node(operator=operator_cos, children=[n_2])

        n_5 = Node(operator=OperatorAdd(),children=[n_3,n_4])

        n_6 = Node(terminal_value=2)

        n_7 = Node(operator=OperatorAdd(), children=[n_5,n_6])

        value = n_7(x=0.5,z=1.5)

        expected = math.cos(0.5) + math.cos(1.5) + 2

        self.assertEquals(value, expected)

    def test_str(self):
        import math
        def custom_func(args):
            return math.cos(args[0])

        operator_cos = Operator(operator_label='cos', function=custom_func)

        n_1 = Node(label='x', arity=1, terminal_value=-1)
        n_2 = Node(label='z', arity=1, terminal_value=-1)

        n_3 = Node(operator=operator_cos, children=[n_1])
        n_4 = Node(operator=operator_cos, children=[n_2])

        n_5 = Node('add', 2, OperatorAdd(), None, [n_3, n_4])

        n_6 = Node('2', 1, None, 2)

        n_7 = Node('add', 2, OperatorAdd(), None, [n_5, n_6])

        value = str(n_7)

        self.assertEquals(value, "add(add(cos(x), cos(z)), 2)")

        n_8 = Node(terminal_value=42)

        self.assertEquals(str(n_8), "42")

    def test_return_node(self):
        x = Node(label='x',terminal_value=True)
        n = Node(operator=op.OperatorReturn(), children=[x])

        value = n(x=42)

        self.assertEquals(value, 42)
        self.assertEquals(str(n), "return(x)")