import unittest
from natural_selection.genetic_programs.node_operators import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv, Operator, OperatorPow, OperatorEq, OperatorLT
import natural_selection.genetic_programs.node_operators as op
from natural_selection.genetic_programs import Node
from natural_selection.genetic_programs.utils import GeneticProgramError

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

    def test_repr_advance_precedence(self):
        n_1 = Node(label='z', arity=1, is_terminal=True)
        n_2 = Node(label='x', arity=1, is_terminal=True)

        # Add
        n_add_1 = Node(operator=OperatorAdd(), children=[n_2, n_1])
        n_add_2 = Node(operator=OperatorAdd(), children=[n_1, n_2])

        r_add_1, r_add_2 = repr(n_add_1), repr(n_add_2)

        self.assertEqual(r_add_1, r_add_2)

        # Multi
        n_add_1 = Node(operator=OperatorMul(), children=[n_2, n_1])
        n_add_2 = Node(operator=OperatorMul(), children=[n_1, n_2])

        r_add_1, r_add_2 = repr(n_add_1), repr(n_add_2)

        self.assertEqual(r_add_1, r_add_2)

        # Eq
        n_add_1 = Node(operator=OperatorEq(), children=[n_2, n_1])
        n_add_2 = Node(operator=OperatorEq(), children=[n_1, n_2])

        r_add_1, r_add_2 = repr(n_add_1), repr(n_add_2)

        self.assertEqual(r_add_1, r_add_2)

        # Sub
        n_sub_1 = Node(operator=OperatorSub(), children=[n_2, n_1])
        n_sub_2 = Node(operator=OperatorSub(), children=[n_1, n_2])

        r_sub_1, r_sub_2 = repr(n_sub_1), repr(n_sub_2)

        self.assertNotEqual(r_sub_1, r_sub_2)

        # Div
        n_sub_1 = Node(operator=OperatorDiv(), children=[n_2, n_1])
        n_sub_2 = Node(operator=OperatorDiv(), children=[n_1, n_2])

        r_sub_1, r_sub_2 = repr(n_sub_1), repr(n_sub_2)

        self.assertNotEqual(r_sub_1, r_sub_2)

        # Pow
        n_sub_1 = Node(operator=OperatorPow(), children=[n_2, n_1])
        n_sub_2 = Node(operator=OperatorPow(), children=[n_1, n_2])

        r_sub_1, r_sub_2 = repr(n_sub_1), repr(n_sub_2)

        self.assertNotEqual(r_sub_1, r_sub_2)

        # LT
        n_sub_1 = Node(operator=OperatorLT(), children=[n_2, n_1])
        n_sub_2 = Node(operator=OperatorLT(), children=[n_1, n_2])

        r_sub_1, r_sub_2 = repr(n_sub_1), repr(n_sub_2)

        self.assertNotEqual(r_sub_1, r_sub_2)

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

    def test_breadth(self):
        n_1 = Node(terminal_value=2, is_terminal=True)
        n_2 = Node(terminal_value=2, is_terminal=True)

        n_mul = Node(label='N', arity=2, operator=OperatorMul(), children=[n_1, n_2])

        n_add = Node(label='N', arity=2, operator=OperatorAdd(), children=[n_1, n_2])

        n_add = Node(operator=OperatorAdd(), children=[n_mul, n_add])

        value = n_add.breadth(depth=3)

        self.assertEquals(value,4)

        n_add = Node(operator=OperatorAdd(), children=[n_1, n_add])

        value = n_add.breadth(depth=4)

        self.assertEquals(value, 4)

        value = n_add.breadth(depth=3)

        self.assertEquals(value, 2)

        with self.assertRaises(GeneticProgramError):
            value = n_add.breadth(depth=5)

        with self.assertRaises(GeneticProgramError):
            value = n_add.breadth(depth=0)

        # Test more broad

        n_2 = Node(terminal_value=2, is_terminal=True)
        n_mul = Node(operator=OperatorMul(), children=[n_2, n_1])

        n_add = Node(operator=OperatorAdd(max_arity=4), children=[n_mul, n_2, n_2, n_2])

        n_mul = Node(operator=OperatorMul(), children=[n_add, n_2])

        value = n_mul.breadth(depth=3)

        self.assertEqual(value, 4)

        value = n_mul.breadth(depth=4)

        self.assertEqual(value, 2)

        # Test no depth passed

        value = n_mul.breadth()

        self.assertEqual(value, 2)

        # Test max

        value, d = n_mul.max_breadth()

        self.assertEqual(value, 4)
        self.assertEqual(d, 3)

    def test_get_subtree(self):
        n_1 = Node(terminal_value=2, is_terminal=True)
        n_2 = Node(terminal_value=2, is_terminal=True)
        n_mul = Node(operator=OperatorMul(), children=[n_2, n_1])

        n_add = Node(operator=OperatorAdd(max_arity=4), children=[n_mul, n_2, n_2, n_2])

        n_mul = Node(operator=OperatorMul(), children=[n_add, n_2])

        sub_tree = n_mul.get_subtree(depth=3, index=3)

        self.assertIsInstance(sub_tree, Node)

        sub_tree = n_mul.get_subtree(depth=3, index=0)

        self.assertIsInstance(sub_tree.operator, OperatorMul)

        sub_tree = n_mul.get_subtree(depth=4, index=0)

        self.assertEqual(sub_tree.terminal_value, 2)

        with self.assertRaises(GeneticProgramError):
            value = n_mul.get_subtree(depth=4, index=2)

        with self.assertRaises(GeneticProgramError):
            value = n_mul.get_subtree(depth=5, index=1)

    def test_set_subtree(self):
        n_1 = Node(terminal_value=2, is_terminal=True)
        n_2 = Node(terminal_value=1, is_terminal=True)
        n_mul = Node(operator=OperatorMul(), children=[n_2, n_1])

        n_1 = Node(terminal_value=3, is_terminal=True)
        n_2 = Node(terminal_value=4, is_terminal=True)
        n_5 = Node(terminal_value=5, is_terminal=True)

        n_add = Node(operator=OperatorAdd(max_arity=4), children=[n_mul, n_1, n_2, n_5])

        n_mul = Node(operator=OperatorMul(), children=[n_add, n_2])

        n_new_tree = Node(operator=OperatorDiv(), children=[Node(terminal_value=10, is_terminal=True), Node(label='X', is_terminal=True)])

        before_repr = str(n_mul)
        depth_before = n_mul.depth()
        breadth_before = n_mul.breadth(4)
        breadth_before_2 = n_mul.breadth(3)

        with self.assertRaises(GeneticProgramError):
            n_mul.set_subtree(depth=3, index=4, subtree=n_new_tree)

        with self.assertRaises(GeneticProgramError):
            n_mul.set_subtree(depth=5, index=0, subtree=n_new_tree)

        n_mul.set_subtree(depth=3, index=3, subtree=n_new_tree)

        after_repr = str(n_mul)
        depth_after = n_mul.depth()
        breadth_after = n_mul.breadth(4)
        breadth_after_2 = n_mul.breadth(3)

        self.assertNotEqual(before_repr, after_repr)
        self.assertEqual(depth_before, depth_after)
        self.assertNotEqual(breadth_before, breadth_after)
        self.assertEqual(breadth_before_2, breadth_after_2)

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