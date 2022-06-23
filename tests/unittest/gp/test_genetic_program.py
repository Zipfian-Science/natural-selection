import unittest
from natural_selection.genetic_programs.node_operators import OperatorAdd, OperatorSub, OperatorMul, OperatorDiv, Operator
from natural_selection.genetic_programs.utils import GeneticProgramError
from natural_selection.genetic_programs import Node, GeneticProgram, random_generate

class TestGeneticProgram(unittest.TestCase):

    def test_random_generate(self):
        operators = [OperatorAdd, OperatorSub]
        terminals = ['X', 2]

        gp = GeneticProgram()

        gp_full = random_generate(genetic_program=gp, operators=operators, terminals=terminals, max_depth=5, min_depth=1, growth_mode='full')

        # Must be of depth 5 because
        depth = gp_full.node_tree.depth()

        self.assertEqual(depth, 5)

        function_value = gp_full(X=3)

        self.assertIsInstance(function_value, int)

        gp_grow = random_generate(genetic_program=gp, operators=operators, terminals=terminals, max_depth=3, min_depth=1, growth_mode='grow')

        depth = gp_grow.node_tree.depth()

        self.assertLessEqual(depth, 3)

        function_value = gp_grow(X=3)

        self.assertIsInstance(function_value, int)

        gp_grow = random_generate(genetic_program=gp, operators=operators, terminals=terminals, max_depth=4, min_depth=3, growth_mode='grow')

        depth = gp_grow.node_tree.depth()

        self.assertLessEqual(depth, 4)
        self.assertGreaterEqual(depth, 3)

        function_value = gp_grow(X=3)

        self.assertIsInstance(function_value, int)

        gp = GeneticProgram(operators=operators, terminals=terminals, max_depth=4, min_depth=4)

        gp_grow = random_generate(genetic_program=gp, growth_mode='grow')

        depth = gp_grow.node_tree.depth()

        self.assertEqual(depth, 4)


    def test_evaluate(self):
        def func(args):
            return abs(args[0] - args[1]) / (args[0] + args[1])

        custom = Operator(operator_label='custom', function=func, min_arity=2, max_arity=2)

        n1 = Node(label='X', is_terminal=True)
        n2 = Node(label='Y', is_terminal=True)

        n = Node(arity=2, operator=custom, children=[n1, n2])

        def fitness_func(program, island, X, Y):
            return program(X=X, Y=Y)


        gp = GeneticProgram(node_tree=n, fitness_function=fitness_func)

        fitness = gp.evaluate({'X':15, 'Y':26})
        expected_fitness = func([15,26])

        self.assertEqual(fitness, expected_fitness)

        # Test no fitness function given
        gp = GeneticProgram(node_tree=n)

        fitness = gp.evaluate({'X': 16, 'Y': 24})
        expected_fitness = func([16, 24])

        self.assertEqual(fitness, expected_fitness)

        node = Node(arity=1, operator=None, terminal_value=77, is_terminal=True)
        gp = GeneticProgram(node_tree=node)

        fitness = gp.evaluate()

        self.assertEqual(fitness, 77)


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