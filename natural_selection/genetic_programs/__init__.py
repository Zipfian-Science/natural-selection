# -*- coding: utf-8 -*-
"""Basic classes for running Genetic Algorithms.
"""
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2021, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"

from typing import List

import natural_selection.genetic_programs.primitives as op

class Node:

    def __init__(self, label : str = None,
                 arity : int = 1,
                 operator : op.Operator = None,
                 terminal_value = False,
                 children : List = None):
        if label:
            self.label = label
        elif operator:
            self.label = operator.operator_label
        else:
            self.label = str(terminal_value)
        self.arity = arity
        self.operator = operator
        self.terminal_value = terminal_value
        if children:
            self.children = children
        else:
            self.children = [None] * self.arity

    def __call__(self, **kwargs):
        if self.terminal_value:
            if self.label in kwargs.keys():
                return kwargs[self.label]
            return self.terminal_value
        else:
            return self.operator.exec([x(**kwargs) for x in self.children])

    def __str__(self):
        if self.terminal_value:
            return self.label
        else:
            return f"{self.label}({', '.join([str(x) for x in self.children])})"

    def add_child(self, child, position : int = 0):
        assert isinstance(child, Node), 'Must be Node type!'
        assert position < len(self.children), 'Index Out of bounds!'
        self.children[position] = child

    def clear_children(self):
        self.children = [None] * self.arity

class GeneticProgram:

    def __init__(self,
                 name : str,
                 operators : List[op.Operator],
                 terminals : List,
                 max_depth : int):
        self.name = name
        self.operators = operators
        self.terminals = terminals
        self.max_depth = max_depth
        self.root_node = Node(label=name,arity=1, operator=op.OperatorReturn())

    def __call__(self, **kwargs):
        return self.root_node(**kwargs)

    def create_node_with_children(self, parent: Node, current_depth: int) -> Node:
        current_depth += 1
        for i in range(parent.arity):
            child = Node()
            if current_depth == self.max_depth:
                child.set_label(self.random_generator.choice(self.terms))
                child.set_arity(0)
            else:
                if parent.get_label() == "if" and i == 0:
                    child.set_label(self.random_generator.choice(self.relational_operators))
                    child.set_arity(2)
                else:
                    if self.random_generator.random() > 0.5:
                        child.set_label(self.random_generator.choice(self.terms))
                        child.set_arity(0)
                    else:
                        self.set_node_label_and_arity(child, current_depth, self.max_depth)
            parent.add_child(self.create_node_with_children(child, current_depth))
        return parent

    def create_node(self) -> Node:
        pass

