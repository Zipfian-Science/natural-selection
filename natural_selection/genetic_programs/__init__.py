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

import uuid
from typing import List, Union, Any, Callable
import pickle
import warnings as w

import natural_selection.genetic_programs.node_operators as op
from natural_selection.genetic_programs.utils import GeneticProgramError


class Node:
    """
    Basic class for building Node trees. A node can be either a terminal or a parent.
    When initialised as a terminal node, `is_terminal` has to be set to True and either a `label` or a `terminal_value` has to be set.
    When setting a `terminal_value`, the terminal is a literal, constant value.

    Example: n = Node(is_terminal=True, terminal_value=42).

    On only setting a `label`, the terminal is treated as a variable passed on through the function.

    Example: n = Node(is_terminal=True, label='x').

    Setting the arity is optional for when no children nodes are added.

    Args:
        label (str): Optionally set the label, only used for variable terminals (default = None).
        arity (int): Optionally set the function arity, the norm being 2 for functions (default = 1).
        operator (Operator): If the node is a function, set the operator (default = None).
        is_terminal (bool): Explicitly define if the node is a terminal (default = None).
        terminal_value (Any): Only set if the node is terminal and a constant value (default = None).
        children (list): Add a list of child nodes, list length must match arity (default = None).
    """

    def __init__(self, label : str = None,
                 arity : int = 1,
                 operator : op.Operator = None,
                 is_terminal = False,
                 terminal_value = None,
                 children : List = None):
        if label:
            self.label = label
        elif operator:
            self.label = operator.operator_label
        else:
            self.label = str(terminal_value)
        self.arity = arity
        self.operator = operator
        self.is_terminal = is_terminal
        self.terminal_value = terminal_value
        if children:
            self.children = children
            self.arity = len(children)
        else:
            self.children = [None] * self.arity

    def __call__(self, **kwargs):
        if self.is_terminal:
            if self.label in kwargs.keys():
                return kwargs[self.label]
            return self.terminal_value
        else:
            return self.operator.exec([x(**kwargs) for x in self.children])

    def __str__(self):
        """
        Essentially equivalent to __repr__, but returns a string in the natural order of nodes.
        Where two functionally same trees will return different string representations. Use __repr__ when comparing tree strings.

        Returns:
            str: String representation of tree in natural order of symbols/labels.
        """
        if self.is_terminal:
            return self.label
        else:
            return f"{self.label}({', '.join([str(x) for x in self.children])})"

    def __repr__(self):
        """
        Essentially equivalent to __str__, but more precisely returns an alphabetically sorted str.
        Where two functionally same trees might return different string representations, they will have the exact same __repr__ string.

        Returns:
            str: String representation of tree in alphabetic order of symbols/labels.
        """
        if self.is_terminal:
            return self.label
        else:
            labels = list()
            for n in self.children:
                labels.append(repr(n))
            return f"{self.label}({', '.join(sorted(labels))})"

    def __setitem__(self, index, node):
        if isinstance(index, slice):
            assert index.start < len(self.children), 'Index Out of bounds!'
        else:
            assert index < len(self.children), 'Index Out of bounds!'

        self.children[index] = node

    def __getitem__(self, index):
        if isinstance(index, slice):
            assert index.start < len(self.children), 'Index Out of bounds!'
        else:
            assert index < len(self.children), 'Index Out of bounds!'

        return self.children[index]

    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < len(self.children):
            gene = self.children[self.__n]
            self.__n += 1
            return gene
        else:
            raise StopIteration

    def __len__(self):
        return len(self.children)

    def clear_children(self):
        self.children = [None] * self.arity

    def depth(self):
        """
        Finds the depth of the current tree. This is done by traversing the tree and returning the deepest depth found.

        Returns:
            int: Deepest node depth found.
        """
        if self.is_terminal:
            return 1
        deepest = 0
        for n in self.children:
            d = n.depth() + 1
            if d > deepest:
                deepest = d
        return deepest

class GeneticProgram:
    """
    A class that encapsulates a single genetic program, with node tree and a fitness evaluation function.

    Args:
        fitness_function (Callable): Function with ``func(Node, island, **params)`` signature.
        node_tree (Node): A starting node tree (default = None).
        operators (list): List of all operators that nodes can be constructed from.
        terminals (list): List of all terminals that can be included in the node tree, can be numeric or strings for variables.
        max_depth (int): Maximum depth that node tree can grow.
        name (str): Name for keeping track of lineage (default = None).
        species_type (str) : A unique string to identify the species type, for preventing cross polluting (default = None).
        filepath (str): Skip init and load from a pickled file.
        individual_properties (dict): For fitness functions, extra params may be given (default = None).

    Attributes:
        fitness (Numeric): The fitness score after evaluation.
        age (int): How many generations was the individual alive.
        genetic_code (str): String representation of node tree.
        history (list): List of dicts of every evaluation.
        parents (list): List of strings of parent names.
    """

    def __init__(self,
                 fitness_function: Callable = None,
                 node_tree : Node = None,
                 operators : List[op.Operator] = None,
                 terminals : List[Union[str,int,float]]= None,
                 max_depth : int = None,
                 name: str = None,
                 species_type : str = None,
                 filepath : str = None,
                 individual_properties : dict = None):
        if not filepath is None:
            self.load(filepath=filepath)
            return
        if fitness_function and '<lambda>' in repr(fitness_function):
            w.warn("WARNING: 'fitness_function' lambda can not be pickled using standard libraries.")
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name
        self.operators = operators
        self.terminals = terminals
        self.max_depth = max_depth
        self.node_tree = node_tree
        self.root_node = Node(label=self.name,arity=1, operator=op.OperatorReturn())

        self.fitness_function = fitness_function
        self.fitness = None
        self.age = 0
        self.genetic_code = None
        self.history = list()
        self.parents = list()
        if species_type:
            self.species_type = species_type
        else:
            self.species_type = "def"

        self.__individual_properties = individual_properties
        if individual_properties:
            for k, v in individual_properties.items():
                self.__dict__.update({k: v})

    def __call__(self, **kwargs):
        if self.node_tree:
            self.root_node[0] = self.node_tree
        else:
            raise GeneticProgramError('No node tree has been constructed!')
        return self.root_node(**kwargs)


    def register_parent_names(self, parents: list, reset_parent_name_list: bool = True):
        """
        In keeping lineage of family lines, the names of parents are kept track of.

        Args:
            parents (list): A list of GeneticProgram of the parents.
        """
        if reset_parent_name_list:
            self.parents = list()
        for parent in parents:
            self.parents.append(parent.name)


    def reset_name(self, name: str = None):
        """
        A function to reset the name of a program, helping to keep linage of families.

        Args:
            name (str): Name (default = None).
        """
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name


    def birthday(self, add: int = 1):
        """
        Add to the age. This is for keeping track of how many generations a program has "lived" through.

        Args:
            add (int): Amount to age.
        """
        self.age += add


    def reset_fitness(self, fitness: Any = None, reset_genetic_code: bool = True):
        """
        Reset (or set) the fitness of the program.

        Args:
            fitness (Any): New fitness value (default = None).
            reset_genetic_code (bool): Whether to reset the genetic code. (default = True)
        """
        self.fitness = fitness
        if reset_genetic_code:
            self.genetic_code = None

    def evaluate(self, params : dict, island=None) -> Any:
        """
        Run the fitness function with the given params.

        Args:
            params (dict): Named dict of eval params.
            island (Island): Pass the Island for advanced fitness functions based on Island properties and populations.

        Returns:
            numeric: Fitness value.
        """
        if not params is None:
            _params = params
        else:
            _params = {}
        try:
            self.fitness = self.fitness_function(program=self, island=island, **_params)
        except Exception as exc:
            if island:
                island.verbose_logging(f"ERROR: {self.name} - {repr(self.node_tree)} - {repr(exc)}")
            raise GeneticProgramError(message=f'Could not evaluate program "{self.name}" due to {repr(exc)}')

        stamp = { "name": self.name,
                  "age": self.age,
                  "fitness": self.fitness,
                  "node_tree": str(self.node_tree),
                  "parents": self.parents,
         }

        if island:
            stamp["island_generation" ] = island.generation_count

        self.history.append(stamp)
        return self.fitness

    def unique_genetic_code(self) -> str:
        """
        Gets the unique genetic code, generating if it is undefined.

        Returns:
            str: String name of Chromosome.
        """
        if self.genetic_code is None:
            self.genetic_code = repr(self.node_tree)
        return self.genetic_code

    def save(self, filepath : str):
        """
        Save an individual to a pickle file.

        Args:
            filepath (str): File path to write to.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath : str):
        """
        Load an individual from a pickle file.

        Args:
            filepath (str): File path to load from.
        """
        with open(filepath, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def add_new_property(self, key : str, value : Any):
        """
        Method to add new properties (attributes).

        Args:
            key (str): Name of property.
            value (Any): Anything.
        """
        if self.__individual_properties:
            self.__individual_properties.update({key: value})
        else:
            self.__individual_properties = {key: value}
        self.__dict__.update({key: value})

    def get_properties(self) -> dict:
        """
        Gets a dict of the custom properties that were added at initialisation or the `add_new_property` method.

        Returns:
            dict: All custom properties.
        """
        return self.__individual_properties

    def get(self, key : str, default=None):
        """
        Gets the value of a property or returns default if it doesn't exist.

        Args:
            key (str): Property name.
            default: Value to return if the property is not found (default = None).

        Returns:
            any: The property of the individual.
        """
        if key in self.__dict__.keys():
            return self.__dict__[key]
        return default

    def __str__(self) -> str:
        return f'GeneticProgram({self.name}:{self.fitness})'

    def __repr__(self) -> str:
        genetic_code = self.unique_genetic_code()
        return f'GeneticProgram({self.name}:{self.fitness}:{self.age}:{self.species_type}:{genetic_code})'

    def __eq__(self, other):
        if isinstance(other, GeneticProgram):
            return self.unique_genetic_code() == other.unique_genetic_code()
        else:
            raise GeneticProgramError(message=f'Can not compare {type(other)}')

    def __ne__(self, other):
        if isinstance(other, GeneticProgram):
            return self.unique_genetic_code() != other.unique_genetic_code()
        else:
            raise GeneticProgramError(message=f'Can not compare {type(other)}')

    def __lt__(self, other):
        if isinstance(other, GeneticProgram):
            return self.fitness < other.fitness
        elif isinstance(other, int):
            return self.fitness < other
        elif isinstance(other, float):
            return self.fitness < other
        else:
            raise GeneticProgramError(message=f'Can not compare {type(other)}')

    def __le__(self, other):
        if isinstance(other, GeneticProgram):
            return self.fitness <= other.fitness
        elif isinstance(other, int):
            return self.fitness <= other
        elif isinstance(other, float):
            return self.fitness <= other
        else:
            raise GeneticProgramError(message=f'Can not compare {type(other)}')

    def __gt__(self, other):
        if isinstance(other, GeneticProgram):
            return self.fitness > other.fitness
        elif isinstance(other, int):
            return self.fitness > other
        elif isinstance(other, float):
            return self.fitness > other
        else:
            raise GeneticProgramError(message=f'Can not compare {type(other)}')

    def __ge__(self, other):
        if isinstance(other, GeneticProgram):
            return self.fitness >= other.fitness
        elif isinstance(other, int):
            return self.fitness >= other
        elif isinstance(other, float):
            return self.fitness >= other
        else:
            raise GeneticProgramError(message=f'Can not compare {type(other)}')