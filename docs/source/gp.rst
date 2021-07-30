.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _gp-page:

Core GP classes
**************************
.. contents:: Table of Contents

Simple classes for encapsulating Genetic Programs (GP).

Nodes
========================

The most atomic part of a GP experiment is the node. Nodes can be either parents or terminal types.
Terminal nodes can be either a constant value or a variable. Depending on which, the correct parameters need to be set.

.. code-block:: python

   from natural_selection.genetic_programs import Node

Terminal nodes
------------------------

Example of constant terminal node:

.. code-block:: python

   n = Node(is_terminal=True, terminal_value=42)

Where ``is_terminal`` has to be set to true, and the ``terminal_value`` is defined.

When the terminal value is a variable of the function, the label needs to be set. The label is the variable name and is
a global name. Example:

.. code-block:: python

   n = Node(is_terminal=True, label='x')

When the function is called with 'x' as a parameter, the returned terminal value of the node will be replaced with the value.

Operator nodes
------------------------

Operator nodes are parent nodes that execute an operation with children nodes, and return the value(s) to their parents.
Instances of nodes can be executed with parameters.

.. code-block:: python

   from natural_selection.genetic_programs.functions import OperatorAdd

   n_1 = Node(is_terminal=True, label='x')
   n_2 = Node(is_terminal=True, terminal_value=42)

   n_add = Node(arity=2,operator=OperatorAdd(),children=[n_1,n_2])

   value = n_add(x=8) #value = 50 = 8 + 42

Node class
------------------------

.. autoclass:: natural_selection.genetic_programs.__init__.Node
   :members:

GeneticProgram
========================
.. autoclass:: natural_selection.genetic_programs.__init__.GeneticProgram
   :members:
