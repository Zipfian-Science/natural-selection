.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _ga-selection-page:

Selection
**************************
.. contents:: Table of Contents

Simple functions for encapsulating performing population selection.

Custom functions
=====================

Custom selection functions can be used in islands.

.. code-block:: python

   def selection_elites_random(individuals : list, n : int = 4, island=None) -> list:
      return random.choice(individuals, size=n).tolist()

Note the following in the above example:

* The function takes a list of individuals
* The function returns a list of individuals
* The functions takes the island as a param
* The custom parameters have default values defined
* All attributes and methods of the Island object are accessible

Important: Parent selection functions ``yield`` lists of individuals instead of returning them.

.. code-block:: python

   def selection_parents_two(individuals : list, n : int = 4, island=None) -> list:
      for parent_1, parent_2 in zip(individuals[::2], individuals[1::2]):
         yield [parent_1, parent_2]


Survivor selection is similar to elite selection.

.. code-block:: python

   def selection_survivors_random(individuals : list, n : int = 4, island=None) -> list:
      return random.choice(individuals, size=n).tolist()

.. _elites-selection:

Elites selection
=====================

Tournament selection
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_elites_tournament

.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_elites_tournament_unique

Random selection
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_elites_random

Top N selection
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_elites_top_n

Parent selection
=====================

Two parents
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_parents_two

Two parents shuffled
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_parents_two_shuffled

N-gram parents
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_parents_n_gram

Survivor selection
=====================

All (default)
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_survivors_all

Random survivors
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.selection.selection_survivors_random