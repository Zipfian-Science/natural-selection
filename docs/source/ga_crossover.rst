.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _ga-crossover-page:

Crossover operations
**************************
.. contents:: Table of Contents

Simple functions for performing crossover operations.

Custom crossover
=====================

Crossover typically involves two parents and a crossover strength. Custom functions can be written and added to the Island.

All crossover functions have to take as input at least a list of individuals and the island, even if the island isn't used.
Custom functions also have to return a list of individuals, and all custom parameters should have default values defined.

.. code-block:: python

   def crossover_custom(individuals : list, min_fitness : float = 0.8,
         max_attempts : int = 3, island = None):
      import copy
      assert len(individuals) > 1, "Not enough individuals given!"

      mother = individuals[0]
      father = individuals[1]
      other = island.create_individual(mother.fitness_function, chromosome=copy.deepcopy(mother.chromosome))

      attempts = 0

      while other.evaluate(island.function_params) < min_fitness and attempts < max_attempts:
         for i in len(mother.chromosome):
            mate = random.choice([mother, father], size=1)[0]
            other.chromosome[i] = mate.chromosome[i]
         attempts += 1
      return [other]

Note the following in the above example:

* The function takes a list of individuals
* The function returns a list of individuals
* The functions takes the island as a param
* The custom parameters have default values defined
* All attributes and methods of the Island object are accessible

Crossover operators
=====================

Uniform Random
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.crossover.crossover_two_uniform

Random Point
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.crossover.crossover_two_one_point

.. autofunction:: natural_selection.genetic_algorithms.operators.crossover.crossover_two_two_point

.. autofunction:: natural_selection.genetic_algorithms.operators.crossover.crossover_two_n_point

Miscellaneous
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.crossover.crossover_one_binary_union