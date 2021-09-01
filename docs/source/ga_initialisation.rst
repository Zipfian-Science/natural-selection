.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _ga-initialisation-page:

Initialisation
**************************
.. contents:: Table of Contents

Simple functions for performing population initialisation.

Custom functions
=====================

To define a custom init function, the following params are required:
- `adam`
- `n`
- `island`

Example:

.. code-block:: python

   def initialise_population_random(adam, n : int = 10, island=None):
      population = list()

      for i in range(n - 1):
         chromosome = island.create_chromosome([x.randomise_new() for x in adam.chromosome])
         eve = island.create_individual(adam.fitness_function, chromosome=chromosome)
         population.append(eve)

      population.append(adam)

      return population

Initialisation
=====================

Random Initialisation
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.initialisation.initialise_population_random

Initialisation with mutation function
-------------------------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.initialise_population_mutation_function
