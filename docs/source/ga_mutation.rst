.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Mutation
**************************
.. contents:: Table of Contents

Simple functions for encapsulating performing mutation operations.

Custom functions
=====================

Custom mutation functions can be used in islands. Like most custom functions, the island is a required param, whether used or not.
The individual is also required, and some default params can be defined. Below an example of the uniform random mutation.

.. code-block:: python

   def mutation_randomize(individual, prob : float = 0.2, island=None):
      for i in range(len(individual.chromosome)):
         if random.random() < prob:
            island._verbose_logging(f"mutate: gene_before {repr(individual.chromosome[i])}")
            individual.chromosome[i].randomise()
            island._verbose_logging(f"mutate: gene_after {repr(individual.chromosome[i])}")

      return individual

Mutation operators
=====================

Uniform Random
---------------------
.. autofunction:: natural_selection.genetic_algorithms.operators.mutation.mutation_randomize
