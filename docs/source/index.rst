.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Natural Selection
**************************
.. toctree::
   :maxdepth: 3
   :caption: Contents:

   ga
   ga_crossover
   ga_selection
   ga_mutation
   ga_initialisation

   ga_prob_functions
   ga_random_functions
   ga_helper_functions

Evolutionary Algorithm tools in Python
======================================

This is a Python package for creating easy EA experiments, containing easy to use functions and classes for setting up and running Genetic Algorithms.
Future work will include Genetic Programming (GP) as well as Grammatical Evolution (GE).

Starting
=====================
At the command line::

    pip install natural-selection

Then import the tools:

.. code-block:: python

    from natural_selection.genetic_algorithms import Gene, Chromosome, Individual, Island
    from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

Then simply create an experiment:

.. code-block:: python

   from natural_selection.genetic_algorithms import Gene, Chromosome, Individual, Island
   from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

   # Create a gene
   g_1 = Gene(name="test_int", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
   g_2 = Gene(name="test_real", value=0.5, gene_max=1.0, gene_min=0.1, randomise_function=random_gaussian)

   # Add a list of genes to a genome
   gen = Chromosome([g_1, g_2])

   # Next, create an individual to carry these genes and evaluate them
   fitness_function = lambda gen, x, y: gen[0].value * x + y
   adam = Individual(fitness_function, name="Adam", chromosome=gen)

   # Now we can create an island for running the evolutionary process
   # Notice the fitness function parameters are given here.
   params = dict()
   params['x'] = 0.5
   params['y'] = 0.2
   isolated_island = Island(function_params=params)

   # Using a single individual, we can create a new population
   isolated_island.initialise(adam, population_size=5)

   # And finally, we let the randomness of life do its thing: optimise
   best_individual = isolated_island.evolve(n_generations=5)

   # After running for a few generations, we have an individual with the highest fitness
   fitness = best_individual.fitness
   genes = best_individual.chromosome

   for gene in genes:
     print(gene.name, gene.value)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
