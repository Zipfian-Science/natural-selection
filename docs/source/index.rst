.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Natural Selection
**************************
.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Contents:

   ga_main_page
   gp_main_page

Evolutionary Algorithm tools in Python
======================================

A Python package for creating easy EA experiments, containing easy to use functions and classes for setting up and
running Genetic Algorithms and Genetic Programs. Natural Selection is built on minimal dependencies, only requiring
``numpy`` for random functions.

Starting
=====================

Installation
---------------------

Using pip::

    pip install natural-selection

Usage
---------------------

Import the tools:

.. code-block:: python

    from natural_selection import Island
    from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
    from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

Then simply create a GA experiment:

.. code-block:: python

   from natural_selection import Island
   from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
   from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

   # Create a gene
   g_1 = Gene(name="test_int", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
   g_2 = Gene(name="test_real", value=0.5, gene_max=1.0, gene_min=0.1, randomise_function=random_gaussian)

   # Add a list of genes to a genome
   gen = Chromosome([g_1, g_2])

   # Next, create an individual to carry these genes and evaluate them
   fitness_function = lambda island, individual, x, y: individual.chromosome[0].value * x + individual.chromosome[0].value * y
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

Islands
========================

Islands are the main engines that drive the evolutionary process. They can be customised with different
selection, crossover, and mutation operators, giving the experimenter more flexibility when creating experiments.

.. code-block:: python

   from natural_selection.genetic_algorithms import Gene
   from natural_selection.genetic_algorithms import Chromosome
   from natural_selection.genetic_algorithms import Individual
   from natural_selection import Island
   from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

   # Create a gene
   g_1 = Gene(name="test_int", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
   g_2 = Gene(name="test_real", value=0.5, gene_max=1.0, gene_min=0.1, randomise_function=random_gaussian)

   # Create a chromosome

   chromosome = Chromosome([g_1, g_2])

   # Next, create an individual to carry these genes and evaluate them
   fitness_function = lambda island, individual, x, y: individual.chromosome[0].value * x + individual.chromosome[1].value * y
   adam = Individual(fitness_function=fitness_function, name="Adam", chromosome=chromosome)

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

Islands are customisable in how they operate on population members.

Island class
------------------------

.. autoclass:: natural_selection.__init__.Island
   :members:

Utils
---------------------

See :ref:`island-helper-functions-page` for helper tools.

Changes and history
---------------------

See :ref:`changelog-page` for version history.

Version 0.2.27 (2022-09-02):

* A ``population_evaluation_function`` can now be added to define custom fitness function calls to individuals.
* This makes the way open for multi server fitness evaluation.
* Added ``natural_selection.utils``.
* Moved ``clone_classic`` to ``natural_selection.utils``.
* Moved ``get_random_string`` to ``natural_selection.utils``.
* Moved ``default_save_checkpoint_function`` to ``natural_selection.utils``.
* Moved ``post_evolution_function_save_all`` to ``natural_selection.utils``.
* Added ``evaluate_individual_multiproc_wrapper`` to ``natural_selection.utils``.
* Added ``evaluate_individuals_sequentially`` to ``natural_selection.utils``.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
