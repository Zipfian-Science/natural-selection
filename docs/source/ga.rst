.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Core GA classes
**************************
.. contents:: Table of Contents

Simple classes for encapsulating Genetic Algorithms.

Genes
========================

The most atomic part of a GA experiment is the gene. These easily encapsulate the direct encoding
of genes, along with the genome. These are built for direct encoding approaches.

.. code-block:: python

   from natural_selection.genetic_algorithms import Gene

Genes encapsulate at core the key-value of a problem being optimised. They contain further attributes to set constraints
and facilitate randomisation. Genes can further be initialised with custom ``gene_properties`` used by custom ``randomise_function``.

.. code-block:: python

   from natural_selection.genetic_algorithms import Gene
   from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

   # Create a gene
   g_1 = Gene(name="test_int", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
   g_2 = Gene(name="test_real", value=0.5, gene_max=1.0, gene_min=0.1, randomise_function=random_gaussian)

Custom random functions
------------------------

See :ref:`Gene Random functions` for standard random functions.
To implement custom random functions, the following signature is required:

.. code-block:: python

   def my_random_int(gene):
      """
      Random integer from range.

      Args:
        gene (Gene): A gene with a set `MIN` and `MAX`.

      Returns:
        int: Random number.
      """
      return random.randint(low=gene.MIN, high=gene.MAX)

The only passed value is the instance of the gene, so all properties may be accessed. The function can then be used:

.. code-block:: python

   # Example of using custom random function with custom properties
   g_1 = Gene(name="test_int", value=3, randomise_function=my_random_int, gene_properties={'MIN' : 1, 'MAX' : 5})

Gene class
------------------------

.. autoclass:: natural_selection.genetic_algorithms.__init__.Gene
   :members:

Chromosomes
========================

Chromosomes encapsulate ordered lists of genes, with some possibilities to do gene verification on updating.

.. code-block:: python

   from natural_selection.genetic_algorithms import Gene
   from natural_selection.genetic_algorithms import Chromosome
   from natural_selection.genetic_algorithms.utils.random_functions import random_int

   # Create a gene
   g_1 = Gene(name="test_int_1", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
   g_2 = Gene(name="test_int_2", value=3, gene_max=10, gene_min=1, randomise_function=random_int)

   # Create a chromosome

   chromosome = Chromosome([g_1, g_2])

Chromosomes will be used in individuals in the next step.

Gene verification
------------------------

Custom gene verification functions can be added to preform logical checks when changing genes, through crossover.

.. code-block:: python

   def verify_gene_type(gene, loc, chromosome):
      """
      A simple example verification to ensure that the swapped gene is the same type.
      """
      if chromosome[loc].name != gene.name
         return False
      return True

   g_1 = Gene(name="test_int_1", value=3, gene_max=10, gene_min=1, randomise_function=random_int)

   chromosome = Chromosome(genes=[g_1], gene_verify_func=verify_gene_type)

The signature takes the current inserted gene, the index of insertion, and the chromosome instance as input.
If ``False`` is returned, the process raises a :ref:`General GA Error Class` exception.

Chromosome class
------------------------

.. autoclass:: natural_selection.genetic_algorithms.__init__.Chromosome
   :members:

Individuals
========================

An individual fully encapsulates a problem solution, having an initialised chromosome with initialised genes,
a fitness value and a evaluation (fitness) function.

.. code-block:: python

   from natural_selection.genetic_algorithms import Gene
   from natural_selection.genetic_algorithms import Chromosome
   from natural_selection.genetic_algorithms import Individual
   from natural_selection.genetic_algorithms.utils.random_functions import random_int

   # Create a gene
   g_1 = Gene(name="test_int_1", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
   g_2 = Gene(name="test_int_2", value=3, gene_max=10, gene_min=1, randomise_function=random_int)

   # Create a chromosome

   chromosome = Chromosome([g_1, g_2])

   # Next, create an individual to carry these genes and evaluate them
   fitness_function = lambda island, gen, x, y: gen[0].value * x + y
   adam = Individual(fitness_function=fitness_function, name="Adam", chromosome=chromosome)

The fitness function
------------------------

The fitness function is what calculates the fitness of an individual, the function being optimised. The genes (or chromosome)
are the parameters to the function being optimised. Every individual is assigned the same fitness function,
although it is technically possible to give individuals different functions.

Fitness functions are defined by you and they need to have the following signature:

.. code-block:: python

   def my_fitness_function(chromosome, island, ...):
      # function logic
      return fitness

Both ``chromosome`` and ``island`` are required function parameters.
The function parameters are defined in the signature:

.. code-block:: python

   def my_fitness_function(chromosome, island, x, c, d):
      # some random example of a fitness function
      fitness = (x * chromosome[0].value) * (c * chromosome[1].value) + (d * chromosome[2].value)
      return fitness

Individual class
------------------------

.. autoclass:: natural_selection.genetic_algorithms.__init__.Individual
   :members:

Islands
========================

Islands are the main engines that drive the evolutionary process. They can be customised with different
selection, crossover, and mutation operators, giving the experimenter more flexibility when creating experiments.

.. code-block:: python

   from natural_selection.genetic_algorithms import Gene
   from natural_selection.genetic_algorithms import Chromosome
   from natural_selection.genetic_algorithms import Individual
   from natural_selection.genetic_algorithms import Island
   from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

   # Create a gene
   g_1 = Gene(name="test_int", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
   g_2 = Gene(name="test_real", value=0.5, gene_max=1.0, gene_min=0.1, randomise_function=random_gaussian)

   # Create a chromosome

   chromosome = Chromosome([g_1, g_2])

   # Next, create an individual to carry these genes and evaluate them
   fitness_function = lambda island, gen, x, y: gen[0].value * x + y
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

Selection functions
------------------------

There are generally three different selection operations that GA employ:

* Elite selection
* Parent selection
* Survivor selection

Islands are initialised by default with classic selection function, but other functions can be used, in addition to
writing custom selection functions.

By default the ``selection_elites_top_n`` function is used in islands. This can be swapped out for other :ref:`Elites selection`
functions:

.. code-block:: python

   from natural_selection.genetic_algorithms.operators.selection import selection_elites_tournament

   isolated_island = Island(function_params=params, elite_selection=selection_elites_tournament)

All selection functions take ``individuals`` (the population, or list of individuals) and ``island`` as parameters by
default, but may take different or custom inputs. With the above example, ``selection_elites_tournament`` takes extra parameters.
To specify the selection function parameters, a dictionary of values can be passed in the ``evolve`` method.

.. code-block:: python

   from natural_selection.genetic_algorithms.operators.selection import selection_elites_tournament

   esp = {'n' : 4, 'tournament_size' : 5}

   isolated_island.evolve(elite_selection_params=esp)

To read up about different elite selection functions and their parameters, see :ref:`Elites selection`.

Island class
------------------------

.. autoclass:: natural_selection.genetic_algorithms.__init__.Island
   :members: