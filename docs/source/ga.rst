.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Genetic Algorithms
**************************
.. contents:: Table of Contents

Simple classes for encapsulating Genetic Algorithms.

Genes
========================

The most atomic part of a GA experiment is the gene. These easily encapsulate the direct encoding
of genes, along with the genome. These are built for direct encoding approaches.

.. autoclass:: natural_selection.ga.__init__.Gene
   :members:

Genomes
========================
.. autoclass:: natural_selection.ga.__init__.Genome
   :members:

Individuals
========================
.. autoclass:: natural_selection.ga.__init__.Individual
   :members:

Islands
========================

Islands are the main evolutionary engines that drive the process. The can be customised with different
selection, crossover, and mutation functions, giving the experimentor more flexibility when creating experiments.

.. autoclass:: natural_selection.ga.__init__.Island
   :members: