.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _changelog-page:

Changelog
**************************

History
==========================

Version 0.2.6 (2021-08-31)
--------------------------

* Renamed ``force_genetic_diversity`` to ``allow_twins``
* Added new tournament selection function of unique only selection, see ``selection_elites_tournament_unique``
* Fixed bug in ``crossover_two_n_point`` where crossover was just swapping genes, and effectively not creating true offspring
* Logging offspring stats too
* Now individuals and chromosomes can have custom properties easily added

Version 0.2.5 (2021-08-18)
--------------------------

* Fitness function now takes the individual instead of chromosome
* Fitness not reset for offspring
* Documentation extended
