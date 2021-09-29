.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _changelog-page:

Changelog
**************************

History
==========================

Version 0.2.12 (2021-09-29)
---------------------------

* Added deep copying on randomly creating new gene (to avoid referencing)

Version 0.2.11 (2021-09-24)
---------------------------

* Fixed flaw where chromosome and individual properties aren't being copied with initialisation.
* Renamed ``_verbose_logging`` to ``verbose_logging`` to publicly expose.

Version 0.2.10 (2021-09-08)
---------------------------

* Fixed bug in ``name`` param of Island.
* Removed the need to pass a dict of params to ``evaluate`` function of individuals

Version 0.2.9 (2021-09-01)
--------------------------

* Fixed major bug in ``initialise_population_mutation_function``, due to chromosomes not being copied

Version 0.2.8 (2021-09-01)
--------------------------

* Added new randomise function: ``mutation_randomize_n_point``
* Added new initialisation function ``initialise_population_mutation_function`` to use the defined mutation function

Version 0.2.7 (2021-08-31)
--------------------------

* Fixed major bug in not adding new offspring due to genetic code not being reset

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
