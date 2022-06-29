.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _changelog-page:

Changelog
**************************

History
==========================

Version 0.2.26 (2022-)
---------------------------

* Major work on expanding package to include Genetic Programming.
* Added tree generation function ``random_generate`` to genetic_programs.
* Renamed ``natural_selection.genetic_programs.functions `` to ``natural_selection.genetic_programs.node_operators ``.
* Added ``is_empty`` to Node class.
* Added ``breadth`` and ``max_breadth`` to Node class, useful for doing crossover.
* Added more history "stamps" to Island for alien spawn, migrant import.
* General work on ``GeneticProgram`` class.
* Added ``create_genetic_program`` to Island for easy wrapper.
* Added ``get_subtree`` to Node class to find a subtree at the given point.
* Added ``set_subtree`` to Node class to set a subtree at the given point.
* Node operators now have a ``strict_precedence`` parameter to solve issues where argument precedence is important.
* Fixed issue with genetic code checks by adding ``force_update`` to both GeneticProgram and Individual.

Version 0.2.25 (2022-06-20)
---------------------------

* Implemented new multiprocessing ability. Can now specify with the ``core_count`` param to split up evaluation over multiple cores.
* Major refactoring. ``Island`` is now imported from the main package, to make it future proof for running genetic programs.


Version 0.2.24 (2022-04-07)
---------------------------

* Added param ``with_replacement`` to selection functions.

Version 0.2.23 (2022-04-05)
---------------------------

* Added a new ability to spawn new aliens at the end of each generation.
* Added ``alien_spawn_function`` to Island initialisation.

Version 0.2.22 (2022-04-03)
---------------------------

* Renamed the ``elite_selection`` to ``parent_selection`` due to misleading name.
* Renamed the original ``parent_selection`` to ``parent_combination`` due to misleading name.
* Removed the ``elite`` in parent selection functions.
* Added ``selection_roulette`` to parent selection.
* Added new ``crossover_one_binary_union`` crossover operator for binary string union.

Version 0.2.21 (2022-03-13)
---------------------------

* Added the ability to add properties to islands, much like with individuals, chromosomes and genes. ``add_new_property``, ``get_properties``, and ``get``.

Version 0.2.20 (2022-03-11)
---------------------------

* Added the param ``maximise_function`` to Island, to either maximise the function (default) or when set to false to minimise the function.

Version 0.2.19 (2021-12-29)
---------------------------

* Fixed minor issue when loading an island and logging not working.

Version 0.2.18 (2021-11-19)
---------------------------

* Print crossover logs after resetting offspring
* Adding all properties of individuals to lineage nodes.
* Added ``get`` to Individuals to safely return properties.
* Custom crossover functions can now add ``gene_inheritance`` to offspring to detail what percentages offspring inherit.

Version 0.2.17 (2021-11-16)
---------------------------

* Added ``write_lineage_json`` to Island to easily dump lineage graph to JSON file.

Version 0.2.16 (2021-11-16)
---------------------------

* Added ``lineage`` to Island, to easily track the lineage of parents/offsrping. This is especially handy for enforcing genetic diversity.
* Added ``write_report`` to Island, for printing generational history to file.
* Logging now writes individuals as strings, not the full representation, thus logs are less cluttered.

Version 0.2.15 (2021-10-08)
---------------------------

* Added ability to give the chromosome creation function through to initialisation ``chromosome_create_func``, overcoming deep copy issues.

Version 0.2.14 (2021-10-02)
---------------------------

* Bug fix in ``create_individual`` of island after adding new init params to Individual class.

Version 0.2.13 (2021-10-02)
---------------------------

* Moved the save checkpoint function from the Island class into utils as ``save_checkpoint_function``.
* Added ``pre_generation_check_function`` to Island ``evolve`` for performing custom pre generation actions.
* Added ``post_generation_check_function`` to Island ``evolve`` for performing custom post generation actions.
* Added ``post_evolution_function`` to Island ``evolve`` for performing custom post evolution actions.

Version 0.2.12 (2021-09-29)
---------------------------

* Added deep copying on randomly creating new gene (to avoid referencing).

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
