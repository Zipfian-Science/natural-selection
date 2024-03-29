.. Natural Selection documentation master file, created by
   sphinx-quickstart on Tue Sep 22 22:57:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _island-helper-functions-page:

Island Helper functions
**************************
.. contents:: Table of Contents

Simple helper functions and classes.

Cloning functions
=====================
.. autofunction:: natural_selection.utils.clone_classic

Island checkpoint functions
===========================
.. autofunction:: natural_selection.utils.default_save_checkpoint_function

Pre/post evolution functions
============================
.. autofunction:: natural_selection.utils.post_evolution_function_save_all

Population growth functions
============================

These functions are used to switch between steady-state, generational, and custom population growth methods.

.. autofunction:: natural_selection.utils.population_growth.population_steady_state_remove_weakest

.. autofunction:: natural_selection.utils.population_growth.population_steady_state_remove_oldest

.. autofunction:: natural_selection.utils.population_growth.population_generational

.. autofunction:: natural_selection.utils.population_growth.population_incremental

Operation probability functions
===============================
.. contents:: Table of Contents

Simple functions for performing operator probability checks.

Crossover probability
=====================
.. autofunction:: natural_selection.utils.probability_functions.crossover_prob_function_classic

Mutation probability
=====================
.. autofunction:: natural_selection.utils.probability_functions.mutation_prob_function_classic


Misc
============================
.. autofunction:: natural_selection.utils.get_random_string

.. autofunction:: natural_selection.utils.evaluate_individual_multiproc_wrapper

.. autofunction:: natural_selection.utils.evaluate_individuals_sequentially