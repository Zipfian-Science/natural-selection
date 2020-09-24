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
   ga_mating
   ga_selection
   ga_mutation

Starting
=====================
At the command line::

    pip install natural-selection

Then import the tools:

.. code-block:: python

    from natural_selection.ga import Gene, Genome, Individual, Island