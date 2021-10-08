import copy
from typing import Callable

def initialise_population_random(adam, n : int = 10, chromosome_create_func : Callable = None, island=None):
    """
    Classic random initialisation function to create a pool of `n` individuals from a starting Individual `adam`.

    Args:
        adam (Individual): An individual already initialised with a chromosome.
        n (int): Population size (default = 10).
        chromosome_create_func (): A function to create the chromosome of new individuals. If none, simply deep copy adam's. (default = None).
        island (Island): Needed to wrap to `create_chromosome` and `create_individual` methods (default = None).

    Returns:
        list: Population members.
    """
    population = list()

    for i in range(n - 1):
        if not chromosome_create_func is None:
            chromosome = chromosome_create_func()
        else:
            chromosome = island.create_chromosome(genes=[x.randomise_new() for x in adam.chromosome],
                                              gene_verify_func=adam.chromosome.gene_verify_func,
                                              chromosome_properties=copy.deepcopy(adam.chromosome.get_properties()))
        eve = island.create_individual(adam.fitness_function,
                                       chromosome=chromosome,
                                       species_type=adam.species_type,
                                       individual_properties=copy.deepcopy(adam.get_properties()))
        population.append(eve)

    population.append(adam)

    return population

def initialise_population_mutation_function(adam, n : int = 10, chromosome_create_func : Callable = None, mutation_params : dict = None, island=None):
    """
    Random initialisation function to create a pool of `n` individuals from a starting Individual `adam`, but uses the ``island.mutation function`` to perform the randomisation.

    Args:
        adam (Individual): An individual already initialised with a chromosome.
        n (int): Population size (default = 10).
        chromosome_create_func (): A function to create the chromosome of new individuals. If none, simply deep copy adam's. (default = None).
        mutation_params (dict): The params of the mutation function, these usually need to be higher values, such as higher ``prob`` value.
        island (Island): Needed to wrap to `create_chromosome` and `create_individual` methods (default = None).

    Returns:
        list: Population members.
    """

    population = list()

    if mutation_params:
        _mutation_params = mutation_params
    else:
        _mutation_params = {}

    for i in range(n - 1):
        if not chromosome_create_func is None:
            chromosome = chromosome_create_func()
        else:
            chromosome = copy.deepcopy(adam.chromosome)
        eve = island.create_individual(adam.fitness_function,
                                       chromosome=chromosome,
                                       species_type=adam.species_type,
                                       individual_properties=copy.deepcopy(adam.get_properties()))
        eve = island.mutation(island=island, individual=eve, **_mutation_params)
        population.append(eve)

    population.append(adam)

    return population