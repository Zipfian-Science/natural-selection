import copy
from typing import Callable

def initialise_population_full_method(adam, n : int, node_tree_create_func : Callable = None, island=None):
    """
    Classic random initialisation function to create a pool of `n` programs from a starting GeneticProgram `adam`.
    This method grows a population of programs each with node trees that extend to `max_depth` for the whole tree.

    Args:
        adam (GeneticProgram): A genetic program already initialised with a node tree.
        n (int): Population size.
        island (Island): Needed to wrap to `create_node` and `create_genetic_program` methods.

    Returns:
        list: Population members.
    """
    population = list()

    for i in range(n - 1):
        if not node_tree_create_func is None:
            node_tree = node_tree_create_func()
            t_func = node_tree_create_func
        else:
            node_tree = adam.tree_generator(growth_mode='full', genetic_program=adam)
            t_func = adam.tree_generator

        eve = island.create_genetic_program(adam.fitness_function,
                                            node_tree=node_tree,
                                            operators=adam.operators,
                                            terminals=adam.terminals,
                                            max_depth=adam.max_depth,
                                            min_depth=adam.min_depth,
                                            growth_mode='full',
                                            terminal_prob=adam.terminal_prob,
                                            tree_generator=t_func,
                                            species_type=adam.species_type,
                                            program_properties=copy.deepcopy(adam.get_properties()))
        population.append(eve)

    population.append(adam)

    return population

def initialise_population_grow_method(adam, n : int, node_tree_create_func : Callable = None, island=None):
    """
    Classic random initialisation function to create a pool of `n` programs from a starting GeneticProgram `adam`.
    Node trees are grown from the root but not necessarily to `max_depth`.

    Args:
        adam (GeneticProgram): A genetic program already initialised with a node tree.
        n (int): Population size.
        island (Island): Needed to wrap to `create_node` and `create_genetic_program` methods.

    Returns:
        list: Population members.
    """
    population = list()

    for i in range(n - 1):
        if not node_tree_create_func is None:
            node_tree = node_tree_create_func()
            t_func = node_tree_create_func
        else:
            node_tree = adam.tree_generator(growth_mode='grow', genetic_program=adam)
            t_func = adam.tree_generator

        eve = island.create_genetic_program(adam.fitness_function,
                                            node_tree=node_tree,
                                            operators=adam.operators,
                                            terminals=adam.terminals,
                                            max_depth=adam.max_depth,
                                            min_depth=adam.min_depth,
                                            growth_mode='grow',
                                            terminal_prob=adam.terminal_prob,
                                            tree_generator=t_func,
                                            species_type=adam.species_type,
                                            program_properties=copy.deepcopy(adam.get_properties()))
        population.append(eve)

    population.append(adam)

    return population