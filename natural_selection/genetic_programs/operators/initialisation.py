def initialise_population_full_method(adam, n : int, island=None):
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
    raise NotImplementedError('Coming sometime soon')

def initialise_population_grow_method(adam, n : int, island=None):
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
    raise NotImplementedError('Coming sometime soon')