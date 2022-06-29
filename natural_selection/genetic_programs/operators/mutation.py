import random

def mutation_random_point_change(individual, prob : float = None, island=None):
    """
    A Classic mutation function, changes a random subtree of the given individual.

    Args:
        individual (GeneticProgram): Individual object containing a Node tree.
        prob (float): Not used in GP mutation (default = None).
        island (Island): The Island calling the method (default = None).

    Returns:
        GeneticProgram: The newly mutated program.
    """

    tree = individual.node_tree


    tree_depth = tree.depth()

    depth = random.randint(2, tree_depth - 1) if tree_depth > 2 else 2

    breadth = tree.breadth(depth=depth)

    random_point = random.randint(0, breadth - 1)

    max_depth = individual.max_depth - depth + 1

    new_subtree = individual.tree_generator(operators=individual.operators,
            terminals=individual.terminals,
            max_depth=max_depth,
            min_depth=1,
            growth_mode=individual.growth_mode,
            terminal_prob=individual.terminal_prob,
            genetic_program=individual
        )

    tree.set_subtree(depth=depth, index=random_point, subtree=new_subtree)

    return individual