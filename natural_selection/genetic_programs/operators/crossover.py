import random

def crossover_two_one_point(individuals: list, island=None) -> list:
    """
    Classic One Point crossover.

    Args:
        individuals (list): A list (length of 2) of Programs to perform crossover.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0].node_tree
    father = individuals[1].node_tree

    mother_depth = mother.depth()
    father_depth = father.depth()

    assert mother_depth > 1, f"{individuals[0].name} has a depth of {mother_depth}"
    assert father_depth > 1, f"{individuals[1].name} has a depth of {father_depth}"

    size = min(mother_depth, father_depth)

    point = random.randint(2, size - 1) if size > 2 else 2

    mother_breadth = mother.breadth(depth=point)
    father_breadth = father.breadth(depth=point)

    random_point_mother = random.randint(0, mother_breadth - 1)
    random_point_father = random.randint(0, father_breadth - 1)

    mother_sub_tree = mother.get_subtree(depth=point, index=random_point_mother)

    father_sub_tree = father.get_subtree(depth=point, index=random_point_father)

    mother.set_subtree(depth=point, index=random_point_mother, subtree=father_sub_tree)

    father.set_subtree(depth=point, index=random_point_father, subtree=mother_sub_tree)

    return [individuals[0], individuals[1]]