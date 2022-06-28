import random

def crossover_two_one_point(individuals: list, island=None) -> list:
    """
    Classic One Point crossover.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]

    size = min(mother.depth(), father.depth())

    point = random.randint(1, size - 1)

    mother_breadth = mother.breadth(depth=point)
    father_breadth = father.breadth(depth=point)

    mother[point:], father[point:] = father[point:], mother[point:]

    return [mother, father]