from numpy import random

def selection_survivors_all(individuals : list, island=None) -> list:
    """
    Simply passes on the individuals as is.

    Args:
        individuals (list): A list of Individuals.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: All individuals.
    """
    return individuals

def selection_survivors_random(individuals : list, n : int = 4, island=None) -> list:
    """
    Selects a list of random individuals to survive.

    Args:
        individuals (list): A list of Individuals.
        n (int): Number of survivors (default = 4).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Surviving individuals.
    """
    return random.choice(individuals, size=n).tolist()