import copy

def clone_classic(individuals : list, island=None):
    """
    Classic cloning function, making a deep copy of population list.

    Args:
        individuals (list): Population members to copy.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        list: Carbon copy of population.
    """
    return copy.deepcopy(individuals)