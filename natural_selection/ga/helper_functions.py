import copy
from natural_selection.ga import Island

def clone_function_classic(population : list, island : Island = None):
    """
    Classic cloning function, making a deep copy of population list.

    Args:
        population (list): Population members to copy.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        list: Carbon copy of population.
    """
    return copy.deepcopy(population)