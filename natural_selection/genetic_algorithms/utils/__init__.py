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

class GeneticAlgorithmError(Exception):
    """
    Encapsulating graceful exception handling during evolutionary runs.

    Args:
        message (str): Message to print.
        exit (bool): Whether to hard exit the process or not (default = False).
    """

    def __init__(self, message : str, exit : bool = False):
        self.message = message
        if exit:
            print(f"GeneticAlgorithmError: {self.message}")
            quit(1)


    def __str__(self):
        return f"GeneticAlgorithmError: {self.message}"