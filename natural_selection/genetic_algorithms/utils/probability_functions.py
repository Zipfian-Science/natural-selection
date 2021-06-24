from numpy import random

def mutation_prob_function_classic(mutation_probability : float = 0.2, island=None) -> float:
    """
    Classic mutation probability function that evaluates whether a random float is less than the mutation probability.

    Args:
        mutation_probability (float): Probability of mutation (default = 0.2).
        island (Island): The Island calling the method (default = None).

    Returns:
        bool: Returns whether to perform mutation.
    """
    if random.uniform(low=0, high=1) < mutation_probability:
        return True
    else:
        return False


def crossover_prob_function_classic(crossover_probability : float = 0.5, island=None):
    """
    Classic crossover probability function that evaluates whether a random float is less than the crossover probability.

    Args:
        crossover_probability (float): Probability of crossover (default = 0.5).
        island (Island): The Island calling the method (default = None).

    Returns:
        bool: Returns whether to perform crossover.
    """
    if random.uniform(low=0, high=1) < crossover_probability:
        return True
    else:
        return False