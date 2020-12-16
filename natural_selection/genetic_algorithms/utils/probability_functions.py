import random

def mutation_prob_function_classic(mutation_probability : float = 0.2, island=None) -> float:
    """
    Classic mutation probability function that evaluates whether a random float is less than the mutation probability.

    Args:
        mutation_probability (float): Probability of mutation.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        bool: Returns whether to perform mutation.
    """
    if random.random() < mutation_probability:
        return True
    else:
        return False


def crossover_prob_function_classic(crossover_probability : float = 0.5, island=None):
    """
    Classic crossover probability function that evaluates whether a random float is less than the crossover probability.

    Args:
        crossover_probability (float): Probability of crossover.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        bool: Returns whether to perform crossover.
    """
    if random.random() < crossover_probability:
        return True
    else:
        return False