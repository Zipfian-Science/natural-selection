def mutation_prob_function_classic(mutation_probability : float, island=None) -> float:
    """
    Classic mutation probability function that just returns the given prob value.

    Args:
        mutation_probability (float): Probability of mutation.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        float: Returns mutation_probability.
    """
    return mutation_probability


def crossover_prob_function_classic(crossover_probability : float, island=None):
    """
    Classic crossover probability function that just returns the given prob value.

    Args:
        crossover_probability (float): Probability of crossover.
        island (Island): The Island calling the method (optional, default = None).

    Returns:
        float: Returns crossover_probability.
    """
    return crossover_probability