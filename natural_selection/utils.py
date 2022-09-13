import copy
import os
from datetime import datetime

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

def get_random_string(length : int = 8, include_numeric=False) -> str:
    """
    Generate a random string with a given length. Used mainly for password generation.
    Args:
        length (int): Length to generate.
        include_numeric (bool): Include numbers?

    Returns:
        str: Random character string.
    """
    import random
    import string

    letters = string.ascii_letters
    if include_numeric:
        letters = f'{letters}{string.digits}'
    return ''.join(random.choice(letters) for i in range(length))

def default_save_checkpoint_function(event, island):
    """
    Default checkpoint saving function. Used to save the Island periodically after certain events.

    Args:
        event (str): The evolutionary event (crossover, eval etc).
        island (Island): The island to save.

    """
    if not os.path.isdir(island.checkpoints_dir):
        os.mkdir(island.checkpoints_dir)
    if not os.path.isdir(f'{island.checkpoints_dir}/{island.name}'):
        os.mkdir(f'{island.checkpoints_dir}/{island.name}')
    fp = f'{island.checkpoints_dir}/{island.name}/{datetime.utcnow().strftime("%H-%M-%S")}_{event}_checkpoint.pkl'
    island.verbose_logging(f'checkpoint: file {fp}')
    island.save_island(fp)

def post_evolution_function_save_all(island):
    """
    Simple function to save (pickle) every population member after completing evolution.

    Args:
        island (Island): Island to get ``population`` from.
    """
    for p in island.population:
        p.save_individual(filepath=f'individual_{p.name}.pkl')

def evaluate_individual_multiproc_wrapper(individual, island, params):
    """
    A wrapper function for spawning multi processes.

    Args:
        individual: Single individual to evaluate.
        island: The calling island
        params: Function eval params.

    Returns:
        numeric : the fitness value.
    """
    island.verbose_logging(f"eval: {str(individual)}")
    return individual.evaluate(params=params, island=island)

def evaluate_individuals_sequentially(individuals, island, params):
    """
    A simple function that wraps the sequential evaluation of population members, all locally.

    Args:
        individuals (list): The individuals to evaluate.
        island (Island): The calling island.
        params (dict): The evaluation parameters.

    Returns:

    """
    for individual in individuals:
        island.verbose_logging(f"eval: {str(individual)}")
        individual.evaluate(island=island, params=params)
    return individuals

def population_steady_state_remove_weakest(population, offspring, island, desc=True):
    """
    A steady-state method where offspring members replace weakest population members.

    Args:
        population (list): The current population to add to or replace.
        offspring (list): The offspring members to add.
        island (Island): The calling Island.
        desc (bool): If used outside Island (default = True).

    Returns:
        tuple: The new population, the removed members
    """
    cloned = copy.deepcopy(population)
    def sortFitness(val):
        return val.fitness

    if island:
        cloned.sort(key=sortFitness, reverse=island.maximise_function)
    else:
        cloned.sort(key=sortFitness, reverse=desc)

    deaths = list(cloned[-len(offspring):])
    cloned[-len(offspring):] = offspring
    return cloned, deaths

def population_steady_state_remove_oldest(population, offspring, island):
    """
    A steady-state method where offspring members replace the oldest members.

    Args:
        population (list): The current population to add to or replace.
        offspring (list): The offspring members to add.
        island (Island): The calling Island.

    Returns:
        tuple: The new population, the removed members
    """
    cloned = copy.deepcopy(population)
    def sortAge(val):
        return val.age

    cloned.sort(key=sortAge, reverse=False)

    deaths = list(cloned[-len(offspring):])
    cloned[-len(offspring):] = offspring
    return cloned, deaths

def population_generational(population, offspring, island):
    """
    The generational method where the offspring replaces the full population.

    Args:
        population (list): The current population to add to or replace.
        offspring (list): The offspring members to add.
        island (Island): The calling Island.

    Returns:
        tuple: The new population, the removed members
    """
    return offspring, copy.deepcopy(population)

def population_incremental(population, offspring, island):
    """
    A simple incremental run where offspring members are added to the population.

    Args:
        population (list): The current population to add to or replace.
        offspring (list): The offspring members to add.
        island (Island): The calling Island.

    Returns:
        tuple: The new population, the removed members
    """
    new_population = copy.deepcopy(population)
    new_population.extend(offspring)
    return new_population, list()
