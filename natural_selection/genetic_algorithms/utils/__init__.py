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