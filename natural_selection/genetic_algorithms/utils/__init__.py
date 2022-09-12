import os
from datetime import datetime


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