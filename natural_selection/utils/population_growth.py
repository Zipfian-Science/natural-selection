import copy
import warnings as w

def population_steady_state_remove_weakest(population, offspring, island, desc=True):
    """
    A steady-state method where offspring members replace the weakest population members.

    Args:
        population (list): The current population to add to or replace.
        offspring (list): The offspring members to add.
        island (Island): The calling Island.
        desc (bool): If used outside Island (default = True).

    Returns:
        tuple: The new population, the removed members
    """
    if len(offspring) == 0:
        return population, list()
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
    if len(offspring) == 0:
        return population, list()
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
    if len(population) != len(offspring):
        w.warn(f"WARNING: offspring size {len(offspring)} != population size {len(population)}. Set selection count to population size!")
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
