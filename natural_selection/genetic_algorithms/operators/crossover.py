# -*- coding: utf-8 -*-
"""Basic methods for crossover operations.
"""
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2020, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"

import random
import numpy as np

def crossover_two_uniform(individuals : list, prob : float = 0.5, island=None) -> list:
    """
    A Classic crossover function, taking a list of 2 individuals and swapping positional genes based on the `prob` strength of crossover.
    Returns these two individuals with modified genomes.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        prob (float): The probability of swapping genes.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]
    size = min(len(mother.chromosome), len(father.chromosome))
    for i in range(size):
        if random.random() < prob:
            mother.chromosome[i], father.chromosome[i] = father.chromosome[i], mother.chromosome[i]

    return [mother, father]

def crossover_two_one_point(individuals: list, island=None) -> list:
    """
    Classic One Point crossover.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]
    size = min(len(mother.chromosome), len(father.chromosome))

    point = random.randint(1, size - 1)

    mother[point:], father[point:] = father[point:], mother[point:]

    return [mother, father]

def crossover_two_two_point(individuals : list, island=None) -> list:
    """
    Classic Two Point crossover.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]
    size = min(len(mother.chromosome), len(father.chromosome))

    point_1 = random.randint(1, size)
    point_2 = random.randint(1, size - 1)
    if point_2 >= point_1:
        point_2 += 1
    else:
        # Swap the two points
        point_1, point_2 = point_2, point_1

    mother[point_1:point_2], father[point_1:point_2] = father[point_1:point_2], mother[point_1:point_2]

    return [mother, father]

def crossover_two_n_point(individuals : list, n_points : int = 1, prob : float = 0.5, island=None) -> list:
    """
    Classic crossover method to randomly select N points for crossover.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        n_points (int): The amount of random points to split at (default = 1).
        prob (float): The probability of swapping genes.
        island (Island): The Island calling the method (default = None).

    Returns:
        list: Two new Individuals.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]
    size = min(len(mother.chromosome), len(father.chromosome))

    point_cut_list = random.sample(range(1,size-1), n_points)
    point_cut_list.sort()
    point_cut_list.insert(0,0)
    point_cut_list.append(size)
    for i in range(len(point_cut_list)-1):
        if random.random() < prob:
            b = point_cut_list[i]
            e = point_cut_list[i+1]
            mother.chromosome[b:e], father.chromosome[b:e] = father.chromosome[b:e], mother.chromosome[b:e]

    return [mother, father]

def crossover_one_binary_union(individuals : list, inherit_from_mother : bool = True, island=None) -> list:
    """
    Crossover of two binary string chromosomes, producing one offspring where the chromosome is the union the two parents.
    If two parents have the following chromosomes [0, 0, 1] and [1, 0, 1], the child chromosome is [1, 0, 1].
    The gene value must be of type ``bool``. The island must be supplied to create offspring.

    Note:
        See this paper for more,
        Nagae, Satsuki, Shin Kawai, and Hajime Nobuhara.
        "Transfer learning layer selection using genetic algorithm."
        2020 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2020.

    Raises:
        AssertionError: If the gene value is not of type ``bool``.

    Args:
        individuals (list): A list (length of 2) of Individuals to perform crossover.
        inherit_from_mother (bool):  Whether the offspring inherits all properties from the first or the second parent (default = True).
        island (Island): The Island calling the method (default = None).

    Returns:
        list: One new Individual.
    """
    assert len(individuals) > 1, "Not enough individuals given!"
    mother = individuals[0]
    father = individuals[1]

    main_parent = mother
    if not inherit_from_mother:
        main_parent = father

    child_genes = []
    for i in range(min(len(mother.chromosome), len(father.chromosome))):
        t = type(mother.chromosome[i].value)
        assert isinstance(mother.chromosome[i].value, bool) \
               or isinstance(mother.chromosome[i].value, np.bool) \
               or mother.chromosome[i].value in [0,1], "Gene value must be of type bool!"
        assert isinstance(father.chromosome[i].value, bool) \
               or isinstance(father.chromosome[i].value, np.bool) \
               or father.chromosome[i].value in [0,1], "Gene value must be of type bool!"

        if mother.chromosome[i].value or father.chromosome[i].value:
            v = True
        else:
            v = False



        child_gene = island.create_gene(name=main_parent.chromosome[i].name,
                                        gene_min=main_parent.chromosome[i].gene_min,
                                        gene_max=main_parent.chromosome[i].gene_max,
                                        randomise_function=main_parent.chromosome[i].randomise_function,
                                        gene_properties=main_parent.chromosome[i].get_properties(),
                                        step_lower_bound=main_parent.chromosome[i].step_lower_bound,
                                        step_upper_bound=main_parent.chromosome[i].step_upper_bound,
                                        choices=main_parent.chromosome[i].choices,
                                        sig=main_parent.chromosome[i].sig,
                                        mu=main_parent.chromosome[i].mu,
                                        value=v)

        child_genes.append(child_gene)

    chromosome = island.create_chromosome(genes=child_genes,
                                          gene_verify_func=main_parent.chromosome.gene_verify_func,
                                          chromosome_properties=main_parent.chromosome.get_properties())

    child = island.create_individual(fitness_function=main_parent.fitness_function,
                                     chromosome=chromosome,
                                     species_type=main_parent.species_type,
                                     individual_properties=main_parent.get_properties())

    return [child]





