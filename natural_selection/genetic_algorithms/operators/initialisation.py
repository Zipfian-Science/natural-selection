def initialise_population_random(adam, n : int = 10, island=None):
    """
    Classic random initialisation function to create a pool of `n` individuals from a starting Individual `adam`.

    Args:
        adam (Individual): An individual already initialised with a chromosome.
        n (int): Population size (default = 10).
        island (Island): Needed to wrap to `create_chromosome` and `create_individual` methods (default = None).

    Returns:
        list: Population members.
    """
    population = list()

    for i in range(n - 1):
        chromosome = island.create_chromosome([x.randomise_new() for x in adam.chromosome])
        eve = island.create_individual(adam.fitness_function, chromosome=chromosome)
        population.append(eve)

    population.append(adam)

    return population