def initialise_population_random(adam, n : int, island=None):
    population = list()

    for i in range(n - 1):
        chromosome = island.create_chromosome([x.randomise_new() for x in adam.chromosome])
        eve = island.create_individual(adam.fitness_function, chromosome=chromosome)
        population.append(eve)

    population.append(adam)

    return population