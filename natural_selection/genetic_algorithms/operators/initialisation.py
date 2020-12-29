def initialise_population_random(adam, n : int, island=None):
    population = list()

    for i in range(n - 1):
        eve = Individual(adam.fitness_function, chromosome=Chromosome([x.randomise_new() for x in adam.chromosome]))
        population.append(eve)

    population.append(adam)

    return population