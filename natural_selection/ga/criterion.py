def criterion_maximise_fitness(desired_minimum, island):
    for i in island.population:
        if i.fitness > desired_minimum:
            return True
    return False

def criterion_minimise_fitness(desired_maximum, island):
    for i in island.population:
        if i.fitness < desired_maximum:
            return True
    return False

def criterion_low_elite_variance(max_std, top_n_individuals, min_genrations, island):
    import copy
    clones = copy.deepcopy(island.population)
    def sortFitness(val):
        return val.fitness

    clones.sort(key=sortFitness, reverse=True)

    elites = clones[0:top_n_individuals]
    elite_fitnesses = [ind.fitness for ind in elites]
    elite_length = len(elites)
    elite_mean = sum(elite_fitnesses) / elite_length
    elite_sum2 = sum(x * x for x in elite_fitnesses)
    elite_std = abs(elite_sum2 / elite_length - elite_mean ** 2) ** 0.5

    if elite_std < max_std and island.generation_count > min_genrations:
        return True

    return False


