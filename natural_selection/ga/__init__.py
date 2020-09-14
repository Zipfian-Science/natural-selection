# -*- coding: utf-8 -*-
'''
    Base classes for running classic GA experiments
    @author Justin Hocking
    @version 0.0.1
'''
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2020, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"

import random
import uuid
import hashlib
import copy
import multiprocessing as mp

from natural_selection.ga.selection import top_n_selection
from natural_selection.ga.mating import classic_mate_function
from natural_selection.ga.mutation import classic_mutate_function

class Gene:

    def __init__(self, name, value, gene_max, gene_min, rand_type_func):
        self.name = name
        self.value = value
        self.gene_max = gene_max
        self.gene_min = gene_min
        self.rand_type_func = rand_type_func

    def randomize(self):
        return Gene(
            self.name,
            self.rand_type_func(self.gene_min, self.gene_max),
            self.gene_max,
            self.gene_min,
            self.rand_type_func
        )

    def __str__(self) -> str:
        return '({0}:{1})'.format(self.name, self.value)


class Genome:

    def __init__(self, genes: list = None):
        if genes:
            self.genes = genes
        else:
            self.genes = list()

    def append(self, gene: Gene):
        assert isinstance(gene, Gene), 'Must be Gene type!'
        self.genes.append(gene)

    def __setitem__(self, index: int, gene: Gene):
        assert isinstance(gene, Gene), 'Must be Gene type!'
        assert index < len(self.genes), 'Index Out of bounds!'

        self.genes[index] = gene

    def __getitem__(self, index: int) -> Gene:
        assert index < len(self.genes), 'Index Out of bounds!'

        return self.genes[index]

    def __iter__(self):
        self.__n = 0
        return self

    def __next__(self):
        if self.__n < len(self.genes):
            gene = self.genes[self.__n]
            self.__n += 1
            return gene
        else:
            raise StopIteration

    def __len__(self):
        return len(self.genes)

    def __str__(self) -> str:
        return '-'.join(['({0}:{1})'.format(t.name, str(t.value)) for t in self.genes])


class Individual:

    def __init__(self, fitness_function, name=None, genome: Genome = None):
        if name is None:
            name = str(uuid.uuid4())
        if genome is None:
            genome = Genome()
        self.name = name
        self.genome = genome
        self.fitness_function = fitness_function
        self.fitness = None
        self.age = 0
        self.genetic_code = None

    def birthday(self, add=1):
        self.age += add

    def reset_fitness(self, fitness=None, reset_genetic_code=True):
        self.fitness = fitness
        if reset_genetic_code:
            self.genetic_code = None

    def add_gene(self, gene):
        self.genome.append(gene)

    def evaluate(self, params):
        self.fitness = self.fitness_function(self.genome, **params)
        return self.fitness

    def unique_genetic_code(self):
        if self.genetic_code is None:
            self.genetic_code = str(hashlib.md5(''.join(str(x) for x in self.genome).encode()).digest())
        return self.genetic_code

    def __str__(self) -> str:
        return '({0}:{1})'.format(self.name, self.fitness)


class EvolutionIsland:

    def __init__(self, function_params, selection_function=None, mate_function=None, mutate_function=None,
                 crossover_prob_function=None, mutation_prob_function=None, clone_function=None,
                 verbose=True):
        self.function_params = function_params
        self.unique_genome = []
        self.generation_info = []
        self.population = []

        if selection_function:
            self.selection = selection_function
        else:
            self.selection = top_n_selection

        if mate_function:
            self.mate = mate_function
        else:
            self.mate = classic_mate_function

        if mutate_function:
            self.mutate = mutate_function
        else:
            self.mutate = classic_mutate_function

        if crossover_prob_function:
            self.crossover_prob = crossover_prob_function
        else:
            self.crossover_prob = self._crossover_prob_function

        if mutation_prob_function:
            self.mutation_prob = mutation_prob_function
        else:
            self.mutation_prob = self._mutation_prob_function

        if clone_function:
            self.clone = clone_function
        else:
            self.clone = self._clone_function

        self.verbose = verbose

        self.elite_list = []
        self.mutants = []
        self.children = []

    def create(self, adam, seed=42, population_size=8):
        random.seed(seed)

        for i in range(population_size):
            eve = Individual(adam.fitness_function, genome=Genome([x.randomize() for x in adam.genome]))
            self.population.append(eve)

        self.population[0] = adam

        for popitem in self.population:
            popitem.evaluate(self.function_params)

    def import_migrants(self, migrants, reset_fitness=False):
        for i in migrants:
            if not i.unique_genetic_code() in self.unique_genome:
                if i.fitness is None or reset_fitness:
                    i.evaluate(self.function_params)
                self.population.append(i)
                self.unique_genome.append(i.unique_genetic_code())

    def _clone_function(self, island, population):
        return copy.deepcopy(population)

    def _mutation_prob_function(self, island, mutation_probability):
        return mutation_probability

    def _crossover_prob_function(self, island, crossover_probability):
        return crossover_probability

    def evolve(self, starting_generation=0, n_generations=5, crossover_probability=0.5, mutation_probability=0.5,
               mating_params=None, mutation_params=None, selection_params=None, multiproc=False):

        for g in range(starting_generation, starting_generation + n_generations):
            if self.verbose:
                print('[{} started]'.format(g))

            elites = self.selection(self.population, **selection_params)

            elites = self.clone(self, elites)

            self.elite_list.extend(elites)

            for child1, child2 in zip(elites[::2], elites[1::2]):
                if random.random() < self.crossover_prob(self, crossover_probability):
                    self.mate(child1, child2, **mating_params)

                    if self.verbose:
                        print('Mating!')

                    self.children.extend([child1, child2])

            for mutant in elites:
                if random.random() < self.mutation_prob(self, mutation_probability):
                    self.mutate(mutant, **mutation_params)

                    if self.verbose:
                        print('Mutating!')

                    self.mutants.append(mutant)

            invalid_ind = [ind for ind in elites if ind.fitness is None]

            if multiproc:
                cpu_count = mp.cpu_count()
                n = len(invalid_ind)

                manager = mp.Manager()

                fitted_pipelines = manager.list()

                for i in range(0, n, cpu_count):
                    jobs = list()
                    for individual in invalid_ind[i:i + cpu_count]:
                        p = mp.Process(target=individual.evaluate, args=(self.function_params))
                        jobs.append(p)
                        p.start()

                    for proc in jobs:
                        proc.join()
            else:
                for popitem in invalid_ind:
                    popitem.evaluate(self.function_params)

            for popitem in elites:
                if not popitem.unique_genetic_code() in self.unique_genome:
                    self.population.append(popitem)
                    self.unique_genome.append(popitem.unique_genetic_code())

            population_fitnesses = [ind.fitness for ind in self.population]
            elite_fitnesses = [ind.fitness for ind in elites]

            population_length = len(self.population)
            population_mean = sum(population_fitnesses) / population_length
            population_sum2 = sum(x * x for x in population_fitnesses)
            population_std = abs(population_sum2 / population_length - population_mean ** 2) ** 0.5

            self.generation_info.append(
                {
                    'stat': 'population',
                    'n': g,
                    'pop_len': population_length,
                    'fitness_mean': population_mean,
                    'fitness_std': population_std,
                    'fitness_min': min(population_fitnesses),
                    'fitness_max': max(population_fitnesses),
                }
            )

            elite_length = len(elites)
            elite_mean = sum(elite_fitnesses) / elite_length
            elite_sum2 = sum(x * x for x in elite_fitnesses)
            elite_std = abs(elite_sum2 / elite_length - elite_mean ** 2) ** 0.5

            self.generation_info.append(
                {
                    'stat': 'elites',
                    'n': g,
                    'pop_len': elite_length,
                    'fitness_mean': elite_mean,
                    'fitness_std': elite_std,
                    'fitness_min': min(elite_fitnesses),
                    'fitness_max': max(elite_fitnesses),
                }
            )

            if self.verbose:
                print('Generation fitness:')
                print('\t= Min %s' % min(population_fitnesses))
                print('\t= Max %s' % max(population_fitnesses))
                print('\t= Avg %s' % population_mean)
                print('\t= Std %s' % population_std)
                print('')
                print('Elite fitness:')
                print('\t= Min %s' % min(elite_fitnesses))
                print('\t= Max %s' % max(elite_fitnesses))
                print('\t= Avg %s' % elite_mean)
                print('\t= Std %s' % elite_std)

            for i in self.population:
                i.birthday()

        best_ind = self.selection(self.population, 1)[0]

        if self.verbose:
            print("Best individual is %s, %s" % (best_ind.name, best_ind.fitness))

        return best_ind
