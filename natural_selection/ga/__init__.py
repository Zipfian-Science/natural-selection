# -*- coding: utf-8 -*-
"""Basic classes for running Genetic Algorithms.
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
import uuid
from typing import Callable, Any


from natural_selection.ga.selection import selection_function_classic, selection_function_parents_classic
from natural_selection.ga.crossover import crossover_function_classic
from natural_selection.ga.mutation import mutation_function_classic
from natural_selection.ga.prob_functions import crossover_prob_function_classic, mutation_prob_function_classic
from natural_selection.ga.helper_functions import clone_function_classic

class Gene:
    """
    A simple class to encapsulate a simple gene.

    Args:
        name (str): Gene name. The gene name also acts as a compatibility reference.
        value (Any): The value, could be any type.
        gene_max (Any, numeric type): Max value or None.
        gene_min (Any, numeric type): Min value or None.
        rand_type_func (Callable): A function to randomise the gene, taking the min and max as input with signature ``func(gene_min, gene_max)``.

    """

    def __init__(self, name : str, value : Any, gene_max : Any, gene_min : Any, rand_type_func: Callable):
        self.name = name
        self.value = value
        self.gene_max = gene_max
        self.gene_min = gene_min
        self.rand_type_func = rand_type_func

    def randomize(self):
        """
        Sets a random value gene with randomised value.
        """
        self.value = self.rand_type_func(self.gene_min, self.gene_max)

    def randomize_new(self):
        """
        Return a new gene with randomised value.

        Returns:
            Gene: Newly created gene.
        """
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
    """
    A class that encapsulates an ordered sequence of Gene objects.

    Args:
        genes (list): list of initialised Gene objects.
    """
    def __init__(self, genes: list = None):
        if genes:
            self.genes = genes
        else:
            self.genes = list()

    def append(self, gene: Gene):
        """
        Simple appending of Gene type objects.

        Args:
            gene (Gene): Gene
        """
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
    """
    A class that encapsulates a single individual, with genetic code and a fitness evaluation function.

    Args:
        fitness_function (Callable): Function with ``func(genome, island, **params)`` signature
        name (str): Name for keeping track of lineage (default = None).
        genome (Genome): A Genome object, initialised (default = None).
        species_type (str) : A unique string to identify the species type, for preventing cross polluting (default = None).
    """

    def __init__(self, fitness_function : Callable, name : str = None, genome: Genome = None, species_type : str = None):
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name
        if genome is None:
            self.genome = Genome()
        else:
            self.genome = genome
        self.fitness_function = fitness_function
        self.fitness = None
        self.age = 0
        self.genetic_code = None
        self.history = list()
        self.parents = list()
        if species_type:
            self.species_type = species_type
        else:
            self.species_type = "DEFAULT_SPECIES"

    def register_parent_names(self, parents : list, reset_parent_name_list : bool = True):
        """
        In keeping lineage of family lines, the names of parents are kept track of.

        Args:
            parents (list): A list of Individuals of the parents.
        """
        if reset_parent_name_list:
            self.parents = list()
        for parent in parents:
            self.parents.append(parent.name)

    def reset_name(self, name : str = None):
        """
        A function to reset the name of an individual, helping to keep linage of families.

        Args:
            name (str): Name (default = None).
        """
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name

    def birthday(self, add : int = 1):
        """
        Add to the age. This is for keeping track of how many generations an individual has "lived" through.

        Args:
            add (int): Amount to age.
        """
        self.age += add

    def reset_fitness(self, fitness : Any = None, reset_genetic_code : bool = True):
        """
        Reset (or set) the fitness oof the individual.

        Args:
            fitness (Any): New fitness value (default = None).
            reset_genetic_code (bool): Whether to reset the genetic code. (default = True)
        """
        self.fitness = fitness
        if reset_genetic_code:
            self.genetic_code = None

    def add_gene(self, gene : Gene):
        """
        Appends a gene to the genome.

        Args:
            gene (Gene): Gene to add.
        """
        self.genome.append(gene)

    def evaluate(self, params : dict, island=None) -> Any:
        """
        Run the fitness function with the given params.

        Args:
            params (dict): Named dict of eval params.
            island (Island): Pass the Island for advanced fitness functions based on Island properties and populations.

        Returns:
            numeric: Fitness value.
        """
        self.fitness = self.fitness_function(genome=self.genome, island=island, **params)
        self.history.append({"name" : self.name, "age" : self.age, "fitness" : self.fitness, "genome" : self.unique_genetic_code})
        return self.fitness

    def unique_genetic_code(self) -> str:
        """
        Gets the unique genetic code, generating if it is undefined.

        Returns:
            str: String name of genome.
        """
        if self.genetic_code is None:
            self.genetic_code = str(self.genome)
        return self.genetic_code

    def __str__(self) -> str:
        return '({0}:{1})'.format(self.name, self.fitness)


class Island:
    """
    A simple Island to perform a Genetic Algorithm. By default the selection, mutation, crossover, and probability functions
    default to the classic functions:
    - `natural_selection.ga.selection.selection_function_classic`
    - `natural_selection.ga.selection.selection_function_parents_classic`
    - `natural_selection.ga.crossover.crossover_function_classic`
    - `natural_selection.ga.mutation.mutation_function_classic`
    - `natural_selection.ga.prob_functions.crossover_prob_function_classic`
    - `natural_selection.ga.prob_functions.mutation_prob_function_classic`
    - `natural_selection.ga.helper_functions.clone_function_classic`

    Args:
        function_params (dict): The parameters for the fitness function.
        selection_function (Callable): Function for selecting individuals for crossover and mutation (default = None).
        parent_selection (Callable): Function for selecting parents for crossover (default = None).
        crossover_function (Callable): Function for crossover (default = None).
        mutate_function (Callable): Function for mutation (default = None).
        crossover_prob_function (Callable): Random probability function for crossover (default = None).
        mutation_prob_function (Callable): Random probability function for mutation (default = None).
        clone_function (Callable): Function for cloning (default = None).
        verbose (bool): Print all information (default = None).
        logging_function (Callable): Function for custom message logging, such as server logging (default = None).
        force_genetic_diversity (bool): Only add new offspring to the population if they have a unique genome (default = True).
    """

    def __init__(self, function_params : dict, selection_function : Callable = None, parent_selection : Callable = None,
                 crossover_function : Callable = None, mutate_function : Callable = None,
                 crossover_prob_function : Callable = None, mutation_prob_function : Callable = None,
                 clone_function : Callable = None, verbose : bool = True, logging_function : Callable = None,
                 force_genetic_diversity : bool = True):
        self.function_params = function_params
        self.unique_genome = list()
        self.generation_info = list()
        self.population = list()

        if selection_function:
            self.selection = selection_function
        else:
            self.selection = selection_function_classic

        if parent_selection:
            self.parent_selection = parent_selection
        else:
            self.parent_selection = selection_function_parents_classic

        if crossover_function:
            self.crossover = crossover_function
        else:
            self.crossover = crossover_function_classic

        if mutate_function:
            self.mutate = mutate_function
        else:
            self.mutate = mutation_function_classic

        if crossover_prob_function:
            self.crossover_prob = crossover_prob_function
        else:
            self.crossover_prob = crossover_prob_function_classic

        if mutation_prob_function:
            self.mutation_prob = mutation_prob_function
        else:
            self.mutation_prob = mutation_prob_function_classic

        if clone_function:
            self.clone = clone_function
        else:
            self.clone = clone_function_classic

        self.logging_function = logging_function
        self.force_genetic_diversity = force_genetic_diversity

        self.verbose = verbose

        self.elites = list()
        self.mutants = list()
        self.children = list()
        self.species_type = "DEFAULT_SPECIES"

    def create(self, adam : Individual, random_seed : int = 42, population_size : int = 8):
        """
        Starts the population by taking an initial individual as template and creating new ones from it.

        Args:
            adam (Individual): Individual to clone from.
            random_seed (int): Random seed (default = 42).
            population_size (int): Size of population.
        """
        random.seed(random_seed)

        self.species_type = adam.species_type

        for i in range(population_size-1):
            eve = Individual(adam.fitness_function, genome=Genome([x.randomize_new() for x in adam.genome]))
            self.population.append(eve)

        self.population.append(adam)

        for popitem in self.population:
            popitem.evaluate(self.function_params)
            self.unique_genome.append(popitem.unique_genetic_code())

    def import_migrants(self, migrants : list, reset_fitness : bool = False, species_check : bool = True,
                        force_genetic_diversity : bool = True):
        """
        Imports a list of individuals, with option to re-evaluate them.
        Skips the individual if the genetic code is already in the population.

        Args:
            migrants (list): List of Individuals.
            reset_fitness (bool): Reset the fitness of new members and evaluate them (default = False).
            species_check (bool): Safely check that imported members are compatible with population  (default = True).
            force_genetic_diversity (bool): Only add migrants to the population if they have a unique genome (default = True).
        """
        for i in migrants:
            if species_check:
                if i.species_type != self.species_type:
                    continue
            if force_genetic_diversity:
                if not i.unique_genetic_code() in self.unique_genome:
                    if i.fitness is None or reset_fitness:
                        i.evaluate(self.function_params)
                    self.population.append(i)
                    self.unique_genome.append(i.unique_genetic_code())
            else:
                if i.fitness is None or reset_fitness:
                    i.evaluate(self.function_params)
                self.population.append(i)
                self.unique_genome.append(i.unique_genetic_code())


    def evolve_generational(self, starting_generation : int = 0, n_generations : int = 5, crossover_probability : float = 0.5,
               mutation_probability : float = 0.5, crossover_params : dict = None, mutation_params : dict = None,
               selection_params : dict = None) -> Individual:
        """
        Starts the evolutionary run.

        Args:
            starting_generation (int): Starting generation.
            n_generations (int): Number of generations to run.
            crossover_probability (float): Initial crossover probability.
            mutation_probability (float): Initial mutation probability.
            crossover_params (dict): Dict of params for custom crossover function (default = None).
            mutation_params (dict): Dict of params for custom mutation function (default = None).
            selection_params (dict): Dict of params for custom selection function (default = None).

        Returns:
            Individual: The fittest Individual found.
        """

        if crossover_params:
            _crossover_params = crossover_params
        else:
            _crossover_params = {'prob':0.5}

        if mutation_params:
            _mutation_params = mutation_params
        else:
            _mutation_params = {'prob':0.2}

        if selection_params:
            _selection_params = selection_params
        else:
            _selection_params = {'n':5, 'desc':True}

        for g in range(starting_generation, starting_generation + n_generations):
            self._verbose_logging('Generation {} started'.format(g))

            self.__evolutionary_engine(g=g, selection_params=_selection_params,
                                       crossover_probability=crossover_probability,
                                       mutation_probability=mutation_probability, crossover_params=_crossover_params,
                                       mutation_params=_mutation_params)

        best_ind = selection_function_classic(island=self, population=self.population, n=1)[0]

        self._verbose_logging("Best individual is {0}, {1}".format(best_ind.name, best_ind.fitness))

        return best_ind

    def evolve_criterion(self, criterion_function : Callable, criterion_params : dict, crossover_probability : float = 0.5,
               mutation_probability : float = 0.5, crossover_params : dict = None, mutation_params : dict = None,
               selection_params : dict = None) -> Individual:
        """
        Starts the evolutionary run and evaluates until the criterion_func returns true.

        Args:
            criterion_function (Callable): A function to evaluate if the desired criterion has been met.
            criterion_params (dict): Function parameters for criterion.
            crossover_probability (float): Initial crossover probability.
            mutation_probability (float): Initial mutation probability.
            crossover_params (dict): Dict of params for custom crossover function (default = None).
            mutation_params (dict): Dict of params for custom mutation function (default = None).
            selection_params (dict): Dict of params for custom selection function (default = None).

        Returns:
            Individual: The fittest Individual found.
        """

        if crossover_params:
            _crossover_params = crossover_params
        else:
            _crossover_params = {'prob':0.5}

        if mutation_params:
            _mutation_params = mutation_params
        else:
            _mutation_params = {'prob':0.2}

        if selection_params:
            _selection_params = selection_params
        else:
            _selection_params = {'n':5, 'desc':True}

        g = 0
        while not criterion_function(island=self, **criterion_params):
            g += 1
            self._verbose_logging('Generation {} started'.format(g))

            self.__evolutionary_engine(g=g, selection_params=_selection_params, crossover_probability=crossover_probability,
                                        mutation_probability=mutation_probability, crossover_params=_crossover_params,
                                        mutation_params=_mutation_params)

        best_ind = selection_function_classic(island=self, population=self.population, n=1)[0]

        self._verbose_logging("Best individual is {0}, {1}".format(best_ind.name, best_ind.fitness))

        return best_ind

    def __evolutionary_engine(self, g, selection_params, crossover_probability, mutation_probability, crossover_params, mutation_params):

        elites = self.selection(island=self,
                                population=self.clone(population=self.population, island=self),
                                **selection_params)

        self.elites.append({'generation' : g, 'elites' : elites})

        # Children are strictly copies or new objects seeing as the have a lineage and parents
        generation_children = list()
        for parents in self.parent_selection(population=elites, island=self):
            if self.crossover_prob(crossover_probability=crossover_probability, island=self):

                self._verbose_logging('Crossover: {0}'.format([str(p) for p in parents]))

                children = self.crossover(island=self,
                                          individuals=self.clone(population=parents, island=self),
                                          **crossover_params)

                for child in children:
                    child.reset_name()
                    child.register_parent_names(parents)
                    child.age = 0
                    child.reset_fitness()

                generation_children.extend(children)

        self.children.append({'generation' : g, 'children' : generation_children})

        # Mutants are not strictly copied but rather only modified seeing as the are part of the children list
        generation_mutants = list()
        for mutant in generation_children:
            if self.mutation_prob(mutation_probability=mutation_probability, island=self):
                mutated = self.mutate(island=self, individual=mutant, **mutation_params)

                self._verbose_logging('Mutating: {}'.format(mutant.name))
                mutated.reset_fitness()
                generation_mutants.append(mutated)

        self.mutants.append({'generation': g, 'mutants': generation_mutants})

        untested_individuals = [ind for ind in generation_children if ind.fitness is None]

        for new_untested_individual in untested_individuals:
            new_untested_individual.evaluate(island=self, params=self.function_params)
            if self.force_genetic_diversity:
                # If we want a diverse gene pool, this must be true
                if not new_untested_individual.unique_genetic_code() in self.unique_genome:
                    self.population.append(new_untested_individual)
                    self.unique_genome.append(new_untested_individual.unique_genetic_code())
            else:
                # Else, add it effectively allowing "twins" to exist
                self.population.append(new_untested_individual)
                self.unique_genome.append(new_untested_individual.unique_genetic_code())


        population_fitnesses = [ind.fitness for ind in self.population]
        elite_fitnesses = [ind.fitness for ind in elites]

        population_length = len(self.population)
        population_mean = sum(population_fitnesses) / population_length
        population_sum2 = sum(x * x for x in population_fitnesses)
        population_std = abs(population_sum2 / population_length - population_mean ** 2) ** 0.5

        self.generation_info.append(
            {
                'stat': 'population',
                'generation': g,
                'pop_len': population_length,
                'fitness_mean': population_mean,
                'fitness_std': population_std,
                'fitness_min': min(population_fitnesses),
                'fitness_max': max(population_fitnesses),
            }
        )

        self._verbose_logging(self.generation_info[-1])

        elite_length = len(elites)
        elite_mean = sum(elite_fitnesses) / elite_length
        elite_sum2 = sum(x * x for x in elite_fitnesses)
        elite_std = abs(elite_sum2 / elite_length - elite_mean ** 2) ** 0.5

        self.generation_info.append(
            {
                'stat': 'elites',
                'generation': g,
                'pop_len': elite_length,
                'fitness_mean': elite_mean,
                'fitness_std': elite_std,
                'fitness_min': min(elite_fitnesses),
                'fitness_max': max(elite_fitnesses),
            }
        )

        self._verbose_logging(self.generation_info[-1])

        for i in self.population:
            i.birthday()

        self.generation_count = g

    def _verbose_logging(self, message):
        if self.verbose:
            print(message)
        if self.logging_function:
            self.logging_function(message)
