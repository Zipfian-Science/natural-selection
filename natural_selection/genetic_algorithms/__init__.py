# -*- coding: utf-8 -*-
"""Basic classes for running Genetic Algorithms.
"""
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2021, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"

import random
import uuid
from typing import Callable, Any, Iterable


from natural_selection.genetic_algorithms.operators.initialisation import initialise_population_random
from natural_selection.genetic_algorithms.operators.selection import selection_elites_top_n, selection_parents_two
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_uniform
from natural_selection.genetic_algorithms.operators.mutation import mutation_randomize
from natural_selection.genetic_algorithms.utils.probability_functions import crossover_prob_function_classic, mutation_prob_function_classic
from natural_selection.genetic_algorithms.utils import clone_classic, GeneticAlgorithmError

class Gene:
    """
    A simple class to encapsulate a simple gene.

    Args:
        name (str): Gene name. The gene name also acts as a compatibility reference.
        value (Any): The value, could be any type.
        randomise_function (Callable): A function to randomise the gene, taking the gene (`self`) as input with signature ``func(self)``.
        gene_max (Any, numeric type): Max value for random number generator (default = None).
        gene_min (Any, numeric type): Min value for random number generator (default = None).
        mu (Any, numeric type): Mean value of distribution to sample from (default = 0).
        sig (Any, numeric type): Std. Dev. value of distribution to sample from (default = 1).
        step_lower_bound (Any, numeric type): For uniform stepping functions, defines lower bound of range (default = -1.0).
        step_upper_bound (Any, numeric type): For uniform stepping functions, defines upper bound of range (default = 1.0).
        choices (Iterable): List of choices, categorical or not, to randomly choose from (default = None).
        gene_properties (dict): For custom random functions, extra params may be given (default = None).

    """

    def __init__(self,
                 name : str,
                 value : Any,
                 randomise_function: Callable,
                 gene_max : Any = None,
                 gene_min : Any = None,
                 mu : Any = 0,
                 sig: Any = 1,
                 step_lower_bound : Any = -1.0,
                 step_upper_bound: Any = 1.0,
                 choices : Iterable  = None,
                 gene_properties : dict = None):
        self.name = name
        self.value = value
        self.gene_max = gene_max
        self.gene_min = gene_min
        self.mu = mu
        self.sig = sig
        self.step_lower_bound = step_lower_bound
        self.step_upper_bound = step_upper_bound
        self.choices = choices
        self.randomise_function = randomise_function
        self.__gene_properties = gene_properties
        if gene_properties:
            for k, v in gene_properties.items():
                self.__dict__.update({k:v})

    def randomise(self):
        """
        Sets a random value gene with randomised value.
        """
        self.value = self.randomise_function(gene=self)

    def randomise_new(self):
        """
        Return a new gene with randomised value.

        Returns:
            Gene: Newly created gene.
        """
        return Gene(
            name=self.name,
            value=self.randomise_function(gene=self),
            randomise_function=self.randomise_function,
            gene_max=self.gene_max,
            gene_min=self.gene_min,
            mu=self.mu,
            sig=self.sig,
            choices=self.choices,
            gene_properties=self.__gene_properties
        )

    def add_new_property(self, key : str, value : Any):
        if self.__gene_properties:
            self.__gene_properties.update({key: value})
        else:
            self.__gene_properties = {key: value}
        self.__dict__.update({key: value})

    def __str__(self) -> str:
        return f'Gene({self.name}:{self.value})'

    def __repr__(self):
        start_str = f'Gene({self.name}:{self.value}:{self.gene_max}:{self.gene_min}:{self.mu}:{self.sig}:{self.step_lower_bound}:{self.step_upper_bound}'
        if self.choices:
            for c in self.choices:
                start_str = f'{start_str}:{str(c)}'
        start_str = f'{start_str}:{str(self.randomise_function)}'
        start_str = f'{start_str}:{str(self.__gene_properties)})'
        return  start_str


class Chromosome:
    """
    A class that encapsulates an ordered sequence of Gene objects.

    Note:
        gene_verify_func should take the gene, index of gene, and chromosome (`self`) as parameters.

    Args:
        genes (list): list of initialised Gene objects.
        gene_verify_func (Callable): A function to verify gene compatibility `func(gene,loc,chromosome)` (default = None).
    """

    def __default_gene_verify_func(self, gene, loc, chromosome):
        """
        Needed to make objects pickle-able.
        Args:
            gene (Gene): Gene being inserted.
            loc (int): Index of gene.
            chromosome (Chromosome): The given "self".

        Returns:
            bool: Whether gene insertion is allowed or not.
        """
        return True

    def __init__(self, genes: list = None, gene_verify_func : Callable = None):
        if genes:
            self.genes = genes
        else:
            self.genes = list()

        if gene_verify_func:
            self.gene_verify_func = gene_verify_func
        else:
            self.gene_verify_func = self.__default_gene_verify_func

    def append(self, gene: Gene):
        """
        Simple appending of Gene type objects.

        Args:
            gene (Gene): Gene
        """
        assert isinstance(gene, Gene), 'Must be Gene type!'
        if not self.gene_verify_func(gene=gene,loc=-1,chromosome=self):
            raise GeneticAlgorithmError(message="Added gene did not pass compatibility tests!")
        self.genes.append(gene)

    def __setitem__(self, index, gene):
        if isinstance(index, slice):
            assert index.start < len(self.genes), 'Index Out of bounds!'
        else:
            assert isinstance(gene, Gene), 'Must be Gene type!'
            assert index < len(self.genes), 'Index Out of bounds!'

        if not self.gene_verify_func(gene=gene, loc=index, chromosome=self):
            raise GeneticAlgorithmError("Index set gene did not pass compatibility tests!")
        self.genes[index] = gene

    def __getitem__(self, index) -> Gene:
        if isinstance(index, slice):
            assert index.start < len(self.genes), 'Index Out of bounds!'
        else:
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
        start_str = '-'.join([str(t) for t in self.genes])
        return f'Chromosome({start_str})'

    def __repr__(self):
        start_str = '-'.join([repr(t) for t in self.genes])
        start_str = f'Chromosome({start_str})'
        return start_str


class Individual:
    """
    A class that encapsulates a single individual, with genetic code and a fitness evaluation function.

    Args:
        fitness_function (Callable): Function with ``func(Chromosome, island, **params)`` signature.
        name (str): Name for keeping track of lineage (default = None).
        chromosome (Chromosome): A Chromosome object, initialised (default = None).
        species_type (str) : A unique string to identify the species type, for preventing cross polluting (default = None).

    Attributes:
        fitness (Numeric): The fitness score after evaluation.
        age (int): How many generations was the individual alive.
        genetic_code (str): String representation of Chromosome.
        history (list): List of dicts of every evaluation.
        parents (list): List of strings of parent names.
    """

    def __init__(self, fitness_function : Callable, name : str = None, chromosome: Chromosome = None, species_type : str = None):
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name
        if chromosome is None:
            self.chromosome = Chromosome()
        else:
            self.chromosome = chromosome
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
        Reset (or set) the fitness of the individual.

        Args:
            fitness (Any): New fitness value (default = None).
            reset_genetic_code (bool): Whether to reset the genetic code. (default = True)
        """
        self.fitness = fitness
        if reset_genetic_code:
            self.genetic_code = None

    def add_gene(self, gene : Gene):
        """
        Appends a gene to the chromosome.

        Args:
            gene (Gene): Gene to add.
        """
        self.chromosome.append(gene)

    def evaluate(self, params : dict, island=None) -> Any:
        """
        Run the fitness function with the given params.

        Args:
            params (dict): Named dict of eval params.
            island (Island): Pass the Island for advanced fitness functions based on Island properties and populations.

        Returns:
            numeric: Fitness value.
        """
        self.fitness = self.fitness_function(chromosome=self.chromosome, island=island, **params)

        stamp = { "name": self.name,
                  "age": self.age,
                  "fitness": self.fitness,
                  "chromosome": str(self.chromosome),
                  "parents": self.parents,
         }

        if island:
            stamp["island_generation" ] = island.generation_count

        self.history.append(stamp)
        return self.fitness

    def unique_genetic_code(self) -> str:
        """
        Gets the unique genetic code, generating if it is undefined.

        Returns:
            str: String name of Chromosome.
        """
        if self.genetic_code is None:
            self.genetic_code = repr(self.chromosome)
        return self.genetic_code

    def __str__(self) -> str:
        return f'Individual({self.name}:{self.fitness})'

    def __repr__(self) -> str:
        genetic_code = self.unique_genetic_code()
        return f'Individual({self.name}:{self.fitness}:{self.age}:{self.species_type}:{genetic_code})'

    def __eq__(self, other):
        if isinstance(other, Individual):
            return self.unique_genetic_code() == other.unique_genetic_code()
        else:
            raise GeneticAlgorithmError(message=f'Can not compare {type(other)}')

    def __ne__(self, other):
        if isinstance(other, Individual):
            return self.unique_genetic_code() != other.unique_genetic_code()
        else:
            raise GeneticAlgorithmError(message=f'Can not compare {type(other)}')

    def __lt__(self, other):
        if isinstance(other, Individual):
            return self.fitness < other.fitness
        elif isinstance(other, int):
            return self.fitness < other
        elif isinstance(other, float):
            return self.fitness < other
        else:
            raise GeneticAlgorithmError(message=f'Can not compare {type(other)}')

    def __le__(self, other):
        if isinstance(other, Individual):
            return self.fitness <= other.fitness
        elif isinstance(other, int):
            return self.fitness <= other
        elif isinstance(other, float):
            return self.fitness <= other
        else:
            raise GeneticAlgorithmError(message=f'Can not compare {type(other)}')

    def __gt__(self, other):
        if isinstance(other, Individual):
            return self.fitness > other.fitness
        elif isinstance(other, int):
            return self.fitness > other
        elif isinstance(other, float):
            return self.fitness > other
        else:
            raise GeneticAlgorithmError(message=f'Can not compare {type(other)}')

    def __ge__(self, other):
        if isinstance(other, Individual):
            return self.fitness >= other.fitness
        elif isinstance(other, int):
            return self.fitness >= other
        elif isinstance(other, float):
            return self.fitness >= other
        else:
            raise GeneticAlgorithmError(message=f'Can not compare {type(other)}')

class Island:
    """
    A simple Island to perform a Genetic Algorithm. By default the selection, mutation, crossover, and probability functions
    default to the classic functions.

    Args:
        function_params (dict): The parameters for the fitness function.
        elite_selection (Callable): Function for selecting individuals for crossover and mutation (default = None).
        parent_selection (Callable): Function for selecting parents for crossover (default = None).
        crossover_function (Callable): Function for crossover (default = None).
        mutation_function (Callable): Function for mutation (default = None).
        crossover_prob_function (Callable): Random probability function for crossover (default = None).
        mutation_prob_function (Callable): Random probability function for mutation (default = None).
        clone_function (Callable): Function for cloning (default = None).
        random_seed (int): Random seed for random and Numpy generators, set to None for no seed (default = 42).
        verbose (bool): Print all information (default = None).
        logging_function (Callable): Function for custom message logging, such as server logging (default = None).
        force_genetic_diversity (bool): Only add new offspring to the population if they have a unique chromosome (default = True).

    Attributes:
        unique_genome (list): List of unique chromosomes.
        generation_info (list): List of dicts detailing info for every generation.
        population (list): The full population of members.
        elites (list): All elites selected during the run.
        mutants (list): All mutants created during the run.
        children (list): All children created during the run.
        generation_count (int): The current generation number.
    """

    def __init__(self, function_params : dict,
                 elite_selection : Callable = selection_elites_top_n,
                 parent_selection : Callable = selection_parents_two,
                 crossover_function : Callable = crossover_two_uniform,
                 mutation_function : Callable = mutation_randomize,
                 crossover_prob_function : Callable = crossover_prob_function_classic,
                 mutation_prob_function : Callable = mutation_prob_function_classic,
                 clone_function : Callable = clone_classic,
                 random_seed: int = 42,
                 verbose : bool = True,
                 logging_function : Callable = None,
                 force_genetic_diversity : bool = True):
        self.function_params = function_params
        self.unique_genome = list()
        self.generation_info = list()
        self.population = list()


        self.elite_selection = elite_selection
        self.parent_selection = parent_selection
        self.crossover = crossover_function
        self.mutation = mutation_function
        self.crossover_prob = crossover_prob_function
        self.mutation_prob = mutation_prob_function
        self.clone = clone_function

        self.logging_function = logging_function
        self.force_genetic_diversity = force_genetic_diversity

        self.verbose = verbose
        self.random_seed = random_seed
        self.elites = list()
        self.mutants = list()
        self.children = list()
        self.species_type = "DEFAULT_SPECIES"
        self.generation_count = 0

        # Set python random seed, as well as Numpy seed.
        if random_seed:
            random.seed(random_seed)
            from numpy.random import seed as np_seed
            np_seed(random_seed)

    def create_gene(self,
                 name : str,
                 value : Any,
                 randomise_function: Callable,
                 gene_max : Any = None,
                 gene_min : Any = None,
                 mu : Any = None,
                 sig: Any = None,
                 choices : Iterable  = None,
                 gene_properties : dict = None):
        """
        Wrapping function to create a new Gene. Useful when writing new initialisation functions. See Gene class.

        Args:
            name (str): Gene name. The gene name also acts as a compatibility reference.
            value (Any): The value, could be any type.
            randomise_function (Callable): A function to randomise the gene, taking the min and max as input with signature ``func(self)``.
            gene_max (Any, numeric type): Max value for random number generator (default = None).
            gene_min (Any, numeric type): Min value for random number generator (default = None).
            mu (Any, numeric type): Mean value of distribution to sample from (default = None).
            sig (Any, numeric type): Std. Dev. value of distribution to sample from (default = None).
            choices (Iterable): List of choices, categorical or not, to randomly choose from (default = None).
            gene_properties (dict): For custom random functions, extra params may be given (default = None).
        Returns:
            gene: A new Gene object.
        """
        return Gene(name, value, randomise_function, gene_max, gene_min, mu, sig, choices, gene_properties)

    def create_chromosome(self,
                          genes: list = None,
                          gene_verify_func : Callable = None):
        """
        Wrapping function to create a new Chromosome. Useful when writing new initialisation functions. See Chromosome class.

        Args:
            genes (list): list of initialised Gene objects.
            gene_verify_func (Callable): A function to verify gene compatibility `func(gene,loc,chromosome)` (default = None).

        Returns:
            chromosome: A new Chromosome.
        """
        return Chromosome(genes, gene_verify_func)

    def create_individual(self,
                          fitness_function : Callable,
                          name : str = None,
                          chromosome: Chromosome = None,
                          species_type : str = None,
                          add_to_population : bool = False):
        """
        Wrapping function to create a new Individual. Useful when writing new initialisation functions. See Individual class.

        Args:
            fitness_function (Callable): Function with ``func(Chromosome, island, **params)`` signature
            name (str): Name for keeping track of lineage (default = None).
            chromosome (Chromosome): A Chromosome object, initialised (default = None).
            species_type (str) : A unique string to identify the species type, for preventing cross polluting (default = None).
            add_to_population (bool): Add this new individual to the population (default = False).

        Returns:
            individual: A new Individual.
        """
        ind = Individual(fitness_function, name, chromosome, species_type)
        if add_to_population:
            self.population.append(ind)
        return ind

    def initialise(self, adam : Individual,
               population_size: int = 8,
               initialisation_function : Callable = initialise_population_random,
               initialisation_params : dict = {},
               evaluate_population : bool = True
               ):
        """
        Starts the population by taking an initial individual as template and creating new ones from it.

        Args:
            adam (Individual): Individual to clone from.
            population_size (int): Size of population.
            initialisation_function (Callable): A function for randomly creating new individuals from the given adam.
            initialisation_params (dict): Custom params for custom initialisation functions.
            evaluate_population (bool): Evaluate the newly created population (default = True).
        """
        self.species_type = adam.species_type

        self.initialise = initialisation_function

        self.population = self.initialise(adam=adam, n=population_size, island=self, **initialisation_params)

        if evaluate_population:
            for popitem in self.population:
                popitem.evaluate(self.function_params, island=self)
                self.unique_genome.append(popitem.unique_genetic_code())

    def import_migrants(self, migrants : list,
                        reset_fitness : bool = False,
                        species_check : bool = True,
                        force_genetic_diversity : bool = True):
        """
        Imports a list of individuals, with option to re-evaluate them.
        Skips the individual if the genetic code is already in the population.

        Args:
            migrants (list): List of Individuals.
            reset_fitness (bool): Reset the fitness of new members and evaluate them (default = False).
            species_check (bool): Safely check that imported members are compatible with population  (default = True).
            force_genetic_diversity (bool): Only add migrants to the population if they have a unique chromosome (default = True).
        """
        for i in migrants:
            if species_check:
                if i.species_type != self.species_type:
                    continue
            if force_genetic_diversity:
                if not i.unique_genetic_code() in self.unique_genome:
                    if i.fitness is None or reset_fitness:
                        i.evaluate(self.function_params, island=self)
                    self.population.append(i)
                    self.unique_genome.append(i.unique_genetic_code())
            else:
                if i.fitness is None or reset_fitness:
                    i.evaluate(self.function_params)
                self.population.append(i)
                self.unique_genome.append(i.unique_genetic_code())


    def evolve(self, starting_generation : int = 0,
                            n_generations : int = 5,
                            crossover_probability : float = 0.5,
                            mutation_probability : float = 0.5,
                            crossover_params : dict = None,
                            mutation_params : dict = None,
                            elite_selection_params : dict = None,
                            parent_selection_params: dict = None,
                            survivor_selection_params: dict = None,
                            criterion_function : Callable = None,
                            criterion_params : dict = None) -> Individual:
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
            criterion_function (Callable): A function to evaluate if the desired criterion has been met (default = None).
            criterion_params (dict): Function parameters for criterion (default = None).

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

        if elite_selection_params:
            _elite_selection_params = elite_selection_params
        else:
            _elite_selection_params = {'n':5, 'desc':True}

        if parent_selection_params:
            _parent_selection_params = parent_selection_params
        else:
            _parent_selection_params = {}

        if survivor_selection_params:
            _survivor_selection_params = survivor_selection_params
        else:
            _survivor_selection_params = {}

        def _default_criterion_function(island):
            return island.generation_count < starting_generation + n_generations - 1

        if criterion_function:
            g_func = criterion_function
        else:
            g_func = _default_criterion_function
            criterion_params = {}

        g = starting_generation
        while g_func(island=self, **criterion_params):
            self._verbose_logging('Generation {} started'.format(g))

            self.__evolutionary_engine(g=g,
                                       elite_selection_params=_elite_selection_params,
                                       parent_selection_params=_parent_selection_params,
                                       survivor_selection_params=_survivor_selection_params,
                                       crossover_probability=crossover_probability,
                                       mutation_probability=mutation_probability,
                                       crossover_params=_crossover_params,
                                       mutation_params=_mutation_params)
            g += 1

        best_ind = selection_elites_top_n(island=self, individuals=self.population, n=1)[0]

        self._verbose_logging("Best individual is {0}, {1}".format(best_ind.name, best_ind.fitness))

        return best_ind

    def __evolutionary_engine(self,
                              g,
                              elite_selection_params,
                              parent_selection_params,
                              survivor_selection_params,
                              crossover_probability,
                              mutation_probability,
                              crossover_params,
                              mutation_params):

        elites = self.elite_selection(island=self,
                                      individuals=self.clone(individuals=self.population, island=self),
                                      **elite_selection_params)

        self.elites.append({'generation' : g, 'elites' : elites})

        # Children are strictly copies or new objects seeing as the have a lineage and parents
        generation_children = list()
        for parents in self.parent_selection(individuals=elites, island=self, **parent_selection_params):
            if self.crossover_prob(crossover_probability=crossover_probability, island=self):

                self._verbose_logging('Crossover: {0}'.format([str(p) for p in parents]))

                children = self.crossover(island=self,
                                          individuals=self.clone(individuals=parents, island=self),
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
                mutated = self.mutation(island=self, individual=mutant, **mutation_params)

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
