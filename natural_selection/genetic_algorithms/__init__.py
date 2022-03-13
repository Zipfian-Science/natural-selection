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
import warnings as w
import pickle
import logging
from datetime import datetime
from collections import OrderedDict
import copy
from time import gmtime

import numpy as np

from natural_selection import get_random_string
from natural_selection.genetic_algorithms.operators.initialisation import initialise_population_random
from natural_selection.genetic_algorithms.operators.selection import selection_elites_top_n, selection_parents_two, selection_survivors_all
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_uniform
from natural_selection.genetic_algorithms.operators.mutation import mutation_randomize
from natural_selection.genetic_algorithms.utils.probability_functions import crossover_prob_function_classic, mutation_prob_function_classic
from natural_selection.genetic_algorithms.utils import clone_classic, default_save_checkpoint_function, GeneticAlgorithmError

from natural_selection import  __version__ as package_version

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
        if '<lambda>' in repr(randomise_function):
            w.warn("WARNING: 'randomise_function' lambda can not be pickled using standard libraries.")
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
            name=copy.copy(self.name),
            value=self.randomise_function(gene=self),
            randomise_function=self.randomise_function,
            gene_max=copy.copy(self.gene_max),
            gene_min=copy.copy(self.gene_min),
            mu=copy.copy(self.mu),
            sig=copy.copy(self.sig),
            choices=copy.deepcopy(self.choices),
            gene_properties=copy.deepcopy(self.__gene_properties)
        )

    def add_new_property(self, key : str, value : Any):
        """
        Method to add new properties (attributes).

        Args:
            key (str): Name of property.
            value (Any): Anything.
        """
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
            start_str = f"{start_str}:[{repr(self.choices)}]"
        start_str = f'{start_str}:{self.randomise_function.__name__}'
        if self.__gene_properties:
            start_str = f'{start_str}:{str(self.__gene_properties)}'
        return  f'{start_str})'


class Chromosome:
    """
    A class that encapsulates an ordered sequence of Gene objects.

    Note:
        gene_verify_func should take the gene, index of gene, and chromosome (`self`) as parameters.

    Args:
        genes (list): list of initialised Gene objects.
        gene_verify_func (Callable): A function to verify gene compatibility `func(gene,loc,chromosome)` (default = None).
        chromosome_properties (dict): For custom functions, extra params may be given (default = None).
    """

    def __init__(self, genes: list = None,
                 gene_verify_func : Callable = None,
                 chromosome_properties : dict = None):
        if genes:
            self.genes = genes
        else:
            self.genes = list()

        if gene_verify_func and '<lambda>' in repr(gene_verify_func):
            w.warn("WARNING: 'gene_verify_func' lambda can not be pickled using standard libraries.")
        self.gene_verify_func = gene_verify_func
        self.__chromosome_properties = chromosome_properties
        if chromosome_properties:
            for k, v in chromosome_properties.items():
                self.__dict__.update({k: v})

    def append(self, gene: Gene):
        """
        Simple appending of Gene type objects.

        Args:
            gene (Gene): Gene
        """
        assert isinstance(gene, Gene), 'Must be Gene type!'
        if self.gene_verify_func and not self.gene_verify_func(gene=gene,loc=-1,chromosome=self):
            raise GeneticAlgorithmError(message="Added gene did not pass compatibility tests!")
        self.genes.append(gene)

    def add_new_property(self, key : str, value : Any):
        """
        Method to add new properties (attributes).

        Args:
            key (str): Name of property.
            value (Any): Anything.
        """
        if self.__chromosome_properties:
            self.__chromosome_properties.update({key: value})
        else:
            self.__chromosome_properties = {key: value}
        self.__dict__.update({key: value})

    def get_properties(self):
        """
        Gets a dict of the custom properties that were added at initialisation or the `add_new_property` method.

        Returns:
            dict: All custom properties.
        """
        return self.__chromosome_properties

    def randomise_gene(self, index : int):
        """
        Randomise a gene at index.

        Args:
            index (int): Index of gene.
        """
        assert index < len(self.genes), 'Index Out of bounds!'
        self.genes[index].randomise()

    def randomise_all_genes(self):
        """
        Randomises all genes in chromosome.
        """
        for gene in self.genes:
            gene.randomise()

    def __setitem__(self, index, gene):
        if isinstance(index, slice):
            assert index.start < len(self.genes), 'Index Out of bounds!'
        else:
            assert isinstance(gene, Gene), 'Must be Gene type!'
            assert index < len(self.genes), 'Index Out of bounds!'

        if self.gene_verify_func and not self.gene_verify_func(gene=gene, loc=index, chromosome=self):
            raise GeneticAlgorithmError("Index set gene did not pass compatibility tests!")
        self.genes[index] = gene

    def to_dict(self) -> OrderedDict:
        """
        Helper function to convert chromosome into a key-value Python dictionary, assuming genes have unique names!

        Returns:
            OrderedDict: Ordered dictionary of genes.
        """
        return OrderedDict({gene.name : gene.value for gene in self.genes})


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
        fitness_function (Callable): Function with ``func(Chromosome, island, **params)`` signature (default = None).
        name (str): Name for keeping track of lineage (default = None).
        chromosome (Chromosome): A Chromosome object, initialised (default = None).
        species_type (str) : A unique string to identify the species type, for preventing cross polluting (default = None).
        filepath (str): Skip init and load from a pickled file.
        individual_properties (dict): For fitness functions, extra params may be given (default = None).

    Attributes:
        fitness (Numeric): The fitness score after evaluation.
        age (int): How many generations was the individual alive.
        genetic_code (str): String representation of Chromosome.
        history (list): List of dicts of every evaluation.
        parents (list): List of strings of parent names.
    """

    def __init__(self, fitness_function : Callable = None,
                 name : str = None,
                 chromosome: Chromosome = None,
                 species_type : str = None,
                 filepath : str = None,
                 individual_properties : dict = None):
        if not filepath is None:
            self.load_individual(filepath=filepath)
            return
        if fitness_function and '<lambda>' in repr(fitness_function):
            w.warn("WARNING: 'fitness_function' lambda can not be pickled using standard libraries.")
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
            self.species_type = "def"

        self.__individual_properties = individual_properties
        if individual_properties:
            for k, v in individual_properties.items():
                self.__dict__.update({k: v})

    def register_parent_names(self, parents : list, reset_parent_name_list : bool = True):
        """
        In keeping lineage of family lines, the names of parents are kept track of.

        Args:
            parents (list): A list of Individuals of the parents.
        """
        if reset_parent_name_list:
            self.parents = list()
        for parent in parents:
            self.parents.append(parent)

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

    def evaluate(self, params : dict = None, island=None) -> Any:
        """
        Run the fitness function with the given params.

        Args:
            params (dict): Named dict of eval params (default = None).
            island (Island): Pass the Island for advanced fitness functions based on Island properties and populations (default = None).

        Returns:
            numeric: Fitness value.
        """
        if not params is None:
            _params = params
        else:
            _params = {}
        try:
            self.fitness = self.fitness_function(individual=self, island=island, **_params)
        except Exception as exc:
            if island:
                island.verbose_logging(f"ERROR: {self.name} - {repr(self.chromosome)} - {repr(exc)}")
            raise GeneticAlgorithmError(message=f'Could not evaluate individual "{self.name}" due to {repr(exc)}')

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

    def save_individual(self, filepath : str):
        """
        Save an individual to a pickle file.

        Args:
            filepath (str): File path to write to.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_individual(self, filepath : str):
        """
        Load an individual from a pickle file.

        Args:
            filepath (str): File path to load from.
        """
        with open(filepath, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def add_new_property(self, key : str, value : Any):
        """
        Method to add new properties (attributes).

        Args:
            key (str): Name of property.
            value (Any): Anything.
        """
        if self.__individual_properties:
            self.__individual_properties.update({key: value})
        else:
            self.__individual_properties = {key: value}
        self.__dict__.update({key: value})

    def get_properties(self) -> dict:
        """
        Gets a dict of the custom properties that were added at initialisation or the `add_new_property` method.

        Returns:
            dict: All custom properties.
        """
        return self.__individual_properties

    def get(self, key : str, default=None):
        """
        Gets the value of a property or returns default if it doesn't exist.

        Args:
            key (str): Property name.
            default: Value to return if the property is not found (default = None).

        Returns:
            any: The property of the individual.
        """
        if key in self.__dict__.keys():
            return self.__dict__[key]
        return default

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
        function_params (dict): The parameters for the fitness function (default = None).
        maximise_function (bool): If True, the fitness value is maximised, False will minimise the function fitness (default = True).
        elite_selection (Callable): Function for selecting individuals for crossover and mutation (default = None).
        initialisation_function (Callable): A function for randomly creating new individuals from the given adam.
        parent_selection (Callable): Function for selecting parents for crossover (default = None).
        crossover_function (Callable): Function for crossover (default = None).
        mutation_function (Callable): Function for mutation (default = None).
        crossover_prob_function (Callable): Random probability function for crossover (default = None).
        mutation_prob_function (Callable): Random probability function for mutation (default = None).
        clone_function (Callable): Function for cloning (default = None).
        survivor_selection_function (Callable): Function for selecting survivors (default = None).
        random_seed (int): Random seed for random and Numpy generators, set to None for no seed (default = None).
        name (str): General name for island, useful when working with multiple islands (default = None).
        verbose (bool): Print all information (default = None).
        logging_function (Callable): Function for custom message logging, such as server logging (default = None).
        save_checkpoint_function (Callable): Function for custom checkpoint saving (default = None).
        filepath (str): If a filepath is specified, the pickled island is loaded from it, skipping the rest of initialisation (default = None).
        save_checkpoint_level (int): Level of checkpoint saving 0 = none, 1 = per generation, 2 = per evaluation (default = 0).
        allow_twins (bool): Only add new offspring to the population if they have a unique chromosome (default = False).

    Attributes:
        unique_genome (list): List of unique chromosomes.
        generation_info (list): List of dicts detailing info for every generation.
        population (list): The full population of members.
        elites (list): All elites selected during the run.
        mutants (list): All mutants created during the run.
        children (list): All children created during the run.
        generation_count (int): The current generation number.
        checkpoints_dir (str): Directory name of where all checkpoints are saved.
        lineage (dict): A graph of lineage for the whole population.
    """

    __stat_key = 'stat'
    __generation_key = 'generation'
    __pop_len_key = 'pop_len'
    __fitness_mean_key = 'fitness_mean'
    __fitness_std_key = 'fitness_std'
    __fitness_min_key = 'fitness_min'
    __fitness_max_key = 'fitness_max'
    __most_fit_key = 'most_fit'
    __least_fit_key = 'least_fit'


    def __init__(self, function_params : dict = None,
                 maximise_function : bool = True,
                 initialisation_function: Callable = initialise_population_random,
                 elite_selection : Callable = selection_elites_top_n,
                 parent_selection : Callable = selection_parents_two,
                 crossover_function : Callable = crossover_two_uniform,
                 mutation_function : Callable = mutation_randomize,
                 crossover_prob_function : Callable = crossover_prob_function_classic,
                 mutation_prob_function : Callable = mutation_prob_function_classic,
                 survivor_selection_function: Callable = selection_survivors_all,
                 clone_function : Callable = clone_classic,
                 random_seed: int = None,
                 name : str = None,
                 verbose : bool = True,
                 logging_function : Callable = None,
                 save_checkpoint_function: Callable = default_save_checkpoint_function,
                 filepath : str = None,
                 save_checkpoint_level : int = 0,
                 allow_twins : bool = False):

        if name is None:
            self.name = get_random_string(include_numeric=True)
        else:
            self.name = name

        self.verbose = verbose
        self.logging_function = logging_function

        if verbose:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(island)-8s %(message)s',
                                filename=datetime.utcnow().strftime('%Y-%m-%d-ga-output.log'),
                                datefmt='%H:%M:%S')

            logging.Formatter.converter = gmtime

        if filepath:
            self.load_island(filepath)
            return

        self.verbose_logging(f"island: create v{package_version}")
        if function_params:
            self.verbose_logging(f"island: param {function_params}")
            self.function_params = function_params
        else:
            self.function_params = dict()
        self.unique_genome = list()
        self.generation_info = list()
        self.lineage = {
            'nodes' : [],
            'edges' : []
        }
        self.population = list()

        self.maximise_function = maximise_function
        self.verbose_logging(f"island: maximise_function {maximise_function}")
        self._initialise = initialisation_function
        self.verbose_logging(f"island: initialisation_function {initialisation_function.__name__}")
        self.elite_selection = elite_selection
        self.verbose_logging(f"island: elite_selection {elite_selection.__name__}")
        self.parent_selection = parent_selection
        self.verbose_logging(f"island: parent_selection {parent_selection.__name__}")
        self.crossover = crossover_function
        self.verbose_logging(f"island: crossover_function {crossover_function.__name__}")
        self.mutation = mutation_function
        self.verbose_logging(f"island: mutation_function {mutation_function.__name__}")
        self.crossover_prob = crossover_prob_function
        self.verbose_logging(f"island: crossover_prob_function {crossover_prob_function.__name__}")
        self.mutation_prob = mutation_prob_function
        self.verbose_logging(f"island: mutation_prob_function {mutation_prob_function.__name__}")
        self.clone = clone_function
        self.verbose_logging(f"island: clone_function {clone_function.__name__}")
        self.survivor_selection = survivor_selection_function
        self.verbose_logging(f"island: survivor_selection_function {survivor_selection_function.__name__}")

        self.save_checkpoint = save_checkpoint_function
        self.verbose_logging(f"island: save_checkpoint_function {save_checkpoint_function.__name__}")

        self.allow_twins = allow_twins
        self.verbose_logging(f"island: allow_twins {allow_twins}")

        self.random_seed = random_seed
        self.verbose_logging(f"island: random_seed {random_seed}")
        self.elites = list()
        self.mutants = list()
        self.children = list()
        self.species_type = "def"
        self.generation_count = 0
        self.save_checkpoint_level = save_checkpoint_level
        self.checkpoints_dir = f'_ga_checkpoints'
        self.verbose_logging(f"island: save_checkpoint_level {save_checkpoint_level}")

        # Set python random seed, as well as Numpy seed.
        if random_seed:
            random.seed(random_seed)
            from numpy.random import seed as np_seed
            np_seed(random_seed)


        if '<lambda>' in repr(initialisation_function):
            w.warn("WARNING: 'initialisation_function' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(elite_selection):
            w.warn("WARNING: 'elite_selection' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(parent_selection):
            w.warn("WARNING: 'parent_selection' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(crossover_function):
            w.warn("WARNING: 'crossover_function' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(mutation_function):
            w.warn("WARNING: 'mutation_function' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(crossover_prob_function):
            w.warn("WARNING: 'crossover_prob_function' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(mutation_prob_function):
            w.warn("WARNING: 'mutation_prob_function' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(clone_function):
            w.warn("WARNING: 'clone_function' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(survivor_selection_function):
            w.warn("WARNING: 'survivor_selection_function' lambda can not be pickled using standard libraries.")

        if logging_function and '<lambda>' in repr(logging_function):
            w.warn("WARNING: 'logging_function' lambda can not be pickled using standard libraries.")

        if save_checkpoint_function and '<lambda>' in repr(save_checkpoint_function):
            w.warn("WARNING: 'save_checkpoint_function' lambda can not be pickled using standard libraries.")

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
        return Gene(name=name, value=value, randomise_function=randomise_function, gene_max=gene_max, gene_min=gene_min,
                    mu=mu, sig=sig, choices=choices, gene_properties=gene_properties)

    def create_chromosome(self,
                          genes: list = None,
                          gene_verify_func : Callable = None,
                          chromosome_properties : dict = None):
        """
        Wrapping function to create a new Chromosome. Useful when writing new initialisation functions. See Chromosome class.

        Args:
            genes (list): list of initialised Gene objects.
            gene_verify_func (Callable): A function to verify gene compatibility `func(gene,loc,chromosome)` (default = None).
            chromosome_properties (dict): For custom functions, extra params may be given (default = None).

        Returns:
            chromosome: A new Chromosome.
        """
        return Chromosome(genes=genes, gene_verify_func=gene_verify_func, chromosome_properties=chromosome_properties)

    def create_individual(self,
                          fitness_function : Callable,
                          name : str = None,
                          chromosome: Chromosome = None,
                          species_type : str = None,
                          add_to_population : bool = False,
                          individual_properties : dict = None):
        """
        Wrapping function to create a new Individual. Useful when writing new initialisation functions. See Individual class.

        Args:
            fitness_function (Callable): Function with ``func(Chromosome, island, **params)`` signature
            name (str): Name for keeping track of lineage (default = None).
            chromosome (Chromosome): A Chromosome object, initialised (default = None).
            species_type (str) : A unique string to identify the species type, for preventing cross polluting (default = None).
            add_to_population (bool): Add this new individual to the population (default = False).
            individual_properties (dict): For fitness functions, extra params may be given (default = None).

        Returns:
            individual: A new Individual.
        """
        ind = Individual(fitness_function=fitness_function, name=name, chromosome=chromosome, species_type=species_type,
                         individual_properties=individual_properties)
        if add_to_population:
            self.population.append(ind)
        return ind

    def initialise(self, adam : Individual,
               population_size: int = 8,
               initialisation_params : dict = None,
               evaluate_population : bool = True
               ):
        """
        Starts the population by taking an initial individual as template and creating new ones from it.
        Island ``species_type`` is set to adam's species type.

        Args:
            adam (Individual): Individual to clone from.
            population_size (int): Size of population.
            initialisation_params (dict): Custom params for custom initialisation functions (default = None).
            evaluate_population (bool): Evaluate the newly created population (default = True).
        """
        self.species_type = adam.species_type

        if initialisation_params:
            _initialisation_params = initialisation_params
        else:
            _initialisation_params = {}

        self.verbose_logging(f"init: pop_size {population_size}")
        self.verbose_logging(f"init: adam {str(adam)}")
        self.population = self._initialise(adam=adam, n=population_size, island=self, **_initialisation_params)

        if self.save_checkpoint_level == 2:
            self.save_checkpoint(event='init_pre', island=self)

        if evaluate_population:
            for popitem in self.population:
                self.verbose_logging(f"init: eval {str(popitem)}")
                popitem.evaluate(self.function_params, island=self)
                self.unique_genome.append(popitem.unique_genetic_code())
        if self.save_checkpoint_level == 2:
            self.save_checkpoint(event='init_post', island=self)
        self.verbose_logging("init: complete")

        for p in self.population:
            self.__add_to_lineage(p)

    def __add_to_lineage(self, individual : Individual):
        node = {
            'name': individual.name,
            'age': individual.age,
            'fitness': individual.fitness,
            'chromosome': str(individual.chromosome),
            '_individual' : individual
        }
        properties = individual.get_properties()
        if not properties is None:
            node.update(properties)
        self.lineage['nodes'].append(node)

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
            self.verbose_logging(f"migration: customs {str(i)}")
            if species_check:
                if i.species_type != self.species_type:
                    continue
            if force_genetic_diversity:
                if not i.unique_genetic_code() in self.unique_genome:
                    if i.fitness is None or reset_fitness:
                        self.verbose_logging(f"migration: eval {str(i)}")
                        i.evaluate(self.function_params, island=self)
                    self.verbose_logging(f"migration: add {str(i)}")
                    self.population.append(i)
                    self.unique_genome.append(i.unique_genetic_code())
                    self.__add_to_lineage(i)
            else:
                if i.fitness is None or reset_fitness:
                    self.verbose_logging(f"migration: eval {str(i)}")
                    i.evaluate(self.function_params)
                self.verbose_logging(f"migration: add {str(i)}")
                self.population.append(i)
                self.unique_genome.append(i.unique_genetic_code())
                self.__add_to_lineage(i)

        self.verbose_logging("migration: imported")


    def evolve(self, starting_generation : int = 0,
                            n_generations : int = 5,
                            crossover_probability : float = 0.5,
                            mutation_probability : float = 0.25,
                            crossover_params : dict = None,
                            mutation_params : dict = None,
                            elite_selection_params : dict = None,
                            parent_selection_params: dict = None,
                            survivor_selection_params: dict = None,
                            criterion_function : Callable = None,
                            criterion_params : dict = None,
                            pre_generation_check_function : Callable = None,
                            post_generation_check_function: Callable = None,
                            post_evolution_function: Callable = None,
               ) -> Individual:
        """
        Starts the evolutionary run.

        Args:
            starting_generation (int): Starting generation (default = 0).
            n_generations (int): Number of generations to run (default = 5).
            crossover_probability (float): Initial crossover probability (default = 0.5).
            mutation_probability (float): Initial mutation probability (default = 0.25).
            crossover_params (dict): Dict of params for custom crossover function (default = None).
            mutation_params (dict): Dict of params for custom mutation function (default = None).
            selection_params (dict): Dict of params for custom selection function (default = None).
            criterion_function (Callable): A function to evaluate if the desired criterion has been met (default = None).
            criterion_params (dict): Function parameters for criterion (default = None).
            pre_generation_check_function (Callable): A function to perform some custom pre-action at the start of every generation, with signature ``func(generation, island)`` (default = None).
            post_generation_check_function (Callable): A function to perform some custom post-action at the end of every generation, with signature ``func(generation, island)``  (default = None).
            post_evolution_function (Callable): A function to perform some custom post-action after full evolution cycle, with signature ``func(island)`` (default = None).

        Returns:
            Individual: The fittest Individual found.
        """

        if crossover_params:
            _crossover_params = crossover_params
        else:
            _crossover_params = {}

        if mutation_params:
            _mutation_params = mutation_params
        else:
            _mutation_params = {}

        if elite_selection_params:
            _elite_selection_params = elite_selection_params
        else:
            _elite_selection_params = {}

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
        self.verbose_logging(f"evolve: start_generation {g}")
        while g_func(island=self, **criterion_params):
            self.verbose_logging(f"evolve: generation_number {g}")

            if pre_generation_check_function:
                pre_generation_check_function(generation=g, island=self)

            self.__evolutionary_engine(g=g,
                                       elite_selection_params=_elite_selection_params,
                                       parent_selection_params=_parent_selection_params,
                                       survivor_selection_params=_survivor_selection_params,
                                       crossover_probability=crossover_probability,
                                       mutation_probability=mutation_probability,
                                       crossover_params=_crossover_params,
                                       mutation_params=_mutation_params)
            if post_generation_check_function:
                post_generation_check_function(generation=g, island=self)
            g += 1


        best_ind = selection_elites_top_n(island=self, individuals=self.population, n=1)[0]

        self.verbose_logging(f"evolve: end")
        self.verbose_logging(f"evolve: best {str(best_ind)}")

        if post_evolution_function:
            post_evolution_function(island=self)

        return best_ind

    def save_island(self, filepath : str):
        """
        Dumps a pickled self to the given file path.

        Args:
            filepath (str): File path to pickled island.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)
        self.verbose_logging(f"save: file {filepath}")

    def load_island(self, filepath : str):
        """
        Loads a pickled self from the given file path.

        Args:
            filepath (str): File path to pickled island.
        """
        with open(filepath, "rb") as f:
            self.__dict__.update(pickle.load(f))
        self.verbose_logging(f"load: file {filepath}")

    def __evolutionary_engine(self,
                              g,
                              elite_selection_params,
                              parent_selection_params,
                              survivor_selection_params,
                              crossover_probability,
                              mutation_probability,
                              crossover_params,
                              mutation_params):

        if self.save_checkpoint_level == 1:
            self.save_checkpoint(event=f'evolve_pre_{g}', island=self)

        elites = self.elite_selection(island=self,
                                      individuals=self.clone(individuals=self.population, island=self),
                                      **elite_selection_params)
        self.verbose_logging(f"select: elites_count {len(elites)}")

        self.elites.append({'generation' : g, 'elites' : elites})

        # Children are strictly copies or new objects seeing as the have a lineage and parents
        generation_children = list()
        for parents in self.parent_selection(individuals=elites, island=self, **parent_selection_params):
            self.verbose_logging(f"select: parent_count {len(parents)}")
            if self.crossover_prob(crossover_probability=crossover_probability, island=self):
                self.verbose_logging(f"cross: parents {[str(p) for p in parents]}")

                children = self.crossover(island=self,
                                          individuals=self.clone(individuals=parents, island=self),
                                          **crossover_params)


                for child in children:
                    child.reset_name()
                    child.register_parent_names(parents)
                    child.age = 0

                self.verbose_logging(f"cross: offspring_count {len(children)}")
                self.verbose_logging(f"cross: offspring {[str(p) for p in children]}")

                generation_children.extend(children)

        self.children.append({'generation' : g, 'children' : generation_children})

        # Mutants are not strictly copied but rather only modified seeing as the are part of the children list
        generation_mutants = list()
        for mutant in generation_children:
            if self.mutation_prob(mutation_probability=mutation_probability, island=self):
                self.verbose_logging(f"mutate: offspring {str(mutant)}")

                mutated = self.mutation(island=self, individual=mutant, **mutation_params)

                generation_mutants.append(mutated)

        self.mutants.append({'generation': g, 'mutants': generation_mutants})

        offspring_fitnesses = list()

        if self.save_checkpoint_level == 2:
            self.save_checkpoint(event=f'evolve_pre_eval_{g}', island=self)
        for individual in generation_children:
            individual.reset_fitness()
            self.verbose_logging(f"evolve: eval {str(individual)}")
            individual.evaluate(island=self, params=self.function_params)
            offspring_fitnesses.append(individual.fitness)
        if self.save_checkpoint_level == 2:
            self.save_checkpoint(event=f'evolve_post_eval_{g}', island=self)

        for child in generation_children:
            self.__add_to_lineage(child)
            for p in child.parents:
                edge = {
                    'from': p.name,
                    'to': child.name,
                    'generation' : g
                }
                gene_inheritance = child.get('gene_inheritance')
                if not gene_inheritance is None:
                    edge['gene_inheritance'] = gene_inheritance[p.name]
                self.lineage['edges'].append(edge)

        for individual in self.survivor_selection(individuals=generation_children, island=self, **survivor_selection_params):
            if self.allow_twins:
                # Else, add it effectively allowing "twins" to exist
                self.verbose_logging(f"evolve: add {str(individual)}")
                self.population.append(individual)
                self.unique_genome.append(individual.unique_genetic_code())
            elif not individual.unique_genetic_code() in self.unique_genome:
                # If we want a diverse gene pool, this must be true
                self.verbose_logging(f"evolve: add {str(individual)}")
                self.population.append(individual)
                self.unique_genome.append(individual.unique_genetic_code())


        population_fitnesses = [ind.fitness for ind in self.population]
        elite_fitnesses = [ind.fitness for ind in elites]

        if len(offspring_fitnesses) > 0:
            self.generation_info.append(
                {
                    self.__generation_key: g,
                    self.__stat_key: 'offspring',
                    self.__pop_len_key: len(offspring_fitnesses),
                    self.__fitness_mean_key: np.mean(offspring_fitnesses),
                    self.__fitness_std_key: np.std(offspring_fitnesses),
                    self.__fitness_min_key: min(offspring_fitnesses),
                    self.__fitness_max_key: max(offspring_fitnesses),
                    self.__most_fit_key: selection_elites_top_n(island=self, individuals=generation_children, n=1)[0].name,
                    self.__least_fit_key : selection_elites_top_n(island=self, individuals=generation_children, n=1)[-1].name
                }
            )
            self.verbose_logging(f"evolve: stats {self.generation_info[-1]}")

        self.generation_info.append(
            {
                self.__generation_key: g,
                self.__stat_key: 'elites',
                self.__pop_len_key: len(elite_fitnesses),
                self.__fitness_mean_key: np.mean(elite_fitnesses),
                self.__fitness_std_key: np.std(elite_fitnesses),
                self.__fitness_min_key: min(elite_fitnesses),
                self.__fitness_max_key: max(elite_fitnesses),
                self.__most_fit_key: selection_elites_top_n(island=self, individuals=elites, n=1)[0].name,
                self.__least_fit_key : selection_elites_top_n(island=self, individuals=elites, n=1)[-1].name
            }
        )

        self.verbose_logging(f"evolve: stats {self.generation_info[-1]}")

        self.generation_info.append(
            {
                self.__generation_key: g,
                self.__stat_key: 'population',
                self.__pop_len_key: len(population_fitnesses),
                self.__fitness_mean_key: np.mean(population_fitnesses),
                self.__fitness_std_key: np.std(population_fitnesses),
                self.__fitness_min_key: min(population_fitnesses),
                self.__fitness_max_key: max(population_fitnesses),
                self.__most_fit_key: selection_elites_top_n(island=self, individuals=self.population, n=1)[0].name,
                self.__least_fit_key: selection_elites_top_n(island=self, individuals=self.population, n=1)[-1].name
            }
        )

        self.verbose_logging(f"evolve: stats {self.generation_info[-1]}")

        for i in self.population:
            i.birthday()

        self.generation_count = g

        if self.save_checkpoint_level == 1:
            self.save_checkpoint(event=f'evolve_post_{g}', island=self)

    def write_lineage_json(self, filename : str):
        """
        Dumps the lineage safely to JSON file.

        Args:
            filename (str): Output file.
        """
        nodes = list()
        for n in self.lineage['nodes']:
            nodes.append({k:v for k, v in n.items() if not k.startswith('_')})
        import json
        with open(filename, 'w', newline='', encoding='utf8') as output_file:
            json.dump({'nodes' : nodes, 'edges' : self.lineage['edges']}, output_file)


    def write_report(self, filename : str, output_json : bool = False):
        """
        Write the generational history to CSV (or JSON) file.

        Args:
            filename (str): Output file.
            output_json (bool): Write as JSON instead of CSV (default = False).
        """
        if output_json:
            import json
            with open(filename, 'w', newline='', encoding='utf8') as output_file:
                json.dump(self.generation_info, output_file)
        else:
            import csv
            keys = [
                self.__generation_key,
                self.__stat_key,
                self.__pop_len_key,
                self.__fitness_mean_key,
                self.__fitness_std_key,
                self.__fitness_min_key,
                self.__fitness_max_key,
                self.__most_fit_key,
                self.__least_fit_key
            ]
            with open(filename, 'w', newline='', encoding='utf8') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.generation_info)



    def verbose_logging(self, event_message):
        if self.verbose:
            logging.info(event_message, extra={'island' : self.name})
        if self.logging_function:
            self.logging_function(event_message=event_message, island=self)
