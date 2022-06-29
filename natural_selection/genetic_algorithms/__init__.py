# -*- coding: utf-8 -*-
"""Basic classes for running Genetic Algorithms.
"""
__author__ = "Justin Hocking"
__copyright__ = "Copyright 2022, Zipfian Science"
__credits__ = []
__license__ = ""
__version__ = "0.2.0"
__maintainer__ = "Justin Hocking"
__email__ = "justin.hocking@zipfian.science"
__status__ = "Development"


import uuid
from typing import Callable, Any, Iterable
import pickle
from collections import OrderedDict
import copy
import warnings as w

from natural_selection.genetic_algorithms.utils import GeneticAlgorithmError


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

    def get_properties(self):
        """
        Gets a dict of the custom properties that were added at initialisation or the `add_new_property` method.

        Returns:
            dict: All custom properties.
        """
        return self.__gene_properties

    def get(self, key: str, default=None):
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

    def to_list(self) -> list:
        """
        Helper function to convert chromosome into a Python list of the gene values.

        Returns:
            list: Values of the genes in order, as a list.
        """
        return [gene.value for gene in self.genes]


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
            self.load(filepath=filepath)
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

    def unique_genetic_code(self, force_update : bool = False) -> str:
        """
        Gets the unique genetic code, generating if it is undefined.

        Args:
            force_update (bool): Force update of genetic_code property (default = False).

        Returns:
            str: String name of Chromosome.
        """
        if self.genetic_code is None or force_update:
            self.genetic_code = repr(self.chromosome)
        return self.genetic_code

    def save(self, filepath : str):
        """
        Save an individual to a pickle file.

        Args:
            filepath (str): File path to write to.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath : str):
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

