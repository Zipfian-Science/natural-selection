__version__ = '0.2.27'
__date__ = "2022-06-29"

from time import gmtime
import multiprocessing as mp
import random
import logging
from datetime import datetime
import warnings as w
from typing import Callable, Any, Iterable, List, Union, Type
import pickle

import numpy as np

from natural_selection.genetic_algorithms import Gene, Chromosome, Individual
from natural_selection.genetic_programs import Node, GeneticProgram, random_generate

from natural_selection.genetic_algorithms.operators.initialisation import initialise_population_random, alien_spawn_default
from natural_selection.genetic_algorithms.operators.selection import selection_elites_top_n, selection_parents_two, selection_survivors_all
from natural_selection.genetic_algorithms.operators.crossover import crossover_two_uniform
from natural_selection.genetic_algorithms.operators.mutation import mutation_randomize
from natural_selection.genetic_algorithms.utils.probability_functions import crossover_prob_function_classic, mutation_prob_function_classic
from natural_selection.genetic_algorithms.utils import clone_classic, default_save_checkpoint_function, GeneticAlgorithmError, evaluate_individual

from natural_selection.genetic_programs.operators.initialisation import initialise_population_full_method
from natural_selection.genetic_programs.utils import GeneticProgramError
import natural_selection.genetic_programs.node_operators as gp_func


class Island:
    """
    A simple Island to perform a Genetic Algorithm. By default the selection, mutation, crossover, and probability functions
    default to the classic functions.

    Args:
        function_params (dict): The parameters for the fitness function (default = None).
        maximise_function (bool): If True, the fitness value is maximised, False will minimise the function fitness (default = True).
        parent_selection (Callable): Function for selecting individuals for crossover (default = None).
        initialisation_function (Callable): A function for randomly creating new individuals from the given adam.
        parent_combination (Callable): Function for combining parents for crossover (default = None).
        crossover_function (Callable): Function for crossover (default = None).
        mutation_function (Callable): Function for mutation (default = None).
        crossover_prob_function (Callable): Random probability function for crossover (default = None).
        mutation_prob_function (Callable): Random probability function for mutation (default = None).
        clone_function (Callable): Function for cloning (default = None).
        survivor_selection_function (Callable): Function for selecting survivors (default = None).
        alien_spawn_function (Callable): Function for spawning new aliens during each generation (default = None).
        random_seed (int): Random seed for random and Numpy generators, set to None for no seed (default = None).
        name (str): General name for island, useful when working with multiple islands (default = None).
        verbose (bool): Print all information (default = None).
        logging_function (Callable): Function for custom message logging, such as server logging (default = None).
        save_checkpoint_function (Callable): Function for custom checkpoint saving (default = None).
        filepath (str): If a filepath is specified, the pickled island is loaded from it, skipping the rest of initialisation (default = None).
        save_checkpoint_level (int): Level of checkpoint saving 0 = none, 1 = per generation, 2 = per evaluation (default = 0).
        core_count (int, float): Number of cores to split evaluation on, for all cores, set -1, use float for fractional (default = 1).
        allow_twins (bool): Only add new offspring to the population if they have a unique chromosome (default = False).

    Attributes:
        unique_genome (list): List of unique chromosomes.
        generation_info (list): List of dicts detailing info for every generation.
        population (list): The full population of members.
        parents (list): All parents selected during the run.
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
                 parent_selection : Callable = selection_elites_top_n,
                 parent_combination : Callable = selection_parents_two,
                 crossover_function : Callable = crossover_two_uniform,
                 mutation_function : Callable = mutation_randomize,
                 crossover_prob_function : Callable = crossover_prob_function_classic,
                 mutation_prob_function : Callable = mutation_prob_function_classic,
                 survivor_selection_function: Callable = selection_survivors_all,
                 alien_spawn_function: Callable = alien_spawn_default,
                 clone_function : Callable = clone_classic,
                 random_seed: int = None,
                 name : str = None,
                 verbose : bool = True,
                 logging_function : Callable = None,
                 save_checkpoint_function: Callable = default_save_checkpoint_function,
                 filepath : str = None,
                 save_checkpoint_level : int = 0,
                 core_count : Union[int, float] = 1,
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

        self.verbose_logging(f"island: create v{__version__}")
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
        self.parent_selection = parent_selection
        self.verbose_logging(f"island: parent_selection {parent_selection.__name__}")
        self.parent_combination = parent_combination
        self.verbose_logging(f"island: parent_combination {parent_combination.__name__}")
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

        self.alien_spawn_function = alien_spawn_function
        self.verbose_logging(f"island: survivor_selection_function {survivor_selection_function.__name__}")

        self.save_checkpoint = save_checkpoint_function
        self.verbose_logging(f"island: save_checkpoint_function {save_checkpoint_function.__name__}")

        self.allow_twins = allow_twins
        self.verbose_logging(f"island: allow_twins {allow_twins}")

        self.random_seed = random_seed
        self.verbose_logging(f"island: random_seed {random_seed}")

        if isinstance(core_count, int):
            if core_count < 1:
                self.core_count = mp.cpu_count()
            else:
                self.core_count = core_count
        elif isinstance(core_count, float) and core_count < 1.0 and core_count > 0.0:
            self.core_count = round(core_count * mp.cpu_count())
        else:
            raise ValueError('The value of core_count must be of type float or int, and 0 < core_count < 1 if float')

        self.verbose_logging(f"island: core_count {core_count} as {self.core_count}")

        self.elites = list()
        self.parents = list()
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

        if '<lambda>' in repr(parent_selection):
            w.warn("WARNING: 'parent_selection' lambda can not be pickled using standard libraries.")

        if '<lambda>' in repr(parent_combination):
            w.warn("WARNING: 'parent_combination' lambda can not be pickled using standard libraries.")

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

        if '<lambda>' in repr(alien_spawn_function):
            w.warn("WARNING: 'alien_spawn_function' lambda can not be pickled using standard libraries.")

        if logging_function and '<lambda>' in repr(logging_function):
            w.warn("WARNING: 'logging_function' lambda can not be pickled using standard libraries.")

        if save_checkpoint_function and '<lambda>' in repr(save_checkpoint_function):
            w.warn("WARNING: 'save_checkpoint_function' lambda can not be pickled using standard libraries.")

        self.__island_properties = dict()

    def create_node(self,
                    label: str = None,
                    arity: int = 1,
                    operator: gp_func.Operator = None,
                    is_terminal=False,
                    terminal_value=None,
                    children: List = None):
        """
        Wrapping function to create a new node. Useful when writing new initialisation functions. See Node class.

        Args:
            label (str): Optionally set the label, only used for variable terminals (default = None).
            arity (int): Optionally set the function arity, the norm being 2 for functions (default = 1).
            operator (Operator): If the node is a function, set the operator (default = None).
            is_terminal (bool): Explicitly define if the node is a terminal (default = None).
            terminal_value (Any): Only set if the node is terminal and a constant value (default = None).
            children (list): Add a list of child nodes, list length must match arity (default = None).

        Returns:
            node: A new Node object.
        """
        return Node(label=label,
             arity=arity,
             operator=operator,
             is_terminal=is_terminal,
             terminal_value=terminal_value,
             children=children)

    def create_gene(self,
                 name : str,
                 value : Any,
                 randomise_function: Callable,
                 gene_max : Any = None,
                 gene_min : Any = None,
                 mu : Any = None,
                 sig: Any = None,
                 step_lower_bound : Any = None,
                 step_upper_bound : Any = None,
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
            step_lower_bound (Any, numeric type): For uniform stepping functions, defines lower bound of range (default = None).
            step_upper_bound (Any, numeric type): For uniform stepping functions, defines upper bound of range (default = None).
            choices (Iterable): List of choices, categorical or not, to randomly choose from (default = None).
            gene_properties (dict): For custom random functions, extra params may be given (default = None).
        Returns:
            gene: A new Gene object.
        """
        return Gene(name=name, value=value, randomise_function=randomise_function, gene_max=gene_max, gene_min=gene_min,
                    mu=mu, sig=sig, step_lower_bound=step_lower_bound, step_upper_bound=step_upper_bound, choices=choices, gene_properties=gene_properties)

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

    def create_genetic_program(self,
                               fitness_function: Callable = None,
                               node_tree: Node = None,
                               operators: List[Union[Type, gp_func.Operator]] = None,
                               terminals: List[Union[str, int, float]] = None,
                               max_depth: int = 3,
                               min_depth: int = 1,
                               growth_mode: str = 'grow',
                               terminal_prob: float = 0.5,
                               tree_generator: Callable = random_generate,
                               name: str = None,
                               species_type: str = None,
                               add_to_population: bool = False,
                               program_properties: dict = None
                               ):
        """
        Wrapping function to create a new GeneticProgram. Useful when writing new initialisation functions. See GeneticProgram class.

        Args:
            fitness_function (Callable): Function with ``func(Node, island, **params)`` signature (default = None).
            node_tree (Node): A starting node tree (default = None).
            operators (list): List of all operators that nodes can be constructed from (default = None).
            terminals (list): List of all terminals that can be included in the node tree, can be numeric or strings for variables (default = None).
            max_depth (int): Maximum depth that node tree can grow (default = 3).
            min_depth (int): Minimum depth that node tree must be (default = 1).
            growth_mode (str): Type of tree growth method to use, "full" or "grow" (default = "grow").
            terminal_prob (float): Probability of a generated node is a terminal (default = 0.5).
            tree_generator (Callable): Function with to create the tree.
            name (str): Name for keeping track of lineage (default = None).
            species_type (str) : A unique string to identify the species type, for preventing cross polluting (default = None).
            add_to_population (bool): Add this new program to the population (default = False).
            program_properties (dict): For fitness functions, extra params may be given (default = None).

        Returns:
            program: Newly created GeneticProgram.
        """
        gp = GeneticProgram(fitness_function=fitness_function,
                              node_tree=node_tree,
                              operators=operators,
                              terminals=terminals,
                              max_depth=max_depth,
                              min_depth=min_depth,
                              growth_mode=growth_mode,
                              terminal_prob=terminal_prob,
                              tree_generator=tree_generator,
                              name=name,
                              species_type=species_type,
                              program_properties=program_properties
                              )
        if add_to_population:
            self.population.append(gp)
        return gp


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

    def initialise(self, adam : Union[Individual,GeneticProgram],
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
            for popitem in self.__evaluate_individuals(self.population):
                self.unique_genome.append(popitem.unique_genetic_code())
        if self.save_checkpoint_level == 2:
            self.save_checkpoint(event='init_post', island=self)
        self.verbose_logging("init: complete")

        for p in self.population:
            p.history.append({"init_island": self.name})
            self.__add_to_lineage(p)

    def __add_to_lineage(self, individual : Union[Individual, GeneticProgram]):
        c = str(individual.chromosome) if isinstance(individual,Individual) else str(individual.node_tree)
        node = {
            'name': individual.name,
            'age': individual.age,
            'fitness': individual.fitness,
            'chromosome': c,
            '_individual' : individual
        }
        properties = individual.get_properties()
        if not properties is None:
            node.update(properties)
        self.lineage['nodes'].append(node)

    def import_migrants(self, migrants : list,
                        reset_fitness : bool = False,
                        species_check : bool = True,
                        allow_twins : bool = True):
        """
        Imports a list of individuals, with option to re-evaluate them.
        Skips the individual if the genetic code is already in the population.

        Args:
            migrants (list): List of Individuals.
            reset_fitness (bool): Reset the fitness of new members and evaluate them (default = False).
            species_check (bool): Safely check that imported members are compatible with population  (default = True).
            allow_twins (bool): Only add migrants to the population if they have a unique chromosome (default = True).
        """
        migrants_for_testing = list()
        migrants_for_adding = list()

        for i in migrants:
            self.verbose_logging(f"migration: customs {str(i)}")
            if species_check:
                if i.species_type != self.species_type:
                    continue
            if allow_twins or not i.unique_genetic_code(force_update=True) in self.unique_genome:
                    if i.fitness is None or reset_fitness:
                        migrants_for_testing.append(i)
                    else:
                        migrants_for_adding.append(i)

        migrants_for_testing = self.__evaluate_individuals(migrants_for_testing)
        migrants_for_adding.extend(migrants_for_testing)

        for i in migrants_for_adding:
            self.verbose_logging(f"migration: add {str(i)}")
            i.history.append({"migration_island": self.name})
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
               parent_selection_params : dict = None,
               parent_combination_params: dict = None,
               survivor_selection_params: dict = None,
               alien_spawn_params: dict = None,
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
            alien_spawn_params (dict): Dict of params for alien spawn function (default = None).
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

        if parent_selection_params:
            _parent_selection_params = parent_selection_params
        else:
            _parent_selection_params = {}

        if parent_combination_params:
            _parent_combination_params = parent_combination_params
        else:
            _parent_combination_params = {}

        if survivor_selection_params:
            _survivor_selection_params = survivor_selection_params
        else:
            _survivor_selection_params = {}

        if alien_spawn_params:
            _alien_spawn_params = alien_spawn_params
        else:
            _alien_spawn_params = {}

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
                                       parent_selection_params=_parent_selection_params,
                                       parent_combination_params=_parent_combination_params,
                                       survivor_selection_params=_survivor_selection_params,
                                       crossover_probability=crossover_probability,
                                       mutation_probability=mutation_probability,
                                       crossover_params=_crossover_params,
                                       mutation_params=_mutation_params,
                                       alien_spawn_params=_alien_spawn_params)
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

    def __evaluate_individuals(self, individuals):
        if self.core_count == 1:
            for individual in individuals:
                self.verbose_logging(f"eval: {str(individual)}")
                individual.evaluate(island=self, params=self.function_params)
            return individuals

        with mp.Pool(self.core_count) as p:
            for f, i in zip(p.starmap(evaluate_individual, [(i, self, self.function_params) for i in individuals]), individuals):
                i.fitness = f

        return individuals

    def __evolutionary_engine(self,
                              g,
                              parent_selection_params,
                              parent_combination_params,
                              survivor_selection_params,
                              crossover_probability,
                              mutation_probability,
                              crossover_params,
                              mutation_params,
                              alien_spawn_params):

        if self.save_checkpoint_level == 1:
            self.save_checkpoint(event=f'evolve_pre_{g}', island=self)

        selected_parents = self.parent_selection(island=self,
                                       individuals=self.clone(individuals=self.population, island=self),
                                       **parent_selection_params)
        self.verbose_logging(f"select: selected_parents_count {len(selected_parents)}")

        self.parents.append({'generation' : g, 'parents' : selected_parents})

        # Children are strictly copies or new objects seeing as the have a lineage and parents
        generation_children = list()
        for parents in self.parent_combination(individuals=selected_parents, island=self, **parent_combination_params):
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

        for individual in self.__evaluate_individuals(generation_children):
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
            if self.allow_twins or not individual.unique_genetic_code(force_update=True) in self.unique_genome:
                # If we want a diverse gene pool, this must be true
                self.verbose_logging(f"evolve: add {str(individual)}")
                self.population.append(individual)
                self.unique_genome.append(individual.unique_genetic_code())


        population_fitnesses = [ind.fitness for ind in self.population]
        selected_parents_fitnesses = [ind.fitness for ind in selected_parents]

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
                self.__stat_key: 'parents',
                self.__pop_len_key: len(selected_parents_fitnesses),
                self.__fitness_mean_key: np.mean(selected_parents_fitnesses),
                self.__fitness_std_key: np.std(selected_parents_fitnesses),
                self.__fitness_min_key: min(selected_parents_fitnesses),
                self.__fitness_max_key: max(selected_parents_fitnesses),
                self.__most_fit_key: selection_elites_top_n(island=self, individuals=selected_parents, n=1)[0].name,
                self.__least_fit_key : selection_elites_top_n(island=self, individuals=selected_parents, n=1)[-1].name
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

        elites = selection_elites_top_n(island=self, individuals=self.population, n=4)
        elite_fitnesses = [ind.fitness for ind in elites]
        self.elites.append({'generation': g, 'elites': elites})

        self.generation_info.append(
            {
                self.__generation_key: g,
                self.__stat_key: 'elites',
                self.__pop_len_key: 4,
                self.__fitness_mean_key: np.mean(elite_fitnesses),
                self.__fitness_std_key: np.std(elite_fitnesses),
                self.__fitness_min_key: min(elite_fitnesses),
                self.__fitness_max_key: max(elite_fitnesses),
                self.__most_fit_key: elites[0].name,
                self.__least_fit_key: elites[-1].name
            }
        )

        self.verbose_logging(f"evolve: stats {self.generation_info[-1]}")

        for i in self.population:
            i.birthday()

        new_aliens = self.alien_spawn_function(**alien_spawn_params ,island=self)

        if len(new_aliens) > 0:
            alien_fitnesses = list()
            for alien in self.__evaluate_individuals(new_aliens):
                if self.allow_twins or not alien.unique_genetic_code(force_update=True) in self.unique_genome:
                    # Else, add it effectively allowing "twins" to exist
                    alien_fitnesses.append(alien.fitness)
                    self.verbose_logging(f"evolve: add {str(alien)}")
                    alien.history.append({"alien_init_island": self.name})
                    self.population.append(alien)
                    self.unique_genome.append(alien.unique_genetic_code())
                    self.__add_to_lineage(alien)

            self.generation_info.append(
                {
                    self.__generation_key: g,
                    self.__stat_key: 'aliens',
                    self.__pop_len_key: len(alien_fitnesses),
                    self.__fitness_mean_key: np.mean(alien_fitnesses),
                    self.__fitness_std_key: np.std(alien_fitnesses),
                    self.__fitness_min_key: min(alien_fitnesses),
                    self.__fitness_max_key: max(alien_fitnesses),
                    self.__most_fit_key: selection_elites_top_n(island=self, individuals=new_aliens, n=1)[0].name,
                    self.__least_fit_key: selection_elites_top_n(island=self, individuals=new_aliens, n=1)[-1].name
                }
            )

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

    def add_new_property(self, key : str, value : Any):
        """
        Method to add new properties (attributes).

        Args:
            key (str): Name of property.
            value (Any): Anything.
        """
        if self.__island_properties:
            self.__island_properties.update({key: value})
        else:
            self.__island_properties = {key: value}
        self.__dict__.update({key: value})

    def get_properties(self):
        """
        Gets a dict of the custom properties that were added at initialisation or the `add_new_property` method.

        Returns:
            dict: All custom properties.
        """
        return self.__island_properties

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

    def verbose_logging(self, event_message):
        if self.verbose:
            logging.info(event_message, extra={'island' : self.name})
        if self.logging_function:
            self.logging_function(event_message=event_message, island=self)


def get_random_string(length : int = 8, include_numeric=False) -> str:
    """
    Generate a random string with a given length. Used mainly for password generation.
    Args:
        length (int): Length to generate.
        include_numeric (bool): Include numbers?

    Returns:
        str: Random character string.
    """
    import random
    import string

    letters = string.ascii_letters
    if include_numeric:
        letters = f'{letters}{string.digits}'
    return ''.join(random.choice(letters) for i in range(length))