# Natural Selection
```
      ,(*                                         
           @@                                     
*@       @@% *@                                   
*@     @@   %@ @                                  
 @@/ @@   @@   @@                                 
   @@@(,@(   @/ @@@@@@@&@@@@@                     
                 @ @&  @@  /@@@#                  
                 /@  @@  ,@@   @@                 
                  ,@@   @@   @@  @                
                    %@@@   @@    @@@@@@@@@@@@@    
                          ,,      @  @@  @@  &@@@ 
                                  %@@  @@  &@@  @@
                                   @%@@  &@@     @
                                    ,@,%@@        
                                       @@@@@@     
             _                   _ 
 _ __   __ _| |_ _   _ _ __ __ _| |
| '_ \ / _` | __| | | | '__/ _` | |
| | | | (_| | |_| |_| | | | (_| | |
|_| |_|\__,_|\__|\__,_|_|  \__,_|_|                                   
          _           _   _             
 ___  ___| | ___  ___| |_(_) ___  _ __  
/ __|/ _ \ |/ _ \/ __| __| |/ _ \| '_ \ 
\__ \  __/ |  __/ (__| |_| | (_) | | | |
|___/\___|_|\___|\___|\__|_|\___/|_| |_|
                                        
by Zipfian Science                               
```
Python tools for creating and running Evolutionary Algorithm (EA) experiments by [Zipfian Science](https://zipfian.science/).

* For documentation, see [docs](http://docs.zipfian.science/natural-selection/index.html).
* Source on [GitHub](https://github.com/Zipfian-Science/natural-selection).
* For history, see [changelog](http://docs.zipfian.science/natural-selection/changelog.html#changelog-page)
## Install

```shell script
$ pip install natural-selection
```

## And use

```python
from natural_selection.genetic_algorithms import Gene, Chromosome, Individual, Island
from natural_selection.genetic_algorithms.utils.random_functions import random_int, random_gaussian

# Create a gene
g_1 = Gene(name="test_int", value=3, gene_max=10, gene_min=1, randomise_function=random_int)
g_2 = Gene(name="test_real", value=0.5, gene_max=1.0, gene_min=0.1, randomise_function=random_gaussian)

# Add a list of genes to a genome
gen = Chromosome([g_1, g_2])

# Next, create an individual to carry these genes and evaluate them
fitness_function = lambda island, individual, x, y: individual.chromosome[0].value * x + individual.chromosome[0].value * y
adam = Individual(fitness_function, name="Adam", chromosome=gen)

# Now we can create an island for running the evolutionary process
# Notice the fitness function parameters are given here.
params = dict()
params['x'] = 0.5
params['y'] = 0.2
isolated_island = Island(function_params=params)

# Using a single individual, we can create a new population
isolated_island.initialise(adam, population_size=5)

# And finally, we let the randomness of life do its thing: optimise
best_individual = isolated_island.evolve(n_generations=5)

# After running for a few generations, we have an individual with the highest fitness
fitness = best_individual.fitness
genes = best_individual.chromosome

for gene in genes:
  print(gene.name, gene.value)
```

## Release

- Date: {pypi_metdata_release_date}
- Version: {pypi_metdata_version_number}

