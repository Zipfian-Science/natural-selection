from numpy import random

def random_int(gene):
    """
    Random integer from range.

    Args:
        gene (Gene): A gene with a set `gene_min` and `gene_max`.

    Returns:
        int: Random number.
    """
    return random.randint(low=gene.gene_min, high=gene.gene_max)

def random_int_step(gene):
    """
    Random integer step from a range. Stepping can be  solely to the right (increment), left (decrement), or in both.
    This depends on `step_lower_bound` and `step_upper_bound` values of the gene. For left steps only, both need to be negative.
    For right only, both should be positive values.

    stepped_value = value + random(int, int)

    Stepped values are still bound to the `gene_min` and `gene_max` constraints and will continue setting until condition is matched.

    Args:
         gene (Gene): A gene with a set `step_lower_bound` and `step_upper_bound`.

    Returns:
        int: Stepped value.
    """
    stepped_value = gene.value + random.randint(low=gene.step_lower_bound, high=gene.step_upper_bound)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = gene.value + random.randint(low=gene.step_lower_bound, high=gene.step_upper_bound)

    return stepped_value

def random_gaussian(gene):
    """
    Random float from gaussian dist.

    Args:
        gene (Gene): A gene with a set `mu` and `sig`.

    Returns:
        float: Random number.
    """
    stepped_value = random.normal(loc=gene.mu, scale=gene.sig)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = random.normal(loc=gene.mu, scale=gene.sig)

    return stepped_value

def random_uniform(gene):
    """
    Random float from uniform dist.

    Args:
        gene (Gene): A gene with a set `gene_min` and `gene_max`.

    Returns:
        float: Random number.
    """
    return random.uniform(low=gene.gene_min, high=gene.gene_max)

def random_gaussian_step(gene):
    """
    Random gaussian step. Stepping can be  solely to the right (increment), left (decrement), or in both.
    This depends on `mu` and `sig` values of the gene. For left steps only, both need to be negative.
    For right only, both should be positive values. These need to be considered in advance.

    stepped_value = value + gaussian(mu, sig)

    Stepped values are still bound to the `gene_min` and `gene_max` constraints and will continue setting until condition is matched.

    Args:
         gene (Gene): A gene with a set `mu` and `sig`.

    Returns:
        float: Stepped value.
    """
    stepped_value = gene.value + random.normal(loc=gene.mu, scale=gene.sig)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = gene.value + random.normal(loc=gene.mu, scale=gene.sig)

    return stepped_value

def random_uniform_step(gene):
    """
    Random uniform step from a range. Stepping can be  solely to the right (increment), left (decrement), or in both.
    This depends on `step_lower_bound` and `step_upper_bound` values of the gene. For left steps only, both need to be negative.
    For right only, both should be positive values.

    stepped_value = value + uniform(l, u)

    Stepped values are still bound to the `gene_min` and `gene_max` constraints and will continue setting until condition is matched.

    Args:
         gene (Gene): A gene with a set `step_lower_bound` and `step_upper_bound`.

    Returns:
        float: Stepped value.
    """
    stepped_value = gene.value + random.uniform(low=gene.step_lower_bound, high=gene.step_upper_bound)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = gene.value + random.uniform(low=gene.step_lower_bound, high=gene.step_upper_bound)

    return stepped_value

def random_choice(gene):
    """
    Randomly select a object, such as strings, from a list. Gene must have defined `choices` list.

    Args:
        gene (Gene): A gene with a set `choices` list.

    Returns:
        object: Selected choice.
    """
    if not 'choices' in gene.__dict__:
        raise KeyError("'choices' not defined in this gene, please include a list values!")
    return random.choice(gene.choices)