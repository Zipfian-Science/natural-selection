from numpy import random

def random_int(gene):
    return random.randint(low=gene.gene_min, high=gene.gene_max)

def random_int_step(gene):
    stepped_value = random.randint(low=gene.step_lower_bound, high=gene.step_upper_bound)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = random.randint(low=gene.step_lower_bound, high=gene.step_upper_bound)

    return stepped_value

def random_gaussian(gene):
    stepped_value = random.normal(loc=gene.mu, scale=gene.sig)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = random.normal(loc=gene.mu, scale=gene.sig)

    return stepped_value

def random_uniform(gene):
    return random.uniform(low=gene.gene_min, high=gene.gene_max)

def random_gaussian_step(gene):
    stepped_value = gene.value + random.normal(loc=gene.mu, scale=gene.sig)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = gene.value + random.normal(loc=gene.mu, scale=gene.sig)

    return stepped_value

def random_uniform_step(gene):
    stepped_value = gene.value + random.uniform(low=gene.step_lower_bound, high=gene.step_upper_bound)

    while stepped_value > gene.gene_max or stepped_value < gene.gene_min:
        stepped_value = gene.value + random.uniform(low=gene.step_lower_bound, high=gene.step_upper_bound)

    return stepped_value

def random_choice(gene):

    if not 'choices' in gene.__dict__:
        raise KeyError("'choices' not defined in this gene, please include a list values!")
    return random.choice(gene.choices)