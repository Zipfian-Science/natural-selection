from numpy import random

def random_int(gene):
    return random.randint(low=gene.gene_min, high=gene.gene_max)

def random_gaussian(gene):
    # Set to mean and std of the gene if available, else, standard normal used
    if 'mu' in gene.__dict__:
        mu = gene.mu
    else:
        mu = 0
    if 'sig' in gene.__dict__:
        sig = gene.sig
    else:
        sig = 1
    return random.normal(loc=mu, scale=sig)

def random_uniform(gene):
    return random.uniform(low=gene.gene_min, high=gene.gene_max)

def random_gaussian_step(gene):
    # Set to mean and std of the gene if available, else, standard normal used
    if 'mu' in gene.__dict__:
        mu = gene.mu
    else:
        mu = 0
    if 'sig' in gene.__dict__:
        sig = gene.sig
    else:
        sig = 1
    return gene.value + random.normal(loc=mu, scale=sig)

def random_uniform_step(gene):
    return gene.value + random.uniform(low=gene.gene_min, high=gene.gene_max)

def random_choice(gene):

    if not 'choices' in gene.__dict__:
        raise KeyError("'choices' not defined in this gene, please include a list values!")
    return random.choice(gene.choices)