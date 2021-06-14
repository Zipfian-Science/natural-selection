# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import json
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(1500)

###########################
# For a full list of toc directives:
# https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
###########################

# -- Project information -----------------------------------------------------

project = 'Natural Selection'
copyright = '2021, Zipfian Science'
author = 'Zipfian Science'

# The full version, including alpha/beta/rc tags
with open("../../version.json", "r") as f:
    release_version = json.load(f)

# Need to take deployment version bumping into account
release_version['patch'] -= 1
release = '{major}.{minor}.{patch}'.format(**release_version)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinxcontrib.napoleon',
              'sphinx.ext.todo',
              'sphinx.ext.githubpages',
              'sphinx.ext.autosectionlabel'
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme ='alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

latex_elements = {
    'papersize' : 'letterpaper',
    'pointsize' : '10pt',
    'preamble' : '',
    'figure_align' : 'htpb',
}

# The master toctree document.
master_doc = 'index'
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

html_sidebars = {'**': ['localtoc.html', 'sourcelink.html', 'searchbox.html']}
