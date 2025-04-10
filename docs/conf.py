# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import datetime

# Add archetypax to path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "riemannax"
author = "mary"
copyright = f"{datetime.now().year}, {author}"

# Import package to get version
import riemannax

version = riemannax.__version__
release = riemannax.__version__

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
    "myst_parser",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add mappings for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_baseurl = "https://riemannax.readthedocs.io/"

# Use relative URLs for resources
html_use_index = True
html_split_index = False
html_copy_source = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

if not os.path.exists("_static"):
    os.makedirs("_static")

on_rtd = os.environ.get("READTHEDOCS", None) == True
if on_rtd:
    pass
else:
    pass

html_context = {
    "display_version": True,
}
