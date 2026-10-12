"""Sphinx configuration for the difftree documentation."""
import difftree

project = "difftree"
author = "John R. Kitchin"
copyright = "2026, John R. Kitchin"
release = difftree.__version__
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "make_logo.py"]

# MyST: $...$ math, \begin{equation*} blocks, ::: fences
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# Notebooks are committed with outputs; executing them takes minutes.
nb_execution_mode = "off"

bibtex_bibfiles = ["../references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

autodoc_default_options = {"show-inheritance": False}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

html_theme = "furo"
html_title = "difftree"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/jkitchin/jax-difftree/",
    "source_branch": "main",
    "source_directory": "docs/",
}
