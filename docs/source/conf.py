# -- Path Setup ----------------------------------------------------------------------

# if extensions (or modules to document with autodoc) are in another directory add these directories
# to sys.path here. If the directory is relative to the documentation root, use
# os.path.abspath to make it absolute
import os
import sys
# sys.path.insert(0, os.path.abspath('../../'))
print("CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======S")
print(os.getcwd())
print("os.path.abspath(os.path.join(__file__, ../..))",os.path.abspath( os.path.join(__file__, "../..")
))
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath(
    os.path.join(__file__, "../..")
))


# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'push'
copyright = '2021, Graziella'
author = 'Graziella'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "nbsphinx"
]
# autodoc_mock_imports = ["django"]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
autosummary_generate = True  # Turn on sphinx.ext.autosummary