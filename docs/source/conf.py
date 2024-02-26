# -- Path Setup ----------------------------------------------------------------------

# if extensions (or modules to document with autodoc) are in another directory add these directories
# to sys.path here. If the directory is relative to the documentation root, use
# os.path.abspath to make it absolute
import os
import sys
# sys.path.insert(0, os.path.abspath('../../'))
# print("CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======CWD======S")
print(os.getcwd())
path = os.path.abspath(os.path.join(__file__,"..","..",".."))
lib_path = os.path.abspath(os.path.join(__file__,"..","..","..","push", "lib"))
bayes_path = os.path.abspath(os.path.join(__file__,"..","..","..","push", "bayes"))

print("path: ",path)

def list_directories(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories
directories_list = list_directories(path)


print("List of directories in {}: {}".format(path, directories_list))

sys.path.append(path)
sys.path.append(lib_path)
sys.path.append(bayes_path)


from inspect import getsourcefile

# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        quiet=True,
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)


# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'PusH'
copyright = 'The Push authors'
author = 'Daniel Huang'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinx.ext.napoleon',
]
autodoc_mock_imports = ["tqdm","numpy"]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']
nbsphinx_allow_errors = True
templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
autosummary_generate = True  # Turn on sphinx.ext.autosummary