# # Configuration file for the Sphinx documentation builder.

# # -- Project information

# project = 'push'
# copyright = '2021, Graziella'
# author = 'Dan Huang, Chris Camano, Jonathan Tsegaye, Jonathan Jones'

# release = '0.1'
# version = '0.1.0'

# # -- General configuration

# extensions = [
#     'sphinx.ext.duration',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.intersphinx',
# ]

# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
# }
# intersphinx_disabled_domains = ['std']

# templates_path = ['_templates']

# # -- Options for HTML output

# html_theme = 'sphinx_rtd_theme'

# # -- Options for EPUB output
# epub_show_urls = 'footnote'


# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import io
import re
import shutil
import sys
import sphinx_rtd_theme  # noqa
import warnings
from typing import ForwardRef


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), "..", "..", *names), encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    try:
        with io.open(os.path.join(os.path.dirname(__file__), *file_paths), encoding="utf8") as fp:
            version_file = fp.read()
        version_match = re.search(r"^__version__ = version = ['\"]([^'\"]*)['\"]", version_file, re.M)
        return version_match.group(1)
    except Exception as e:
        raise RuntimeError("Unable to find version string:\n", e)


sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# - Copy over examples folder to docs/source
# This makes it so that nbsphinx properly loads the notebook images

examples_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

# Include examples in documentation
# This adds a lot of time to the doc buiod; to bypass use the environment variable SKIP_EXAMPLES=true
for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)

            # If we're skipping examples, put a dummy file in place
            if os.getenv("SKIP_EXAMPLES"):
                if dest_filename.endswith("index.rst"):
                    shutil.copyfile(source_filename, dest_filename)
                else:
                    with open(os.path.splitext(dest_filename)[0] + ".rst", "w") as f:
                        basename = os.path.splitext(os.path.basename(dest_filename))[0]
                        f.write(f"{basename}\n" + "=" * 80)

            # Otherwise, copy over the real example files
            else:
                shutil.copyfile(source_filename, dest_filename)

# -- Project information -----------------------------------------------------

project = "push"
copyright = "Placeholder Copyright"
author = "Daniel Huang"

# The short X.Y version
version = "0.1"
# version = find_version("..", "..", "push", "version.py")
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    'sphinx.ext.napoleon',
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "m2r2",
]

# Configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "linear_operator": ("https://linear-operator.readthedocs.io/en/stable/", None),
}

# Disable docstring inheritance
autodoc_inherit_docstrings = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    "_build", "**.ipynb_checkpoints", "examples/**/README.rst",
    "examples/README.rst", "examples/index.rst"
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    # 'logo_only': False,
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pushdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, "push.tex", "push Documentation", "Daniel Huang", "manual")]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "push", "push Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "push",
        "push Documentation",
        author,
        "push",
        "One line description of project.",
        "Miscellaneous",
    )
]


# -- Function to format  typehints ----------------------------------------------
# Adapted from
# https://github.com/cornellius-gp/linear_operator/blob/2b33b9f83b45f0cb8cb3490fc5f254cc59393c25/docs/source/conf.py
def _process(annotation, config):
    """
    A function to convert a type/rtype typehint annotation into a :type:/:rtype: string.
    This function is a bit hacky, and specific to the type annotations we use most frequently.
    This function is recursive.
    """
    # Simple/base case: any string annotation is ready to go
    if type(annotation) == str:
        return annotation

    # Convert Ellipsis into "..."
    elif annotation == Ellipsis:
        return "..."

    # Convert any class (i.e. torch.Tensor, LinearOperator, gpytorch, etc.) into appropriate strings
    # For external classes, the format will be e.g. "torch.Tensor"
    # For any linear_operator class, the format will be e.g. "~linear_operator.operators.TriangularLinearOperator"
    # For any internal class, the format will be e.g. "~gpytorch.kernels.RBFKernel"
    elif hasattr(annotation, "__name__"):
        module = annotation.__module__ + "."
        if module.split(".")[0] == "linear_operator":
            if annotation.__name__.endswith("LinearOperator"):
                module = "~linear_operator."
            elif annotation.__name__.endswith("LinearOperator"):
                module = "~linear_operator.operators."
            else:
                module = "~" + module
        elif module.split(".")[0] == "push":
            module = "~" + module
        elif module == "builtins.":
            module = ""
        res = f"{module}{annotation.__name__}"

    # Convert any Union[*A*, *B*, *C*] into "*A* or *B* or *C*"
    # Also, convert any Optional[*A*] into "*A*, optional"
    elif str(annotation).startswith("typing.Union"):
        is_optional_str = ""
        args = list(annotation.__args__)
        # Hack: Optional[*A*] are represented internally as Union[*A*, Nonetype]
        # This catches this case
        if args[-1] is type(None):  # noqa E721
            del args[-1]
            is_optional_str = ", optional"
        processed_args = [_process(arg, config) for arg in args]
        res = " or ".join(processed_args) + is_optional_str

    # Convert any Tuple[*A*, *B*] into "(*A*, *B*)"
    elif str(annotation).startswith("typing.Tuple"):
        args = list(annotation.__args__)
        res = "(" + ", ".join(_process(arg, config) for arg in args) + ")"

    # Convert any List[*A*] into "list(*A*)"
    elif str(annotation).startswith("typing.List"):
        arg = annotation.__args__[0]
        res = "list(" + _process(arg, config) + ")"

    # Convert any List[*A*] into "list(*A*)"
    elif str(annotation).startswith("typing.Dict"):
        res = str(annotation)

    # Convert any Iterable[*A*] into "iterable(*A*)"
    elif str(annotation).startswith("typing.Iterable"):
        arg = annotation.__args__[0]
        res = "iterable(" + _process(arg, config) + ")"

    # Handle "Callable"
    elif str(annotation).startswith("typing.Callable"):
        res = "callable"

    # Handle "Any"
    elif str(annotation).startswith("typing.Any"):
        res = ""

    # Special cases for forward references.
    # This is brittle, as it only contains case for a select few forward refs
    # All others that aren't caught by this are handled by the default case
    elif isinstance(annotation, ForwardRef):
        res = str(annotation.__forward_arg__)

    # For everything we didn't catch: use the simplist string representation
    else:
        warnings.warn(f"No rule for {annotation}. Using default resolution...", RuntimeWarning)
        res = str(annotation)

    return res


# -- Options for typehints ----------------------------------------------
always_document_param_types = True
# typehints_use_rtype = False
typehints_defaults = None  # or "comma"
simplify_optional_unions = False
typehints_formatter = _process

