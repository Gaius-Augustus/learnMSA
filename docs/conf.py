import os
import sys

# Add project root and hidten/ to sys.path so autodoc can import your code
sys.path.insert(0, os.path.abspath(".."))

project = 'learnMSA'
copyright = '2025, Felix Becker'
author = 'Felix Becker'
release = '2.0.13'

# Base extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Only include viewcode for local/private builds
# Set SPHINX_PUBLIC_BUILD=1 in CI to disable source code links
if not os.environ.get('SPHINX_PUBLIC_BUILD'):
    extensions.append("sphinx.ext.viewcode")

# Napoleon settings
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = ['algorithms/**']

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

html_theme = "sphinx_rtd_theme"
