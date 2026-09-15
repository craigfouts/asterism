# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.append(os.path.abspath('../../'))

def remove_docstring(app, obj_type, name, obj, options, lines):
    if obj_type == 'module':
        del lines[:]

def setup(app):
    app.connect('autodoc-process-docstring', remove_docstring)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Asterism'
copyright = '2026, Craig Fouts'
author = 'Craig Fouts'
# release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_design', 'sphinx.ext.autodoc']
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    # 'external_links': [
    #     {'name': 'GitHub', 'url': 'https://github.com/craigfouts/asterism/tree/main'}
    # ],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/craigfouts/asterism/tree/main',
            'icon': 'fa-brands fa-github',
            'type': 'fontawesome',
        }
    ],
    'secondary_sidebar_items': ['page-toc'],
}
html_sidebars = {
    'install': [],
    'examples': [],
}
html_static_path = ['_static']
html_title = 'Asterism'
html_favicon = '../assets/images/icon.png'
html_show_sphinx = False
