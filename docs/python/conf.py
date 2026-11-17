"""Sphinx configuration of the documentation of the Python package ``comnum``.

Build with ``just docs-py`` (or ``sphinx-build -b html docs/python docs/python/_build/html``)
after ``maturin develop --release``: the API pages are generated from the docstrings of
the compiled extension, and the gallery runs every script of ``examples/``.
"""

import re
import sys
from pathlib import Path

import comnum

# `reset_style` below is referenced by name by the gallery; the examples that read the
# specifications of the repository import `_gallery_utils`; the style and the palette
# are those of `comnum.plot`, the themes those of `tools/_common.py`
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1] / "tools"))

from _common import GALLERY_THEMES  # noqa: E402

project = "comnum"
author = "Vincent Choqueuse"
copyright = "2022-2026, Vincent Choqueuse"
release = comnum.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
]

# the docstrings of the extension write code and formulas between single backticks
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
templates_path = ["_templates"]
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_member_order = "groupwise"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
exclude_patterns = ["_build", "examples", "tutorials", "basics"]
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# --- figures: one style for the whole gallery ---------------------------------------


def reset_style(gallery_conf, fname):
    """The style of :func:`comnum.plot.style` from the matplotlib defaults, before every
    example."""
    import matplotlib as mpl

    mpl.rcdefaults()
    comnum.plot.style()


sphinx_gallery_conf = {
    # three galleries: the basics (a short course, one idea per chapter), the tutorials
    # (the library step by step) and the classical results, one
    # sub-gallery per theme (the reproductions of named figures in their theme)
    "examples_dirs": ["basics", "tutorials", "examples"],
    "gallery_dirs": ["auto_basics", "auto_tutorials", "auto_examples"],
    "filename_pattern": r"/plot_",
    "subsection_order": [f"examples/{t}" for t in GALLERY_THEMES],
    "within_subsection_order": "FileNameSortKey",
    "reset_modules": ("matplotlib", "conf.reset_style"),
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
    "image_scrapers": ("matplotlib",),
    # the highlights at the head of the gallery are a minigallery of file paths
    "backreferences_dir": "gen_modules/backreferences",
}

# --- theme --------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"comnum {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/logo.svg"
html_theme_options = {
    "logo": {"text": "comnum", "image_light": "_static/logo.svg", "image_dark": "_static/logo.svg"},
    "github_url": "https://github.com/com-rs/comnum",
    # the browser playground, published next to this site by the job docs-python
    "external_links": [
        {"name": "Playground", "url": "https://com-rs.github.io/comnum/playground/"}
    ],
    "header_links_before_dropdown": 8,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_sidebars = {"index": [], "getting_started": []}
html_context = {"default_mode": "light"}

# --- references ---------------------------------------------------------------------

# The docstrings cite their sources as NumPy does: ``[1]_`` in the text and a
# ``References:`` section of ``.. [1] G. P. Agrawal, ...`` entries. Two adjustments when
# they are rendered, help() keeping the plain text:
# - a module page (``comnum.sim``) shows several docstrings, whose labels "1" would
#   collide: the labels become auto-numbered footnotes named after the object;
# - a one-line entry that opens with an initial ("G. P. Agrawal") would be read by
#   reStructuredText as an enumerated list ("G.", "P."), which drops the initials: the
#   period of a leading initial is escaped.
_CITATION = re.compile(r"\[(\d+)\]_")
_REFERENCE = re.compile(r"^(\s*)\.\. \[(\d+)\] (?:([A-Z])\. )?")


def _render_references(app, what, name, obj, options, lines):
    key = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")

    def entry(m):
        initial = f"{m.group(3)}\\. " if m.group(3) else ""
        return f"{m.group(1)}.. [#{key}-{m.group(2)}] {initial}"

    for i, line in enumerate(lines):
        line = _REFERENCE.sub(entry, line)
        lines[i] = _CITATION.sub(lambda m: f"[#{key}-{m.group(1)}]_", line)


def setup(app):
    app.connect("autodoc-process-docstring", _render_references)
