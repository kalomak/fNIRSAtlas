from pathlib import Path

import fnirsatlas
from fnirsatlas import datasets

project = "fnirsatlas"
author = "fnirsatlas contributors"
version = fnirsatlas.__version__
release = version

extensions = [
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
autodoc_mock_imports = ["torch"]
templates_path = ["_templates"]
exclude_patterns = ["build", "skills", "Thumbs.db", ".DS_Store"]
html_theme = "pydata_sphinx_theme"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "plot_gallery": False,
}


def dataset_names():
    """Exported dataset classes with ``METADATA``."""
    return [
        name
        for name in datasets.__all__
        if isinstance(cls := getattr(datasets, name), type)
        and issubclass(cls, datasets.BaseFNIRSDataset)
        and not name.startswith("Base")
        and hasattr(cls, "METADATA")
    ]


def write_datasets_page(path):
    """Write the datasets page, one autosummary entry per dataset."""
    entries = "\n".join(f"   {name}" for name in dataset_names())
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"""Datasets
========

.. currentmodule:: fnirsatlas.datasets

.. autosummary::
   :toctree: .
   :template: class.rst

{entries}
""")


write_datasets_page(Path(__file__).parent / "generated" / "datasets.rst")
