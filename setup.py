import glob
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "0.0.1"

'''
Use the following to build extension:
python3 setup.py install
'''

ext_modules = [
    Pybind11Extension(
        "csrreader",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
    ),
]

setup(
    name="csrreader",
    version=__version__,
    author="Samraj Moorjani",
    author_email="samrajm2@illinois.edu",
    url="https://github.com/smoorjani/CSRReader",
    description="Reads CSR matrices from CSV/TSV/MMIO formats.",
    long_description="",
    ext_modules=ext_modules,
)