import glob
from pybind11.setup_helpers import Pybind11Extension

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

'''
Use the following to build extension:
python3 setup.py install
'''



setup(
    name="csrreader",
    version=__version__,
    author="Samraj Moorjani",
    url="https://github.com/smoorjani/CSRReader",
    description="Reads CSR matrices from CSV/TSV/MMIO formats.",
    ext_modules= [
        Pybind11Extension(
            "csrreader",
            sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
)