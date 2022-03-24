import os
from importlib.machinery import SourceFileLoader

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

version = (
    SourceFileLoader("bar_optimization.version", os.path.join(here, "bar_optimization", "version.py"))
    .load_module()
    .VERSION
)

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.read().splitlines()

setup(
    name="bar_optimization",
    version=version,
    long_description=long_description,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
)
