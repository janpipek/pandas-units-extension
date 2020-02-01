#!/usr/bin/env python
import os
from setuptools import setup, find_packages

THIS_DIR = os.path.dirname(__file__)


def read_description():
    readme_file = os.path.join(THIS_DIR, "README.md")
    with open(readme_file, "r") as f:
        return f.read()


def read_info():
    """Single source of version number and other info.

    Inspiration:
    - https://packaging.python.org/guides/single-sourcing-package-version/
    - https://github.com/psf/requests/blob/master/setup.py
    """
    scope = {}
    version_file = os.path.join(THIS_DIR, "pandas_units_extension", "version.py")
    with open(version_file, "r") as f:
        exec(f.read(), scope)  # pylint: disable=exec-used
    return scope


INFO = read_info()

SETUP_OPTIONS = dict(
    name="pandas-units-extension",
    version=INFO["__version__"],
    packages=find_packages(),
    license="MIT",
    description="Extension pandas dtype and array for physical units.",
    author=INFO["__author__"],
    author_email=INFO["__author_email__"],
    long_description=read_description(),
    url=INFO["__url__"],
    install_requires=["pandas>=0.25.3, <1.0", "astropy~=4.0"],
    python_requires=">=3.6",
)

if __name__ == "__main__":
    setup(**SETUP_OPTIONS)
