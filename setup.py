#!/usr/bin/env python
"""
Copyright 2018 James D. Triveri
"""
import pathlib
from setuptools import setup, find_packages

NAME = "trikit"
DESCRIPTION = "A Pythonic Approach to Actuarial Reserving"
AUTHOR = "James D. Triveri"
AUTHOR_EMAIL = "james.triveri@gmail.com"
URL = "https://github.com/trikit/trikit"
LICENSE = "MIT"
BASE_DIR = pathlib.Path(__file__).parent.resolve()
LONG_DESCRIPTION = (BASE_DIR / "README.md").read_text(encoding="utf-8")
VERSION = (BASE_DIR / "VERSION").read_text(encoding="utf-8")
REQUIREMENTS = (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").split()



setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="MIT",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
        ],
    keywords=[
        "actuarial finance reserving chainladder insurance",
        ],
    install_requires=REQUIREMENTS,
    include_package_data=True,
    )
