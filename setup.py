# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:48:19 2019

@author: ecramer
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peac-cireric",
    version="0.0.1",
    author="Eric Cramer",
    author_email="ericscrum@gmail.com",
    description="A package for running PEAC analysis of reduced dimensionality spaces.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emcramer/peac",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)