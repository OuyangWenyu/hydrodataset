#!/usr/bin/env python

"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:22:54
LastEditTime: 2023-07-31 17:21:05
LastEditors: Wenyu Ouyang
Description: The setup script
FilePath: /hydrodataset/setup.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import io
import pathlib
from os import path as op
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()
here = op.abspath(op.dirname(__file__))


# get the dependencies and installs
with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "git+" not in x]

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Wenyu Ouyang",
    author_email="wenyuouyang@outlook.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="A Python package for downloading and reading hydrological datasets",
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="hydrodataset",
    name="hydrodataset",
    packages=find_packages(include=["hydrodataset", "hydrodataset.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OuyangWenyu/hydrodataset",
    version='0.1.12',
    zip_safe=False,
)
