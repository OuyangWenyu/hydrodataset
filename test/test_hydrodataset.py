"""
Author: Wenyu Ouyang
Date: 2022-09-05 23:20:24
LastEditTime: 2022-09-06 11:23:30
LastEditors: Wenyu Ouyang
Description: test funcs
FilePath: \hydrodataset\test\test_hydrodataset.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
#!/usr/bin/env python

"""Tests for `hydrodataset` package."""

import pytest

from hydrodataset import hydrodataset


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
