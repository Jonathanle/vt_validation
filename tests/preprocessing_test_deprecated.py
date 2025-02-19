
import pytest
import torch
import numpy as np
import os

import sys
sys.path.append('..')

from dataset import Preprocessor  # Update with correct import

"""
when running a test relies on conftest to make the functions

how tests use defined objects
project/
├── conftest.py          # (3) Third place to look
├── test_stuff.py
│
└── analytics/
    ├── conftest.py     # (2) Second place to look priorities when generaitng fixtures
    └── test_data.py    # (1) First place to look 

idea 2- no need to import modells because models from other classes are returned and welll defined.
"""


# test when one test has a dependency on the other test this is an example of "polluting"
def test_list(numpy_array):

    numpy_array.append('b')
    assert len(numpy_array) == 4 


def test_list2(numpy_array):
    numpy_array.append('c')


    assert len(numpy_array) == 5 