from __future__ import annotations

import os
import sys

import pytest
from _pytest.nodes import Item
# from numpy import isclose
from logging import getLogger

log = getLogger()

cwd = os.getcwd()
print(cwd)
sys.path.append(cwd + "/pydar-utils/")


# def assert_close(*args):
#     if not isclose(*args):
#         log.error(f'Assertion failed: isclose({args=}')
#     assert isclose

