"""Legacy shim for tools that still expect a setup.py.

All project metadata lives in pyproject.toml and setup.cfg.
"""

from setuptools import setup

setup()
