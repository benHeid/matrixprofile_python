#!/usr/bin/env python

from setuptools import setup, find_packages

import sys

from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


setup(name='Matrixprofile',
      version='0.01',
      description='Python implementation of matrix profiles',
      author='benheid',
      author_email='',
      tests_require=["pytest"],
      cmdclass={"test": PyTest},     
      url='',
      packages=find_packages(),
     )