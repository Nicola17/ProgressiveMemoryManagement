from setuptools import setup
from distutils.sysconfig import get_python_lib
import glob
import os
import sys

if os.path.exists('readme.rst'):
    print("""The setup.py script should be executed from the build directory.
            Please see the file 'readme.rst' for further instructions.""")
    sys.exit(1)


setup(
    name = 'pypmm',
    package_dir = {'': 'src/python'},
    data_files = [(get_python_lib(), glob.glob('src/python/*.so')),],
    author = 'Nicola Pezzotti',
    author_email = 'nicola.pezzotti@gmail.com',
    description = 'Progressive Memory Management.',
    test_require = ['nose'],
    url = 'http://',
    zip_safe = False
    )
