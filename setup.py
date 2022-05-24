import sys
from setuptools import setup, find_packages

if sys.version_info < (3,8):
    sys.exit('Sorry, Python < 3.8 is not supported')

setup(
    name="adaptive_sampling",
    version="0.1.0",
    description="sampling algorithms",
    author="Andreas Hulm",
    packages=find_packages(include=["adaptive_sampling"]),
    install_requires=[
        'torch>=1.10.2'
        'numpy>=1.19.5'
    ],
    setup_requires=['pytest', 'flake8'],
    tests_requires=['pytest'],
    zip_safe=False,
)
