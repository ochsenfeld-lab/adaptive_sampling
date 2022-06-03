import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit("Sorry, Python < 3.8 is not supported")

setup(
    name="adaptive_sampling",
    packages=["adaptive_sampling"],
    version="0.1",
    license="MIT",
    description="Sampling algorithms for molecular transitions",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Andreas Hulm",
    author_email="andreas.hulm@cup.uni-muenchen.de", 
    url="https://github.com/ahulm/adaptive_sampling",
    download_url="https://github.com/ahulm/adaptive_sampling/XXXX",
    keywords=["sampling", "molecular dynamics", "free energy", "chemical reactions"],
    install_requires=[
        "torch>=1.10.2", 
        "numpy>=1.19.5", 
        "scipy>=1.8.0",
        ],
    setup_requires=[
        "pytest", 
        "flake8",
        ],
    tests_requires=["pytest"],
    classifiers=[
        'Development Status :: 4 - Beta',  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Environment :: Console',
        'Intended Audience :: End Users',      
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3.8',
        ],
    zip_safe=False,
)
