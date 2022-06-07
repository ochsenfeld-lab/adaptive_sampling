import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit("Sorry, Python < 3.8 is not supported")

setup(
    name="adaptive_sampling",
    packages=["adaptive_sampling"],
    version="1.0.0",
    license="MIT",
    description="Sampling algorithms for molecular transitions",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Andreas Hulm",
    author_email="andreas.hulm@cup.uni-muenchen.de", 
    url="https://github.com/ochsenfeld-lab/adaptive_sampling",
    download_url="https://github.com/ochsenfeld-lab/adaptive_sampling/archive/refs/tags/v1.0.0.zip",
    keywords=["computational chemistry", "molecular dynamics", "free energy", "chemical reactions"],
    install_requires=[
        "torch>=1.10.2", 
        "numpy>=1.19.5", 
        "scipy>=1.7.0",
        ],
    setup_requires=["pytest"],
    tests_requires=["pytest"],
    classifiers=[
        'Development Status :: 5 - Production/Stable',  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3.8',
        ],
    zip_safe=False,
)
