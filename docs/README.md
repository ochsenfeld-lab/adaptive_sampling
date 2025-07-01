Build the Documentation:
=========================

To create the documentation install the sphinx package, the read-the-docs theme and sphinx extensions:
```
$ pip install sphinx sphinx-rtd-theme sphinx-prompt sphinx-copybutton
```
To update the code documentation from docstrings, run:
```
$ sphinx-apidoc -o ./source/code ../adaptive_sampling
```
Note, that new modules have to be added to `./source/code.rst` to appear in the documentation.

New sections can be added to the documentation by adding `.rst` files to the `./source` directory and listing them in the `./source/index.rst`. 

To build the documentation, run:
```
$ make clean
$ make html
```
The documentation can be viewed by opening the `index.html` file:
```
firefox build/html/index.html
```

Parse Jupyter Notebooks with `nbsphinx`
=======================================

Jupyter notebooks can be parsed as documentation pages using the nbsphinx package.
```
$ conda install -c conda-forge nbsphinx
```
Add tutorial notebooks to the `./source/tutorials` folder and list them in the `tutorials.rst` to let them appear in the documentation.

Note, that whenever the documentation is built, the notebooks are executed, so they should not be computationally intensive!