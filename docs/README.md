Documentation:
==============

To create the documentation install the sphinx package and the read-the-docs theme:
```
$ pip install sphinx sphinx-rtd-theme
```
To update the code documentation from docstrings, run:
```
$ sphinx-apidoc -o ./source/code ../adaptive_sampling
```
Note, that new modules have to be added to `./source/code.rst` to appear in the documentation.

Other section can be added to the documentation by adding `.rst` files to the `./source` sirectory. To appear in the documentation the new `.rst` files have to be added to `./source/index.rst`. 