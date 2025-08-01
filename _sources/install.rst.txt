Installation
============

To install the package, download the latest source code from the `GitHub repository <https://github.com/ochsenfeld-lab/adaptive_sampling>`_.

.. code-block:: console
    :linenos:

    git clone https://github.com/ochsenfeld-lab/adaptive_sampling.git

Then, navigate to the package directory and run install the package using pip.

.. code-block:: console
    :linenos:

    cd adaptive_sampling
    pip install .

Alternatively, you can install the package directly from GitHub using pip:

.. code-block:: console
    :linenos:

    pip install git+https://github.com/ochsenfeld-lab/adaptive_sampling.git

Dependencies
------------ 

The package requires the following dependencies:
 * `numpy <https://numpy.org/>`_
 * `scipy <https://www.scipy.org/>`_
 * `torch <https://pytorch.org/>`_
 * `ase <https://wiki.fysik.dtu.dk/ase/>`_ (for QM simulations)
 * `openmm <https://openmm.org/>`_ (for MM simulations)
 * `ash <https://ash.readthedocs.io/en/latest/>`_ (for QM/MM simulations)