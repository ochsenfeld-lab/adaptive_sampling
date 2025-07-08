Collective Variables (CVs)
==========================

CVs are used to define the reaction coordinates of interest in adaptive sampling simulations. 
The `adaptive_sampling.colvars` module provides a variety of CV definitions.

The CVs are initialized internally in the `EnhancedSampling` class, which is the base class for most sampling methods in `adaptive_sampling.sampling_tools`.
The basic syntax for defining CVs is given below, where the distance between two atoms is used as a CV and a grid of 20 bins is created between 1.0 and 3.0 with a bin width of 0.1:

.. code-block:: python

    cv_type = 'distance'  # Type of CV, e.g. 'distance', 'angle', 'torsion', etc
    cv_def = [0, 1]       # Definition of CV, e.g., for 'distance' indices of atoms 
    min_cv = 1.0          # Minimum value of CV in the range of interest 
    max_cv = 3.0          # Maximum value of CV in the range of interest
    bin_width = 0.1       # Width of the histogram bin for the CV in the range of interest
    the_cv = [
        [cv_type, cv_def, min_cv, max_cv, bin_width], 
        # An additional second CV can be added to the list
    ]

Internally, in the `EnhancedSampling` class the `get_cv` function of the `CV` class in the `adaptive_sampling.colvars` module is called in every MD step.
Therefore, all CVs that are defined in the `CV.get_cv` function are available for sampling. 

Each CV is gets associated with a `cv_type` in the `get_cv` function, that defines how units of CVs are handled. 
The `cv_type` can be one of the following:
    * "`distance`": For distance based CVs, input and output units are in Angstrom.
    * "`angle`": For angle based CVs, input and output units are in degrees. 
    * "`None`": For CVs in arbitrary units, input and output units are not converted.

In the following examples of how popular CVs can be defined are given:

Common, simple CVs
------------------

The following examples show how to define common CVs for sampling algorithms that derive from the `EnhancedSampling` class.

* **Distance** between two atoms or groups of atoms:

.. code-block:: python
    :linenos:

    # Distance between two atoms
    cv_type = 'distance'

    # Indices of atoms for which distance is calculated
    cv_def = [0, 1]     
    
    # or use center of mass of a groups of atoms
    cv_def = [[0, 1, 2], 3] 

* **Angle** between three atoms or groups of atoms:

.. code-block:: python
    :linenos:

    # Angle between three atoms
    cv_type = 'angle'

    # Indices of atoms or list of indices to calculate angle between centers of mass
    cv_def = [0, 1, 2]     

* **Torsion** angle between four atoms or groups of atoms:

.. code-block:: python
    :linenos:

    # Torsion angle between four atoms
    cv_type = 'torsion'

    # Indices of atoms or list of indices to calculate dihedral angle between centers of mass
    cv_def = [0, 1, 2, 3]  

* **Switching function** between two groups of atoms:

.. math::

    f(r) = \frac{1-(\frac{r}{r_0})^n}{1-(\frac{r}{r_0})^m}

.. code-block:: python
    :linenos:

    # Minimum distance between two groups of atoms
    cv_type = 'switching_function'
    
    r_0 = 3.0  # Switching distance in Angstrom (default: 3.0)
    n = 6      # Exponent nominator (default: 6)
    m = 12     # Exponent denominator (default: 12)

    # List containing distance definition and additional parameters of switching function 
    cv_def = [0, 1, r_0, n, m]

* **Minimum distance** out of a list of distances:

.. math::

    d_\mathrm{min} = \mathrm{Min}\left[d_0, d_1, \ldots, d_n \right]

.. code-block:: python
    :linenos:

    # Minimum distance between two groups of atoms
    cv_type = 'distance_min'

    # List of distance definitions, minimum distance out of the list used as CV.
    cv_def = [[0, 1], [2, 3]]  

* **Linear combinations** of the above CVs:

.. math::

    f_\mathrm{LC} = \sum_{i=0}^{n} a_i\, f_i

.. code-block:: python
    :linenos:

    # Linear combination of two CVs, three versions are available:
    cv_type = 'linear_combination'   # type of CV is None, so that units are not converted
    cv_type = 'lin_comb_dists'       # type of CV is 'distance', so that input and output units are Angstrom
    cv_type = 'lin_comb_angles'      # type of CV is 'angle', so that input and output units are degrees

    # Definition of a linear combination containing the CV type, prefactor and individual CV definitions 
    cv_def = [
        ['distance', 1.0, [0, 1]],
        ['distance',-1.0, [2, 3]],
        #... more CVs can be added
    ]

Path CVs (PCVs)
---------------



Machine Learning CVs (MLCVs):
-----------------------------

The Modified Center-of-Excess Charge (mCEC):
--------------------------------------------

Graph-based CVs (Graph-CVs):
----------------------------







