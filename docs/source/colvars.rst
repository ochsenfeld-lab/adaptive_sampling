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

Internally, in the `EnhancedSampling` class the `get_cv` function of the `adaptive_sampling.colvars.CV` class is called in every MD step.
Therefore, all CVs that are defined in the `CV.get_cv()` function are available for sampling. 

Each CV is gets associated with a `type` in the `get_cv` function, that defines how units of CVs are handled. 
The CV `type` can be one of the following:
    * "`distance`": For distance based CVs, input and output units are in Angstrom. Internally, atomic units (Bohr) are used. 
    * "`angle`": For angle based CVs, input and output units are in degrees. Internally, radians are used. 
    * "`None`": For CVs in arbitrary units, input and output units are not converted.

In the following section, we provide examples of setting up different CVs.

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

* **Minimum distance** out of a list of distances `d`:

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

    # Linear combination of CVs, three versions are available:
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

In PCVs, the CV is given by the progress $s$ along a high-dimensional path and the distance $z$ to the path. 

The `adaptive_sampling.colvars.PathCV` class implements two different types of PCVs:

1. The **arithmetic PCV**, as suggested by `Branduardi et al. <https://doi.org/10.1063/1.2432340>`_:

.. math::

    s = \frac{1}{P-1}\frac{\sum_{i=0}^{P} (i-1) e^{-\lambda |\mathbf{x}-\mathbf{x}_i|}}{\sum_{i=0}^{P} e^{-\lambda |\mathbf{x}-\mathbf{x}_i|}}

.. math::

    z = -\frac{1}{\lambda} \ln\left(\sum_{i=0}^{P} e^{-\lambda |\mathbf{x}-\mathbf{x}_i|}\right) 

with :math:`P` being the number of nodes along the path, :math:`\mathbf{x}_i` the Cartesian coordinates of the :math:`i`-th path node, and :math:`\lambda` a parameter that ensures that the path is smooth and differentiable.
Any distance metric can be used to calculate the distance :math:`|\mathbf{x}-\mathbf{x}_i|` between the current coordinates :math:`\mathbf{x}` and the path nodes :math:`\mathbf{x}_i`.

2. The **geometric PCV**, as suggested by `Leines et al. <https://doi.org/10.1103/PhysRevLett.109.020601>`_: 

.. math::

    s(\mathbf{z}) = \frac{m}{M} \pm \frac{1}{2M} \bigg(\frac{\sqrt{\mathbf{v}_1 \cdot \mathbf{v}_3)^2 - |\mathbf{v}_3|^2 (|\mathbf{v}_1|^2-|\mathbf{v}_2|^2)} - (\mathbf{v}_1\cdot\mathbf{v}_3)}{|\mathbf{v}_3|^2} - 1 \bigg) \ ,

.. math::

    z(\mathbf{z}) = \bigg| \mathbf{v}_1 + \frac{1}{2} \bigg(\frac{ \sqrt{\mathbf{v}_1 \cdot \mathbf{v}_3)^2 - |\mathbf{v}_3|^2 (|\mathbf{v}_1|^2-|\mathbf{v}_2|^2)} - (\mathbf{v}_1\cdot\mathbf{v}_3)}{|\mathbf{v}_3|^2} - 1 \bigg) \mathbf{v}_4 \bigg| \ ,

with index of the first, second and third-closest nodes :math:`m`, :math:`n`, :math:`k`, respectively, and
vectors :math:`\textbf{v}_1=\textbf{z}_{m} - \textbf{z}`, :math:`\textbf{v}_2=\textbf{z} - \textbf{z}_{m-1}`, 
:math:`\textbf{v}_3=\textbf{z}_{m+1} - \textbf{z}_{m}` and :math:`\textbf{v}_4=\mathbf{z}_{} - \mathbf{z}_{m-1}`.
The :math:`\pm` is negative if :math:`\mathbf{z}` is left of the closest path node, and positive otherwise. 
Other than the arithmetic PCV, the geometric PCV is always applied in the space :math:`\mathbf{z}` of selected CVs. 

PCVs can be defined as outlined below:

.. code-block:: python
    :linenos:

    cv_space = [
        ['distance', [0, 1]],  # distance between atoms 0 and 1
        ['distance', [2, 3]],  # distance between atoms 2 and 3
        #... more CVs can be added
    ]

    # Define parameters of the `adaptive_sampling.colvars.PathCV` class as a dictionary
    cv_def = {
        "guess_path": "path.xyz",           # Path file containing the path nodes, can be a `.xzy` or `.npy` file.
        "coordinate_system": "cv_space",    # Coordinate system used by the PCV, e.g. "cv_space" or "cartesian".
        "active": cv_space,                 # If `coordinate_system="cv_space"`, the CV space used by the PCV. If `coordinate_system="cartesian"`, indices of atoms that are included in the path CV
        "n_interpolate": 0,                 # Number of nodes that are added between original nodes by linear interpolation, if negative, slice path nodes according to `self.path[::abs(n_interpolate)]`
        "smooth_damping": 0.1,              # Controls smoothing of path (0: no smoothing, 1: linear interpolation between neighbours).
        "reparam_steps": 100,               # Maximum number of steps for reparametrization of the path to ensure equidistant spacing of path nodes.
        "reparam_tol": 1e-5,                # Tolerance for reparametrization of the path to ensure equidistant spacing of path nodes.
        "metric": "rmsd",                   # Distance metric used to calculate distance between current coordinates and path nodes.
        "adaptive": False,                  # If True, the path is adapted during the simulation to converge to the minimum free energy path. Only for the geometric PCV.
        "requires_z": True,                 # If True, the distance to the path is calculated along with the progress along the path. If False, only the progress along the path is calculated.
    }

    # Definition of the PCV as required for methods derived from the `EnhancedSampling` base class.
    #                           MIN  MAX  BIN_WIDTH
    the_cv = [["gpath", cv_def, 0.0, 1.0, 0.01]]       # geometric PCV
    the_cv = [["path", cv_def, 0.0, 1..0, 0.01]]       # arithmetic PCV
   
    # Definition of using the both the `s` and `z` values in the simulation. NOT YET TESTED!
    the_cv = [
        ["gpath", cv_def, 0.0, 1.0, 0.01], 
        ["path_z", cv_def, 0.0, 5.0, 0.1], 
    ]

In the CV space of PCVs, currently the following CVs are supported:
 * **Distance**: Distance between two atoms or groups of atoms.
 * **Angle**: Angle between three atoms or groups of atoms.
 * **Torsion**: Torsion angle between four atoms or groups of atoms.
 * **Conntact**: A switching function between two groups of atoms. (see `Switching function` in the "Common, simple CVs" section)
 * **Min_distance**: Minimum distance out of a list of distances.
 * **Coordination_number**: Coordination numbers of atoms.
 * **CEC**: (Modified) Center-of-Excess Charge (mCEC) coordinate.

Machine Learning CVs (MLCVs):
-----------------------------

MLCVs are defined using the `mlcolvars` package, which is based on PyTorch and provides a framework for training a number of popular MLCVs.
For more information on types of implemented MLCVs, as well on how to train them, please refer to the `mlcolvars documentation <https://mlcolvar.readthedocs.io/en/stable/>`_.

To use MLCVs together with the `adaptive_sampling` package, pretrained models need to be saved in torchscript format:

.. code-block:: python
    :linenos:

    model.to_torchscript('model.ptc', method="trace")

The `adaptive_sampling.colvars.MLCOLVAR` class can then be used to run simulations with the pretrained CV model:

.. code-block:: python
    :linenos:

    # definition of the CV space the MLCV uses 
    cv_space = [
        ['distance', [1,2]],  # distance between atoms 1 and 2
        ['distance', [3,4]],  # distance between atoms 3 and 4
        #...
    ]

    # Define parameters of the `adaptive_sampling.colvars.MLCOLVAR` class as a dictionary
    cv_def = {
        "model": "model.ptc",                              # path to the pretrained model in torchscript format.
        "coordinate_system": "cv_space",                   # coordinate system used by the MLCV, e.g. "cv_space" or "cartesian".
        "cv_def": cv_space,                                # definition of the CV space the MLCV uses.
        "cv_idx": 0,                                       # For MLCVs with multiple output nodes, the index of the node to use as CV, e.g., 0 for the first node.
        "unit_conversion_factor": units.BOHR_to_ANGSTROM,  # conversion factor for the CV space, e.g. BOHR_to_ANGSTROM if the MLCV was trained in Angstrom.
        "device": "cpu",                                   # device to run the model on, e.g. "cpu" or "cuda:0".
        "type": None,                                      # type of CV, None means no unit conversion.
    }

    # Definition of the MLCV as required for methods derived from the `EnhancedSampling` base class.
    #                               MIN  MAX  BIN_WIDTH
    the_cv = [["mlcolvar", cv_def, -1.0, 1.0, 0.05]] 

In the CV space of mlcolvars, currently the following CVs are supported:
 * **Distance**: Distance between two atoms or groups of atoms.
 * **Angle**: Angle between three atoms or groups of atoms.
 * **Torsion**: Torsion angle between four atoms or groups of atoms.
 * **Conntact**: A switching function between two groups of atoms. (see `Switching function` in the "Common, simple CVs" section)
 * **Min_distance**: Minimum distance out of a list of distances.
 * **Coordination_number**: Coordination numbers of atoms.
 * **CEC**: (Modified) Center-of-Excess Charge (mCEC) coordinate.

The Modified Center-of-Excess Charge (mCEC):
--------------------------------------------

Graph-based CVs (Graph-CVs):
----------------------------







