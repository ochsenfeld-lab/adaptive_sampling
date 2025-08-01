Sampling tools
==============

The `sampling_tools` subpackage provides a set of tools for importance sampling of molecular transitions.

The `EnhancedSampling` Base Class
---------------------------------

The `EnhancedSampling` class is the base class for all adaptive enhanced sampling methods in the `sampling_tools` subpackage.

It implements basic functionalities for running adaptive sampling simulations. Parameters of the `EnhancedSampling`, which are relevant for all derived subclasses are:

.. code-block:: python
    :linenos:

    EnhancedSampling(
        the_md,                     # The MD interface from `adaptive_sampling.interface`
        the_cv,                     # The definition of collective variables (CV) from `adaptive_sampling.colvars`
        equil_temp=300.0,           # Equilibrium temperature of the MD
        kinetics=True,              # Calculate and output necessary metrics to obtain accurate kinetics
        f_conf=100.0,               # Force constant for confinement of CVs to the range of interest with harmonic walls in kJ/mol/(CV units)^2
        output_freq=100,            # Frequency of writing outputs in MD steps
        multiple_walker=False,      # Use shared bias multiple walker implementation to synchronize time dependent biasing potentials with other simulations via buffer file
        periodicity=None,           # Periodic boundary conditions for periodic CVs, list of boundaries of `shape(len(CVs),2)`, [[lower_boundary0, upper_boundary0], ...]
        verbose=True,               # Print verbose information
    )   


(Well-tempered) metadynamics (WTM)
----------------------------------
In WTM the bias potential is constructed from a superposition of Gaussian hills of the form 

.. math::

    G(\textbf{z},\textbf{z}_t) = h_\mathrm{G}\: e^{-(\mathbf{z}-\mathbf{z}_t)^2 / 2\sigma_\mathrm{G}^2} \,,

where :math:`\mathbf{z}` is the current value of the collective variable (CV), :math:`\mathbf{z}_t` is the CV vector at time :math:`t`, :math:`h_\mathrm{G}` is the height of the Gaussian, and :math:`\sigma_\mathrm{G}` is the width of the Gaussian.
The bias potential is then given by

.. math:: 
    
    U^\mathrm{WTM}(\mathbf{z}, t) = \sum_{t=0,\tau_\mathrm{G}, 2\tau_\mathrm{G},...} e^{- \beta U^\mathrm{WTM}(\mathbf{z},t-1)/(\gamma-1)}\: G(\textbf{z},\textbf{z}_t) \,,

where Gaussian hills are added at regular time intervals :math:`t = 0, \tau_\mathrm{G}, 2\tau_\mathrm{G}, ...` to the bias potential :math:`U^\mathrm{WTM}(\mathbf{z},t)`.
If the well-tempered version of metadynamics is used, the height of new Gaussian hills is scaled with the factor :math:`e^{- \beta U^\mathrm{WTM}(\mathbf{z},t-1)/(\gamma-1)}`, such that Gaussian hills shrink over time ensuring smooth convergence. 
For :math:`\gamma = \infty` the scaling factor is 1 and standard metadynamics is recovered, where the height of Gaussian hills is constant over time.

The WTM algorithm can be used as follows:

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import WTM

    the_md = ... # initialize the molecular dynamics interface using the `adaptive_sampling.interface` module
    the_cv = ... # define the collective variable (CV) using the `adaptive_sampling.colvars` module

    the_bias = WTM(
        the_md,
        the_cv,
        hill_height=1.0,            # Height of the Gaussian hills in kJ/mol
        hill_std=0.2,               # Standard deviation of the Gaussian hills in units of the CV (e.g. Angstrom for distance CVs, degree for angle CVs), can also be a list of floats for 2D CVs
        hill_drop_freq=100,         # Frequency of adding new Gaussian hills in MD steps (e.g. every 100 steps)
        bias_factor=20,             # The bias factor gamma, which controls the shrinking of Gaussian hills over time, Default: None
        well_tempered_temp=None,    # The bias temperature DeltaT is an alternative to setting gamma. Note, that setting DeltaT always overwrites gamma! gamma=DeltaT/(T+1) with temperature of the MD simulation T. Default: np.inf (standard metadynamics)
        force_from_grid=True,       # Always recommended. If True, bias potentials and forces are accumulated on a grid, if False, the sum of Gaussian hills is calculated in every step, which can be expensive for long runs.
        #...,                       # Additional inherited keyword arguments from the `EnhancedSampling` class.
    )

For more information on MtD and WTM, refer to the original publications:
    * MtD: https://doi.org/10.1103/PhysRevLett.96.090601
    * WTM: https://doi.org/10.1103/PhysRevLett.100.020603

On-the-fly probability enhanced sampling (OPES)
-----------------------------------------------

OPES uses a similar superposition of Gaussian hills than WTM, but instead of the PMF the probability density is estimated from the Gaussian hills.
The kernel density estimation of the probability density is given by

.. math::
    
    \widetilde{\rho}(\mathbf{z}, t) = \frac{\sum_{t=\tau_\mathrm{G}, 2\tau_\mathrm{G},...} w_t\,G(\textbf{z},\textbf{z}_t)}{\sum_{t=\tau_\mathrm{G}, 2\tau_\mathrm{G},...}w_t} \,,

where :math:`w_t = e^{\beta U^\mathrm{OPES}(\mathbf{z}_t, t-1)}` is the weight of the new Gaussian hill at time :math:`t`.
Note that unlike before the height of Gaussian hills :math:`h_\mathrm{G}` is no free parameter as changing it only corresponds to changing the normalization of the probability density.
From the estimated probability density the bias potential is calculated as

.. math::
    
    U^\mathrm{OPES}(\mathbf{z},t)= \left(1-\frac{1}{\gamma}\right) \beta^{-1} \log \left(\frac{\widetilde{\rho}(\mathbf{z}, t)}{Z_t} + \epsilon\right) \,,

where :math:`Z_t` is a normalization factor and :math:`\epsilon` is a small constant to avoid numerical instabilities.
By choosing :math:`\epsilon=e^{-\beta \Delta E / (1-1/\gamma)}`, OPES allows for setting an upper bound to the biasing potential, which is given by the parameter :math:`\Delta E`.
As for WTM, :math:`\gamma` corresponds to the bias factor, which controls how much the original probability distribution is smoothed out by the bias potential.
A full example of setting up OPES simulations is given below:

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import OPES

    the_md = ... # initialize the molecular dynamics interface using the `adaptive_sampling.interface` module
    the_cv = ... # define the collective variable (CV) using the `adaptive_sampling.colvars` module

    the_bias = OPES(
        the_md,
        the_cv,
        kernel_std=0.1,             # Initial standard deviation of OPES kernels, if None, kernel_std will be estimated from initial MD with `adaptive_std_freq*update_freq` steps
        update_freq=100,            # Frequency of adding new Gaussian kernels in MD steps (e.g. every 100 steps)
        energy_barr=20.0,           # Barrier factor in kJ/mol, which sets an upper bound to the bias potential, should roughly correspond to the energy barrier of the transition to be sampled.
        bandwidth_rescaling=True,   # If True, the kernel standard deviation shrinks over time to converge finer details of the PMF.
        bias_factor=None,           # The bias factor gamma, which controls the smoothing of the probability density, Default: default is `beta * energy_barr`
        adaptive_std=True,         # If True, the kernel standard deviation is adapted based on the standard deviation of the CV, useful for simulations using poor CVs. 
        adaptive_std_freq=10,       # Exponential decay time for running estimate of the CVs standard deviation
        explore=False,              # If True, use the exploration mode of OPES.
        normalize=True,             # Always recommended. Normalize OPES probability density over explored space. 
        approximate_norm=True,      # Always recommended. Enables linear scaling approximation of the normalization factor, which is faster.
        merge_threshold=1.0,        # Threshold for merging Gaussian kernels, if the Mahalanobis distance between two kernels is smaller than this threshold, they are merged.
        recursive_merge=True,       # Always recommended. If True, recursively merge Gaussian kernels until no more kernels can be merged.
        force_from_grid=True,       # Always recommended. If True, bias potentials and forces are accumulated on a grid, if False, the sum of Gaussian hills is calculated in every step, which can be expensive for long runs.
        #...,                       # Additional inherited keyword arguments from the `EnhancedSampling` class.
    )

While the OPES implementation features many options, most of them are not critical and should almost always be left at the default option. For the subset of parameters that toggle the features of the present OPES implementationthe default settings have been shown to provide the best results in general. These include the linear scaling normalization (`approximate_norm=True` and `normalize=True`), the saturating number of Gaussian kernels by recursive merging (`recursive_merge=True`, `merge_threshold=1.0`), and the efficient refinement over time (`bandwidth_rescaling=True`, `adaptive_std=True`). 
Furthermore, there are parameters that can affect the simulation within a given choice of features. 
 - The `kernel_std` parameter controls the intial standard deviation of the kernels and significantly enhances the efficiency if being set to provide fast escape from the intial basin. However, if there is no preliminary knowledge about the PMF one can set `kernel_std=None` to automatically estimate the initial standard deviation from an unbiased MD trajectory in the starting basin.
 - The `update_freq` parameter controls how often new Gaussian kernels are added to the bias potential. A value of 100 to 500 steps is a good default, but it can be adjusted based on the system and the desired sampling rate.
 - The `energy_barr` parameter sets the barrier height for the bias potential. This value should be chosen based on the expected energy barrier of the transition being sampled.

A more minimalistic example of using OPES is given below:

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import OPES

    the_md = ... # initialize the molecular dynamics interface from adaptive_sampling.interface
    the_cv = ... # define the collective variable (CV) using adaptive_sampling.colvars

    the_bias = OPES(
        the_md,
        the_cv,
        kernel_std=None,            # Estimate initial standard deviation from `adaptive_std_freq*update_freq` initial steps
        update_freq=100,            # Frequency of adding new Gaussian kernels in MD steps (e.g. every 100 steps)
        energy_barr=20.0,           # Expected energy barrier in kJ/mol
        adaptive_std_freq=10,       # Initial kernel standard deviation obtained from `adaptive_std_freq*update_freq` MD steps (1000 steps).
        #...,                       # Additional inherited keyword arguments from the `EnhancedSampling` class.
    )

A variant of the OPES method is `OPES_Explore`, which is designed to maximize sampling of the CV space. Exploration can come to halt if the algorithm discovers a new meta-stable state in an already explored CV region for a suboptimal variable. This would not lead to a significant change in the normalization factor Zn , which should refer to the exploration history and contribute to pushing the system out of local minima. In the case of exploring a already sampled space, the bias needs to change, so that an exit can effectively be accelerated. With OPESexplore, the bias is built from an on-the-fly estimate of the sampled distribution where for this variant the target distribution :math:`p^{target}(\xi)= p^{WT}(\xi)` is well tempered. This altered target distribution leads to a sacrifice in convergence efficiency and can even prevent the system from converging to the correct PMF. It can be used to sample an unknown CV space efficiently when accuracy is not the primary concern, but rather the exploration of the CV space. Then again, OPES can be used to refine the sampling and provide accurate results.

For more information on OPES and OPES explore, refer to the original publications:
    * OPES: https://doi.org/10.1021/acs.jpclett.0c00497
    * OPES_Explore: https://doi.org/10.1021/acs.jctc.2c00152

Extended-system dynamics
------------------------

In extended system dynamics, additional degrees of freedom, which are suspect of the same dynamics as the physical system, are harmonically coupled to the CVs and act as proxies for the application of time-dependent bias potentials.

.. math::

    U^\mathrm{ext}(\mathbf{x}, \lambda) = U(\mathbf{x})+\sum_{i=1}^d \frac{1}{2 \beta\sigma_i^2}\left(\xi_i(\mathbf{x})-\lambda_i\right)^2 + U^\mathrm{bias}(\lambda,t)\,,

where :math:`\lambda` denotes additional degrees of freedom (extended system), :math:`\xi_i(\mathbf{x})` are the CVs, and :math:`\sigma_i` is the coupling width of the extended system to CVs and :math:`U^{bias}(\lambda,t)` can be any time-dependent bias potantial acting on :math:`\lambda`.

Multiple methods based on extended system dynamics are implemented, with differ in how the bias potential :math:`U^{bias}(\lambda,t)` is constructed:

 * `eABF`: extended adaptive biasing force (Ref: https://doi.org/10.1021/acs.jpcb.6b10055)
 * `WTMeABF`: applies both the WTM and ABF bias potentials to the extended system (Ref: https://doi.org/10.1021/acs.accounts.9b00473)
 * `OPESeABF`: applies both the OPES and ABF bias potentials to the extended system (Ref: https://doi.org/10.1021/acs.jctc.5c00395)

The different types of extended-system dynamics can be used as follows:

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import eABF, WTMeABF, OPESeABF

    the_md = ... # initialize the molecular dynamics interface using the `adaptive_sampling.interface` module
    the_cv = ... # define the collective variable (CV) using the `adaptive_sampling.colvars` module

    the_bias = eABF(
        the_md,
        the_cv,
        ext_sigma=0.1,          # Coupling width of the extended system to CVs in units of the CV (e.g. Angstrom for distance CVs, degree for angle CVs)
        ext_mass=100,           # The bias factor gamma, which controls the smoothing of the bias potential, Default: None
        nfull=100,              # Defines linear ramp for scaling up the adaptive biasing force (ABF), at `nfull` samples the full force is applied. 
        #...,                   # Additional inherited keyword arguments from the `ABF`, and `EnhancedSampling` class.
    )

    the_bias = WTMeABF(
        the_md,
        the_cv,
        ext_sigma=0.1,          # Coupling width of the extended system to CVs in units of the CV (e.g. Angstrom for distance CVs, degree for angle CVs)
        ext_mass=100,           # The bias factor gamma, which controls the smoothing of the bias potential, Default: None
        enable_abf=True,        # If True, the ABF bias is applied to the extended system
        nfull=100,              # Defines linear ramp for scaling up the adaptive biasing force (ABF), at `nfull` samples the full force is applied. 
        hill_height=1.0,        # Height of the Gaussian hills in kJ/mol
        hill_std=0.2,           # Standard deviation of the Gaussian hills in units of the CV (e.g. Angstrom for distance CVs, degree for angle CVs), can also be a list of floats for 2D CVs
        hill_drop_freq=100,     # Frequency of adding new Gaussian hills in MD steps (e.g. every 100 steps)
        #...,                   # Additional inherited keyword arguments from the `WTM`, `ABF` and `EnhancedSampling` class.
    )

    the_bias = OPESeABF(
        the_md,
        the_cv,
        ext_sigma=0.1,          # Coupling width of the extended system to CVs in units of the CV (e.g. Angstrom for distance CVs, degree for angle CVs)
        ext_mass=100,           # The bias factor gamma, which controls the smoothing of the bias potential, Default: None
        enable_abf=True,        # If True, the ABF bias is applied to the extended system
        nfull=100,              # Defines linear ramp for scaling up the adaptive biasing force (ABF), at `nfull` samples the full force is applied. 
        kernel_std=0.1,         # Initial standard deviation of OPES kernels, if None, kernel_std will be estimated from initial MD with `adaptive_std_freq*update_freq` steps
        update_freq=100,        # Frequency of adding new Gaussian kernels in MD steps (e.g. every 100 steps)
        energy_barr=20.0,       # Expected energy barrier in kJ/mol
        #...,                   # Additional inherited keyword arguments from the `OPES`, `ABF` and `EnhancedSampling` class.
    )

Accelerated molecular dynamics (aMD)
-------------------------------------

Accelerated molecular dynamics (aMD) is a method to enhance the sampling of rare events by globally modifying the potential energy surface of the system to lower the energy barriers of transitions.
Especially, aMD methods do not require a CV, but instead apply a bias potential to the entire system.

The bias potential is given by:

.. math::

    U^\mathrm{aMD}(\mathbf{x}, U) = 
        \begin{cases}
            U(\mathbf{x}) & \mathrm{if} \; U(\mathbf{x}) \geq E, \\
            U(\mathbf{x}) + \Delta U(U(\mathbf{x})) & \mathrm{if} \; U(\mathbf{x}) <  E \:.
        \end{cases}

where :math:`E` is a threshold energy and :math:`\Delta U(U(\mathbf{x}))` is the boost energy.
For the boost energy, different options are available:

 * `aMD`: accelerated MD as introduced by Hamelberg et al. * `aMD`: accelerated MD as introduced by Hamelberg et al. (https://doi.org/10.1063/1.1755656)
 * `GaMD`: Gaussian accelerated MD as introduced by Miao et al. (https://doi.org/10.1021/acs.jctc.5b00436)
 * `SaMD`: Sigmoid accelerated MD as introduced by Zhao et al. (https://doi.org/10.1021/acs.jpclett.2c03688)

The different types of aMD can be used as follows:

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import aMD

    the_md = ... # initialize the molecular dynamics interface using the `adaptive_sampling.interface` module
    the_cv = ... # define the collective variable (CV) using the `adaptive_sampling.colvars` module

    the_bias = aMD(
        amd_parameter,             # Acceleration parameter; SaMD, GaMD == sigma0; aMD == alpha
        init_step,                 # Initial steps where no bias is applied to estimate min, max and var of potential energy
        equil_steps,               # Equilibration steps, min, max and var of potential energy is still updated, force constant of coupling is calculated from previous steps
        the_md,                    # The MD interface from `adaptive_sampling.interface`
        the_cv,                    # The CV does not affect sampling in aMD, but is still required for the `EnhancedSampling` base class. Can be used to monitor CVs of interest.
        amd_method='GaMD_lower',   # 'aMD': accelerated MD, 'GaMD_lower': lower bound of Gaussian accelerated MD, 'GaMD_upper': upper bound of Gaussian accelerated MD, 'SaMD': sigmoid accelerated MD
        confine=False,             # If system should be confined at boundaries of the CV definition with harmonic walls.
        #...,                      # Additional inherited keyword arguments from the `EnhancedSampling` class.
    )

The global conformational sampling as provided by `aMD` can be combined with local sampling acceleration of the selected CVs in the `WTMeABF` method (Ref: https://doi.org/10.1021/acs.jctc.1c00103):

.. code-block:: python
    :linenos:

    from adaptive_sampling.sampling_tools import aWTMeABF

    the_md = ... # initialize the molecular dynamics interface using the `adaptive_sampling.interface` module
    the_cv = ... # define the collective variable (CV) using the `adaptive_sampling.colvars` module

    the_bias = aWTMeABF(
        amd_parameter,             # Acceleration parameter; SaMD, GaMD == sigma0; aMD == alpha
        init_step,                 # Initial steps where no bias is applied to estimate min, max and var of potential energy
        equil_steps,               # Equilibration steps, min, max and var of potential energy is still updated, force constant of coupling is calculated from previous steps
        the_md,                    # The MD interface from `adaptive_sampling.interface`
        the_cv,                    # The CV does not affect sampling in aWTMeABF, but is still required for the `EnhancedSampling` base class. Can be used to monitor CVs of interest.
        amd_method='GaMD_lower',   # 'aMD': accelerated MD, 'GaMD_lower': lower bound of Gaussian accelerated MD, 'GaMD_upper': upper bound of Gaussian accelerated MD, 'SaMD': sigmoid accelerated MD
        confine=True,              # If system should be confined at boundaries of the CV definition with harmonic walls.
        ext_sigma=0.1,             # Coupling width of the extended system to CVs in units of the CV (e.g. Angstrom for distance CVs, degree for angle CVs)
        ext_mass=100,              # The bias factor gamma, which controls the smoothing of the bias potential, Default: None
        nfull=100,                 # Defines linear ramp for scaling up the adaptive biasing force (ABF), at `nfull` samples the full force is applied. 
        hill_height=1.0,           # Height of the Gaussian hills in kJ/mol
        hill_std=0.2,              # Standard deviation of the Gaussian hills in units of the CV (e.g. Angstrom for distance CVs, degree for angle CVs), can also be a list of floats for 2D CVs
        hill_drop_freq=100,        # Frequency of adding new Gaussian hills in MD
        #...,                      # Additional inherited keyword arguments from the `aMD`, `WTM`, `ABF` and `EnhancedSampling` classes.
    )