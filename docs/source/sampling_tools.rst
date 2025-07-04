Sampling tools
==============

The `sampling_tools` subpackage provides a set of tools for importance sampling of molecular transitions.

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
        **kwargs,                   # Additional inherited keyword arguments from the `EnhancedSampling` class.
    )

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
        adaptive_std=False,         # If True, the kernel standard deviation is adapted based on the standard deviation of the CV, useful for simulations using poor CVs. 
        adaptive_std_freq=10,       # Exponential decay time for running estimate of the CVs standard deviation
        explore=False,              # If True, use the exploration mode of OPES.
        normalize=True,             # Always recommended. Normalize OPES probability density over explored space. 
        approximate_norm=True,      # Always recommended. Enables linear scaling approximation of the normalization factor, which is faster.
        merge_threshold=1.0,        # Threshold for merging Gaussian kernels, if the Mahalanobis distance between two kernels is smaller than this threshold, they are merged.
        recursive_merge=True,       # Always recommended. If True, recursively merge Gaussian kernels until no more kernels can be merged.
        force_from_grid=True,       # Always recommended. If True, bias potentials and forces are accumulated on a grid, if False, the sum of Gaussian hills is calculated in every step, which can be expensive for long runs.
        **kwargs,                   # Additional inherited keyword arguments from the `EnhancedSampling` class.
    )

While the OPES implementation features many options, most of them are not critical and should almost always be left at the default option. A more minimalistic example of using OPES is given below:

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
        **kwargs,                   # Additional inherited keyword arguments from the `EnhancedSampling` class.
    )

Extended-system dynamics
-------------------------

