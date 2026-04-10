Processing tools
================

The `processing_tools` subpackage contains functions for the analysis of free energy or exploratory simulations. 

Extended-system dynamics
------------------------

The analysis of extended system dynamics trajectories only requires the trajectories of the CVs and of the coupled degrees of freedom :math:`\lambda`.
Per default, both are stored in the `CV_traj.dat` file during simulations. You can read the `.dat` file for example using numpy:

.. code-block:: python
    :linenos:

    import numpy as np

    data = np.loadtxt("CV_traj.dat", skiprows=1)  
    
    time = data[:,0]    # The simulation time 
    cv   = data[:,1]    # The trajectory of the CV
    la   = data[:,2]    # The trajectory of the fictitious particle

A fast estimate of the PMF can be obtained from thermodynamic integration using the corrected z-averaged restraint (CZAR) estimator as described by `Lesage et al. <https://doi.org/10.1021/acs.jpcb.6b10055>`_:

.. code-block:: python
    :linenos:

    import adaptive_sampling.processing_tools.thermodynamic_integration as ti

    ext_sigma = 0.1                         # The coupling width of the extended system 
    bin_width = 0.1                         # Bin width of the grid for the integration and final PMF

    grid = np.arange(1.5, 4.0, bin_width)   # Setup the grid for TI

    mean_force = ti.czar(
        grid,
        cv,
        la, 
        ext_sigma,
        equil_temp=300.0,
    )

    pmf, rho = ti.integrate(
        mean_force,
        bin_width=bin_width,
        equil_temp=300.0,
    )

Unbiased statistical weights of data frames can be obtained from the MBAR estimator as described `here <https://doi.org/10.1063/5.0095554>`_.
For this purpose the continuous trajectories of CVs and :math:`\lambda`'s are transformed into static simulation windows, to which the MBAR estimator is applied:

.. code-block:: python
    :linenos:

    from adaptive_sampling.processing_tools import mbar

    # obtain simulation windows from continous trajectory
    traj_list, indices, meta_f = mbar.get_windows(
        grid, 
        cv, 
        la, 
        ext_sigma, 
        equil_temp=300.0
    )

    # run the MBAR estimator to get the statistical weights of simulation frames
    exp_U, frames_per_traj = mbar.build_boltzmann(
        traj_list, 
        meta_f, 
        equil_temp=300.0,
    )

    weights = mbar.run_mbar(
        exp_U,
        frames_per_traj,
        max_iter=10000,
        conv=1.0e-4,
        conv_errvec=1.0,
        outfreq=100,
        device='cpu',
    )

    # obtain the PMF from the statistical weights
    pmf, rho = mbar.pmf_from_weights(
        grid, 
        cv[indices], # order according to `indices`, such that frames in `weights` match frames in `cv` arrays
        weights, 
        equil_temp=300.0,
    )






