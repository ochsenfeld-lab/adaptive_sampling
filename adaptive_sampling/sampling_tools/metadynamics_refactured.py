import numpy as np
from adaptive_sampling.sampling_tools.enhanced_sampling import EnhancedSampling
from adaptive_sampling.units import *
from .utils import correct_periodicity
from .utils import welford_var


class WTM(EnhancedSampling):
    """Well-Tempered Metadynamics

    see: Barducci et. al., Phys. rev. lett. (2008); https://doi.org/10.1103/PhysRevLett.100.020603

    An repulsive biasing potential is built by a superposition of Gaussian hills along the reaction coordinate.

    Args:
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CVs (can be Bohr, Degree, or None)
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if None, hills are not scaled down (normal metadynamics)
        force_from_grid: forces are accumulated on grid for performance (recommended),
                         if False, forces are calculated from sum of Gaussians in every step
        estimator: if "TI", PMF is estimated from integral of bias force, else PMF directly estimated from force
    """

    def __init__(
        self,
        hill_height: float,
        hill_std: np.array,
        *args,
        hill_drop_freq: int = 20,
        well_tempered_temp: float = 3000.0,
        estimator: str = "Potential",
        force_from_grid: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if int(hill_drop_freq) <= 0:
            raise ValueError(" >>> Error: Update interval has to be int > 0!")
        if hill_height <= 0:
            raise ValueError(" >>> Error: Gaussian height for MtD has to be > 0!")
        if well_tempered_temp is None:
            if self.verbose:
                print(
                    " >>> Info: Well-tempered scaling of MtD hill_height switched off"
                )
        elif well_tempered_temp <= 0:
            raise ValueError(
                " >>> Error: Effective temperature for Well-Tempered MtD has to be > 0!"
            )
    
        # initialize MtD parameters
        hill_std = [hill_std] if not hasattr(hill_std, "__len__") else hill_std
        self.numerical_forces = force_from_grid
        self.verbose = verbose
        self.well_tempered_temp = well_tempered_temp
        self.well_tempered = False if well_tempered_temp == None else True
        self.force_from_grid = force_from_grid
        self.estimator = estimator
        self.hill_drop_freq = hill_drop_freq

        # hill parameters
        if not hasattr(hill_std, "__len__"):
            self.hill_std = self.unit_conversion_cv(np.asarray([hill_std]))[0]
        else:
            self.hill_std = self.unit_conversion_cv(np.asarray(hill_std))[0]
        self.hill_height = hill_height / atomic_to_kJmol
        self.hills_center = []
        self.hills_height = []
        self.hills_std = []

        self.bias_potential = np.copy(self.histogram)
        self.bias_pot = 0.0
        self.bias_pot_traj = []

    def step_bias(
        self,
        traj_file: str = "CV_traj.dat",
        out_file: str = "opes.out",
        restart_file: str = "restart_opes",
        **kwargs,
    ) -> np.array:
        """Apply OPES bias to MD

        Returns:
            bias_force: Bias force that has to be added to system forces
        """

        self.md_state = self.the_md.get_sampling_data()
        (cv, grad_cv) = self.get_cv(**kwargs)

        # get mtd bias force
        mtd_forces = self.mtd_bias(cv)
        bias_force = self.harmonic_walls(cv, grad_cv)
        for i in range(self.ncoords):
            bias_force += mtd_forces[i] * grad_cv[i]

        # correction for kinetics
        if self.kinetics:
            self._kinetics(grad_cv)

        # shared-bias metadynamics
        #if self.shared:
        #    self.shared_bias(cv, **kwargs)

        # store biased histogram along CV for output
        if out_file and self._check_boundaries(cv):
            bink = self.get_index(cv)
            self.histogram[bink[1], bink[0]] += 1

        # Save values for traj output
        if traj_file:
            self.traj = np.append(self.traj, [cv], axis=0)
            self.epot.append(self.md_state.epot)
            self.temp.append(self.md_state.temp)
            self.bias_pot_traj.append(self.bias_pot)

        # Write output
        if self.md_state.step % self.out_freq == 0:
            if traj_file and len(self.traj) >= self.out_freq:
                self.write_traj(filename=traj_file)
            if out_file:
                self.pmf = self.get_pmf()
                output = {
                    "hist": self.histogram,
                    "free energy": self.pmf * atomic_to_kJmol,
                    "OPES Pot": self.bias_potential * atomic_to_kJmol,
                }
                self.write_output(output, filename=out_file)
            if restart_file:
                self.write_restart(filename=restart_file)

        return bias_force

    def get_pmf(self) -> np.array:
        """Calculate current PMF estimate on `self.grid`

        Returns:
            pmf: current PMF estimate from OPES kernels
        """

        pmf -= pmf.min()
        return pmf

    def write_traj(self, filename="CV_traj.dat"):
        data = {
            "Epot [Ha]": self.epot,
            "T [K]": self.temp,
            "Biaspot [Ha]": self.bias_pot_traj,
        }
        self._write_traj(data, filename=filename)

        # Reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
        self.bias_pot_traj = [self.bias_pot_traj[-1]]

    def shared_bias(self):
        raise ValueError(
            " >>> ERROR: Multiple-walker shared bias not available for MtD!"
        )

    def write_restart(self, filename: str = "restart_opes"):
        """Dumps state of OPES to restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            height=self.hills_height,
            center=self.hills_center,
            sigma=self.hills_std,
        )

    def restart(self, filename: str = "restart_opes"):
        """Restart OPES from previous simulation

        Args:
            filename: name of restart
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> ERROR: restart file {filename}.npz not found!")

        # Load dictionary entries from restart file
        self.hills_height = data["height"]
        self.hills_center = data["center"]
        self.hills_std = data["sigma"]
        if self.verbose and self.md_state.step % self.hill_drop_freq == 0:
            print(f" >>> Info: Adaptive sampling restarted from {filename}!")

    def mtd_bias(self, cv: np.array) -> np.array:
        """Calculate MtD bias force from KDE of hills
        which is updated according to `self.update_freq`

        Args:
            cv: new value of CV

        Returns:
            bias force: len(ncoords) array of bias forces
        """
        # get bias potential and forces
        if self.numerical_forces and self._check_boundaries(cv):
            idx = self.get_index(cv)
            self.bias_pot = self.bias_potential[idx[1], idx[0]]
            mtd_force = [self.bias[i][idx[1], idx[0]] for i in range(self.ncoords)]
        else:
            gaussians = self.calc_gaussians(cv)
            mtd_force = gaussians

        # OPES KDE update
        if self.md_state.step % self.update_freq == 0:
            self.update_kde(cv)

        return mtd_force

    def calc_gaussians(self, cv, requires_grad: bool = False) -> np.array:
        """Get normalized value of gaussian hills

        Args:
            cv: value of CV where the kernels should be evaluated
            requires_grad: if True, accumulated gradient of KDE is returned as second argument

        Returns:
            gaussians: values of gaussians at CV
            kde_derivative: derivative of KDE, only if requires_grad
        """

        if len(self.kernel_center) == 0:
            if requires_grad:
                return 0.0, np.zeros(self.ncoords)
            return 0.0

        # distance to kernel centers
        s_diff = cv - np.asarray(self.hills_center)
        for i in range(self.ncoords):
            s_diff[:, i] = correct_periodicity(s_diff[:, i], self.periodicity[i])

        # evaluate values of kernels at cv
        gaussians = np.asarray(self.hills_height) * np.exp(
            -0.5
            * np.sum(np.square(np.divide(s_diff, np.asarray(self.hills_std))), axis=1)
        )
        if requires_grad:
            kde_derivative = (
                np.sum(
                    -gaussians
                    * np.divide(s_diff, np.square(np.asarray(self.hills_std))).T,
                    axis=1,
                )
            )
            return gaussians, kde_derivative

        return gaussians

    def update_kde(self, cv: np.array):
        """on-the-fly update of kernel density estimation of probability density along CVs

        Args:
            CV: new value of CVS
        """
        self.add_kernel(self.hill_height, cv, self.hill_std)
        self.grid_potential()

    def add_kernel(self, h_new: float, s_new: np.array, std_new: np.array):
        """Add new Kernel to KDE

        Args:
            h_new: hills height
            s_new: hills position
            std_new: hills standard deviation
        """
        self.hills_height.append(h_new)
        self.hills_center.append(s_new)
        self.hills_std.append(std_new)

    def grid_potential(self):
        """Calculate bias potential and forces from kernels in bins of `self.grid`"""
        potential = np.zeros_like(self.histogram)
        derivative = np.zeros_like(self.bias)
        if self.ncoords == 1:
            for i, cv in enumerate(self.grid[0]):
                if not self.numerical_forces:
                    val_gaussians = self.calc_gaussians(cv)
                else:
                    val_gaussians, kde_der = self.calc_gaussians(cv, requires_grad=True)
                    derivative[0][0, i] = kde_der
                potential[0, i] = np.sum(val_gaussians)
        else:
            for i, x in enumerate(self.grid[0]):
                for j, y in enumerate(self.grid[1]):
                    if not self.numerical_forces:
                        val_gaussians = self.calc_gaussians(np.asarray([x, y]))
                    else:
                        val_gaussians, kde_der = self.calc_gaussians(
                            np.asarray([x, y]), requires_grad=True
                        )
                        derivative[0][j, i] = kde_der[0]
                        derivative[1][j, i] = kde_der[1]
                    potential[j, i] = np.sum(val_gaussians)

        self.bias_potential = potential
        if self.numerical_forces:
            for i in range(self.ncoords):
                self.bias[i] = potential * derivative[i]