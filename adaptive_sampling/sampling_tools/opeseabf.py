import os, time
import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var, combine_welford_stats, diff
from .eabf import eABF
from .opes import OPES
from ..units import *
from adaptive_sampling.processing_tools.thermodynamic_integration import *

class OPESeABF(eABF, OPES, EnhancedSampling):
    """Well-Tempered Metadynamics extended-system Adaptive Biasing Force method

       see: Fu et. al., J. Phys. Chem. Lett. (2018); https://doi.org/10.1021/acs.jpclett.8b01994

    The collective variable is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particel is biased using a combination of ABF and Metadynamics.

    Args:
        ext_sigma: thermal width of coupling between collective and extended variable
        ext_mass: mass of extended variable in atomic units
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        friction: friction coefficient for Lagevin dynamics of the extended-system
        seed_in: random seed for Langevin dynamics of extended-system
        nfull: Number of force samples per bin where full bias is applied,
               if nsamples nfull the bias force is scaled down by nsamples/nfull
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
        kernel_std: standard deviation of first kernel
        explore: enables exploration mode
        energy_barr: free energy barrier that the bias should help to overcome [kJ/mol]
        update_freq: interval of md steps in which new kernels should be 
        approximate_norm: enables approximation of norm factor
        exact_norm: enables exact calculation of norm factor, if both are enabled, exact is used every 100 updates
        merge_threshold: threshold distance for kde-merging in units of std, "np.inf" disables merging
        recursion_merge: enables recursive merging
        bias_factor: allows setting a default bias factor instead of calculating it from energy barrier
        print_pmf: enables calculation of pmf on the fly
        adaptive_sigma: enables adaptive sigma calculation nontheless with rescaling
        unbiased_time: time in update frequencies for unbiased estimation of sigma if no input is given
        fixed_sigma: disables bandwidth rescaling and uses input sigma for all kernels
        enable_eabf: enables eABF biasing, if False only OPES is applied

    """

    def __init__(
        self, 
        *args,
        enable_eabf: bool = True,
        enable_opes: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.abf_forces = np.zeros_like(self.bias)
        self.enable_eabf = enable_eabf
        self.enable_opes = enable_opes
        if self.enable_eabf == False and self.enable_opes == False:
            raise ValueError(" >>> Error: At least one biasing method has to be enabled!")

    def step_bias(
        self, 
        write_output: bool = True, 
        write_traj: bool = True, 
        stabilize: bool = False,
        stabilizer_threshold: float = None,
        output_file: str = 'wtmeabf.out',
        traj_file: str = 'CV_traj.dat', 
        restart_file: str = 'restart_wtmeabf',
        **kwargs,
    ) -> np.ndarray:
        """Apply WTM-eABF to MD simulation

        Args:
            write_output: if on-the-fly free energy estimate and restart files should be written
            write_traj: if CV and extended system trajectory file should be written
            stabilize: if stabilisation algorithm should be applied for discontinous CVs
            stabilizer_threshold: treshold for stabilisation of extended system
            output_file: name of the output file
            traj_file: name of the trajectory file
            restart_file: name of the restart file

        Returns:
            bias_force: WTM-eABF biasing force of current step that has to be added to molecular forces
        """

        self.md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        if stabilize and len(self.traj)>0:
            self.stabilizer(xi, threshold=stabilizer_threshold)

        self._propagate()

        mtd_forces = self.opes_bias(np.copy(self.ext_coords)) / atomic_to_kJmol / kJ_to_kcal
        bias_force = self._extended_dynamics(xi, delta_xi)  # , self.hill_std)
        force_sample = [0 for _ in range(2 * self.ncoords)]

        if self._check_boundaries(self.ext_coords):

            bin_la = self.get_index(self.ext_coords)
            self.ext_hist[bin_la[1], bin_la[0]] += 1

            for i in range(self.ncoords):
                
                if self.enable_eabf: 

                    # linear ramp function
                    ramp = (
                        1.0
                        if self.ext_hist[bin_la[1], bin_la[0]] > self.nfull
                        else self.ext_hist[bin_la[1], bin_la[0]] / self.nfull
                    )

                    # apply bias force on extended variable
                    force_sample[i] = self.ext_k[i] * diff(self.ext_coords[i], xi[i], self.periodicity[i])
                    (
                        self.abf_forces[i][bin_la[1], bin_la[0]],
                        self.m2_force[i][bin_la[1], bin_la[0]],
                        self.var_force[i][bin_la[1], bin_la[0]],
                    ) = welford_var(
                        self.ext_hist[bin_la[1], bin_la[0]],
                        self.abf_forces[i][bin_la[1], bin_la[0]],
                        self.m2_force[i][bin_la[1], bin_la[0]],
                        force_sample[i],
                    )
                    if self.enable_opes:
                        self.ext_forces -= (ramp * self.abf_forces[i][bin_la[1], bin_la[0]] - mtd_forces[i])
                    else:
                        self.ext_forces -= ramp * self.abf_forces[i][bin_la[1], bin_la[0]]
                else:
                    self.ext_forces += mtd_forces[i]
        # xi-conditioned accumulators for CZAR
        if self._check_boundaries(xi):
            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                force_sample[self.ncoords+i] = self.ext_k[i] * diff(
                    self.ext_coords[i], self.grid[i][bink[i]], self.periodicity[i]
                )
                self.correction_czar[i][bink[1], bink[0]] += force_sample[self.ncoords+i]
        
        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        # Save values for traj
        self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
        self.traj = np.append(self.traj, [xi], axis=0)
        self.epot.append(self.md_state.epot)
        self.temp.append(self.md_state.temp)

        if self.md_state.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj(filename=traj_file)

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"opesforce {i}"] = mtd_forces[i]
                    output[f"abf force {i}"] = self.abf_forces[i]
                    output[f"czar force {i}"] = self.czar_force[i]
                    # TODO: output variance of CZAR for error estimate
                    # output[f"var force {i}"] = self.var_force[i]
                output[f"opespot"] = self.potential

                self.write_output(output, filename=output_file)
                #self.write_restart(filename=restart_file)
        #if self.md_state.step % self.update_freq == 0:
        #print("mtd forces", mtd_forces)
        #print(self.ext_forces+mtd_forces)
        return bias_force


    def reinit(self):
        """Reinit WTM-eABF and start building new bias potential
        """
        self.histogram = np.zeros_like(self.histogram)
        self.bias = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.m2_force)
        self.ext_hist = np.zeros_like(self.ext_hist)
        self.correction_czar = np.zeros_like(self.correction_czar)
        self.abf_forces = np.zeros_like(self.abf_forces)
        self.center = []
        self.metapot = np.zeros_like(self.metapot)
        self.reinit_ext_system(self.traj[-1])

    def write_restart(self, filename: str="restart_wtmeabf"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            m2=self.m2_force,
            ext_hist=self.ext_hist,
            czar_corr=self.correction_czar,
            abf_force=self.abf_forces,
            ext_momenta=self.ext_momenta,
            ext_coords=self.ext_coords,
            sum_weigths=self.sum_weights,
            sum_weigths_square = self.sum_weights_square,
            norm_factor = self.norm_factor,
            kernel_heigth = self.kernel_height,
            kernel_center = self.kernel_center,
            kernel_sigma = self.kernel_sigma,
            explore = self.explore,
            n = self.n
        )

    def restart(self, filename: str = "restart_wtmeabf", restart_ext_sys: bool=False):
        """restart from restart file

        Args:
            filename: name of restart file
            restart_ext_sys: restart coordinates and momenta of extended system
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.histogram = data["hist"]
        self.m2_force = data["m2"]
        self.ext_hist = data["ext_hist"]
        self.correction_czar = data["czar_corr"]
        self.abf_forces = data["abf_force"]
        self.sum_weights = float(data["sum_weigths"])
        self.sum_weights_square = float(data["sum_weigths_square"])
        self.norm_factor = float(data["norm_factor"])
        self.kernel_height = data["kernel_height"]
        self.kernel_center = data["kernel_center"]
        self.kernel_sigma = data["kernel_sigma"]
        self.explore = data["explore"]
        self.n = data["n"]
        if restart_ext_sys:
            self.ext_momenta = data["ext_momenta"]
            self.ext_coords = data["ext_coords"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restarted from {filename}!")

    def write_traj(self, filename: str = 'CV_traj.dat'):
        """save trajectory for post-processing"""

        data = self._write_ext_traj()
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp

        self._write_traj(data, filename=filename)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.ext_traj = np.array([self.ext_traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]

    def czar_pmf_history1d(
        self,
        grid: np.array,
        cv_x: np.array,
        cv_la: np.array,
        ext_sigma: float,
        pmf_hist_res: int = 100,
    ):
        """Calculate PMF history for CZAR
        
        Args:
            grid: grid for CZAR
            cv_x: CV trajectory
            cv_la: extended system trajectory
            ext_sigma: thermal width of coupling between CV and extended variable
            pmf_hist_res: resolution of PMF history
            
        Returns:
            pmf_hist: PMF history
            scattered_time: scattered time points
            rho_hist: density history
        """
        
        dx = grid[1]-grid[0]
        n = int(len(cv_x)/pmf_hist_res)
        scattered_time = []
        pmf_hist = []
        rho_hist = []
        print("Integrating CZAR...")
        for j in range(pmf_hist_res):
            n_sample = j * n + n
            scattered_time.append(n_sample)
            czar_grad = czar(grid, cv_x[0:n_sample], cv_la[0:n_sample], ext_sigma)
            pmf, rho = integrate(czar_grad, dx)
            pmf_hist.append(pmf)
            rho_hist.append(rho)
        print("Done!")

        return pmf_hist, scattered_time, rho_hist