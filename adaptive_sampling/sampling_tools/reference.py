import os
import time
import numpy as np

from adaptive_sampling.units import R_in_SI
from .enhanced_sampling import EnhancedSampling


class Reference(EnhancedSampling):
    """Unbiased simulation with restraint to region of interest in cv space with harmonic walls
    Can be used for equilibration prior to production or free energy estimation from unbiased simulation

    Args:
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step_bias(self, write_output: bool = True, write_traj: bool = True, traj_file="ref_traj.dat", **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        bias_force = np.zeros_like(md_state.forces)

        if self._check_boundaries(xi):

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            if self.kinetics:
                for i in range(self.ncoords):
                    self.cv_crit[i][bink[1], bink[0]] += abs(
                        np.dot(delta_xi[i], md_state.forces)
                    )

        else:
            bias_force = self.harmonic_walls(xi, delta_xi)

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        if md_state.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj(filename=traj_file)

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                self.write_output(output, filename="reference.out")
                self.write_restart()

        return bias_force

    def get_pmf(self) -> np.ndarray:
        """get free energy profile from histogram"""
        self.pmf = (
            -R_in_SI
            / 1000.0  # kJ/mol
            * self.equil_temp
            * np.log(
                self.histogram,
                out=np.zeros_like(self.histogram),
                where=(0 != self.histogram),
            )
        )
        self.pmf -= self.pmf.min()
        return self.pmf

    def shared_bias(
        self, filename: str = "./shared_bias", sync_interval: int = 100, trial: int = 10
    ):
        """sync histogram between multiple walkers

        TODO: fix me

        Args:
            filename: name of bias buffer
            sync_interval: interval in MD steps between syncs of bias
            trial: number of recursive calls if buffer is occupied by other walkers
        """
        # TODO: fix update here
        if os.path.isfile(filename):
            if not os.access(filename, os.W_OK):

                # deny access for other walkers during sync
                os.chmod(filename + ".npz", 0o644)
                sb = np.load(filename + ".npz", allow_pickle=True)
                try:
                    diff = self._last_sync_histogram - self.histogram
                except:
                    if self.verbose:
                        print(" >>> Info: new shared-bias instance created!")
                    self._local_histogram = np.copy(self.histogram)
                    self._last_sync_histogram = np.copy(self.histogram)
                    diff = np.copy(self.histogram)
                os.chmod(filename + ".npz", 0o444)

                self._local_histogram += diff
                self.histogram += diff
                self._last_sync_histogram = np.copy(self.histogram)

            elif trial > 0:
                # recursive calls to wait for other walker to finish writing
                time.sleep(0.1)
                self.shared_bias(filename, sync_interval=sync_interval, trial=trial - 1)

            elif self.verbose:
                print(f" >>> Warning: failed to sync bias with {filename}!")

        else:
            if self.verbose:
                print(f" >>> Info: created shared-bias buffer {filename}!")
            self.write_restart(filename=filename)
            self.write_restart(filename=filename + "_local")
            os.chmod(filename + ".npz", 0o444)

    def write_restart(self, filename: str = "restart_ref"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(filename=filename, hist=self.histogram, pmf=self.pmf)

    def restart(self, filename: str = "restart_ref"):
        """restart from restart file

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz", allow_pickle=True)
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.histogram = data["hist"]
        self.pmf = data["pmf"]

        if self.verbose:
            print(f" >>> Info: Sampling restartet from {filename}!")

    def write_traj(self, filename: str="CV_traj.dat"):
        """save trajectory for post-processing"""
        data = {
            "Epot [H]": self.epot,
            "T [K]": self.temp,
        }
        self._write_traj(data, filename=filename)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
