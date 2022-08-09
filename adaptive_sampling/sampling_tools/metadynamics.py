import os, time
import numpy as np
from typing import Tuple
from .enhanced_sampling import EnhancedSampling
from .utils import diff, sum
from ..units import *
from ..processing_tools.thermodynamic_integration import integrate


class WTM(EnhancedSampling):
    """Well-Tempered Metadynamics

       see: Barducci et. al., Phys. rev. lett. (2008); https://doi.org/10.1103/PhysRevLett.100.020603

    An repulsive biasing potential is built by a superposition of Gaussian hills along the reaction coordinate.

    Args:
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CV (can be Bohr, Degree, or None)
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if None, hills are not scaled down (normal metadynamics)
        force_from_grid: forces are accumulated on grid for performance (recommended),
                         if False, forces are calculated from sum of Gaussians in every step
        estimator: if "TI", PMF is estimated from integral of bias force, else PMF directly estimated from force
    """

    def __init__(
        self,
        hill_height: float,
        hill_std: list,
        *args,
        hill_drop_freq: int = 20,
        well_tempered_temp: float = 3000.0,
        force_from_grid: bool = True,
        estimator: str = "TI",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if int(hill_drop_freq) <= 0:
            raise ValueError(" >>> Error: Update interval has to be int > 0!")

        if hill_height <= 0:
            raise ValueError(" >>> Error: Gaussian height for MtD has to be > 0!")

        if well_tempered_temp is None and self.verbose:
            print(" >>> Info: well-tempered scaling of WTM hill_height switched off")
        elif well_tempered_temp <= 0:
            raise ValueError(
                " >>> Error: Effective temperature for Well-Tempered MtD has to be > 0!"
            )

        hill_std = [hill_std] if not hasattr(hill_std, "__len__") else hill_std

        self.hill_height = hill_height / atomic_to_kJmol
        self.hill_std = self.unit_conversion_cv(np.asarray(hill_std))[0]
        self.hill_var = self.hill_std * self.hill_std
        self.hill_drop_freq = int(hill_drop_freq)
        self.well_tempered_temp = well_tempered_temp
        self.well_tempered = False if well_tempered_temp == None else True
        self.force_from_grid = force_from_grid
        self.estimator = estimator

        self.metapot = np.zeros_like(self.histogram)
        self.center = []

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        bias_force = np.zeros_like(md_state.forces)

        mtd_force = self.get_wtm_force(xi)
        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                bias_force += mtd_force[i] * delta_xi[i]

        bias_force += self.harmonic_walls(xi, delta_xi)  # , 1.6 * self.hill_std)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

        # shared-bias metadynamics
        if self.shared:
            self.shared_bias(**kwargs)

        if self.the_md.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj()

            if write_output:
                self.get_pmf()
                output = {
                    "hist": self.histogram,
                    "free energy": self.pmf,
                    "metapot": self.metapot * atomic_to_kJmol,
                }

                self.write_output(output, filename="wtm.out")
                self.write_restart()

        return bias_force

    def get_pmf(self):
        if self.estimator == "TI" and self.ncoords == 1:
            # PMF from integration of bias force
            self.pmf[0, :], _ = integrate(
                -self.bias[0][0],
                self.dx,
                equil_temp=self.equil_temp,
                method="trapezoid",
            )

        else:
            # PMF from bias potential
            self.pmf = -self.metapot

        if self.well_tempered:
            self.pmf *= (
                self.equil_temp + self.well_tempered_temp
            ) / self.well_tempered_temp
        self.pmf *= atomic_to_kJmol
        self.pmf -= self.pmf.min()

    def shared_bias(
        self,
        sync_interval: int=5,
        mw_file: str="../shared_bias",
        n_trials: int=10,
    ):
        """TODO"""
        md_state = self.the_md.get_sampling_data()
        if md_state.step == 0:        
            if self.verbose:
                print(" >>> Info: Creating a new instance for shared-bias metadynamics.")
                print(" >>> Info: Data of local walker stored in `restart_wtm_local.npz`.")
            
            # create seperate restart file with local data only
            self._write_restart(
                filename="restart_wtm_local",
                hist=self.histogram,
                pmf=self.pmf,
                force=self.bias,
                metapot=self.metapot,
                centers=self.center,
            )
                        
            if not os.path.isfile(mw_file+".npz"):
                if self.verbose:
                    print(f" >>> Info: Creating buffer file for shared-bias metadynamics: `{mw_file}.npz`.")
                self._write_restart(
                    filename=mw_file,
                    hist=self.histogram,
                    pmf=self.pmf,
                    force=self.bias,
                    metapot=self.metapot,
                    centers=self.center,
                )
                os.chmod(mw_file + ".npz", 0o444)
            elif self.verbose:
                print(f" >>> Info: Syncing with existing buffer file for shared-bias metadynamics: `{mw_file}.npz`.")
        

        count = md_state.step % sync_interval
        if count == sync_interval-1:
            
            hist = np.zeros_like(self.histogram)
            for x in self.traj_last_sync:
                if (x <= self.maxx).all() and (x >= self.minx).all():
                    bink = self.get_index([x])
                    hist[bink[1], bink[0]] += 1
            self.traj_last_sync = [self.traj_last_sync[-1]]

            self.bias = np.zeros_like(self.bias)
            self.metapot = np.zeros_like(self.metapot)
            
            step = md_state.step
            md_state.step = self.hill_drop_freq
            for center in self.center_last_sync:
                self._accumulate_wtm_force([center])
            md_state.step = step
            
            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):

                    last_sync_bias = np.copy(self.bias)
                    last_sync_metapot = np.copy(self.metapot)
                    
                    v = self.verbose
                    self.verbose = False
                    self.restart(filename="restart_wtm_local")
                    self.verbose = v
                    self.bias += last_sync_bias
                    self.metapot += last_sync_metapot
                    self.get_pmf()

                    self._write_restart(
                        filename="restart_wtm_local",
                        hist=hist+self.histogram,
                        pmf=self.pmf,
                        force=self.bias,
                        metapot=self.metapot,
                        centers=np.append(self.center, self.center_last_sync),
                    )

                    # grant write access only to one walker during sync
                    os.chmod(mw_file + ".npz", 0o666) 
                    shared_data = np.load(mw_file + ".npz")  

                    self.histogram = shared_data["hist"] + hist 
                    self.bias = shared_data["force"] + self.bias
                    self.metapot = shared_data["metapot"] + self.metapot
                    self.center = np.append(shared_data["centers"], self.center_last_sync).tolist()
                    self.get_pmf()

                    self._write_restart(
                        filename=mw_file,
                        hist=self.histogram,
                        pmf=self.pmf,
                        force=self.bias,
                        metapot=self.metapot,
                        centers=self.center,                       
                    )                                                
                    os.chmod(mw_file + ".npz", 0o444)  # other walkers can access again
                    if self.verbose:
                        print(f" >>> Info: Metadynamics potential synced with `{mw_file}.npz`!")
                    break                     

                elif trial < n_trials:
                    if self.verbose:
                        print(f" >>> Warning: Retry to open shared buffer file after {trial} failed attempts.")
                    time.sleep(0.1)
                else:
                    raise Exception(f" >>> Fatal Error: Failed to sync bias with `{mw_file}.npz`.")   

    def get_wtm_force(self, xi: np.ndarray) -> list:
        """compute well-tempered metadynamics bias force from superposition of gaussian hills

        Args:
            xi: state of collective variable

        Returns:
            bias_force: bias force from metadynamics
        """
        if self.the_md.step % self.hill_drop_freq == 0:
            self.center.append(xi[0]) if self.ncoords == 1 else self.center.append(xi)

            if self.shared and self.the_md.get_sampling_data().step == 0:
                self.center_last_sync = [self.center[0]]
            else: 
                self.center_last_sync.append(xi[0])

        if self.shared and self.the_md.get_sampling_data().step == 0:
            self.traj_last_sync = [xi[0]]
        else:
            self.traj_last_sync.append(xi[0])

        is_in_bounds = (xi <= self.maxx).all() and (xi >= self.minx).all()
        bias_force = [0, 0]
        if is_in_bounds:
            bias_force = self._accumulate_wtm_force(xi)

        if not is_in_bounds or not self.force_from_grid:
            bias_force, _ = self._analytic_wtm_force(xi)

        return bias_force

    def _accumulate_wtm_force(self, xi: np.ndarray) -> list:
        """accumulate metadynamics potential and its gradient on grid

        Args:
            xi: state of collective variable

        Returns:
            bias_force: bias force from metadynamics
        """
        bink = self.get_index(xi)
        if self.the_md.step % self.hill_drop_freq == 0:
            if self.ncoords == 1:

                if self.well_tempered:
                    w = self.hill_height * np.exp(
                        -self.metapot[bink[1], bink[0]]
                        / (kB_in_atomic * self.well_tempered_temp)
                    )
                else:
                    w = self.hill_height

                dx = diff(self.grid[0], xi[0], self.cv_type[0])
                epot = w * np.exp(-(dx * dx) / (2.0 * self.hill_var[0]))
                self.metapot[0] += epot
                self.bias[0][0] -= epot * dx / self.hill_var[0]

                self._smooth_boundary_grid(xi, w)

            else:
                # TODO: implement for 2D
                raise NotImplementedError(
                    " >>> Error: metadynamics currently only supported for 1D coordinates"
                )

        return [self.bias[i][bink[1], bink[0]] for i in range(self.ncoords)]

    def _analytic_wtm_force(self, xi: np.ndarray) -> Tuple[list, float]:
        """compute analytic WTM bias force from sum of gaussians hills

        Args:
            xi: state of collective variable

        Returns:
            bias_force: bias force from metadynamics
        """
        local_pot = 0.0
        bias_force = [0.0 for _ in range(self.ncoords)]

        if len(self.center) == 0:
            if self.verbose:
                print(" >>> Warning: no metadynamics hills stored")
            return bias_force

        if self.ncoords == 1:

            dx = diff(xi[0], np.asarray(self.center), self.cv_type[0])
            ind = np.ma.indices((len(self.center),))[0]
            ind = np.ma.masked_array(ind)
            ind[abs(dx) > 3 * self.hill_std[0]] = np.ma.masked

            # can get slow in long run, so only iterate over significant elements
            for i in np.nditer(ind.compressed(), flags=["zerosize_ok"]):

                if self.well_tempered:
                    w = self.hill_height * np.exp(
                        -local_pot / (kB_in_atomic * self.well_tempered_temp)
                    )
                else:
                    w = self.hill_height

                epot = w * np.exp(-(dx[i] * dx[i]) / (2.0 * self.hill_var[0]))
                local_pot += epot
                bias_force[0] -= epot * dx[i] / self.hill_var[0]

                local_pot, bias_force = self._smooth_boundary_analytic(
                    xi, w, self.center[i], local_pot, bias_force
                )
        else:
            # TODO: implement for 2D
            pass

        return bias_force, local_pot

    def _smooth_boundary_grid(self, xi: np.ndarray, w: float):
        """ensures linear profile of bias potential at boundary

        see: Crespo et al. (2010), https://doi.org/10.1103/PhysRevE.81.055701

        Args:
            xi: Collective variable
        """
        chi1 = 3.0 * self.hill_std

        if self.ncoords == 1:
            dx_list = []
            w_list = []

            if diff(self.maxx[0], chi1[0], self.cv_type[0]) <= xi[0] <= self.maxx[0]:
                center_out = self.maxx[0] + diff(self.maxx[0], xi[0], self.cv_type[0])
                dx_list.append(diff(self.grid[0], center_out, self.cv_type[0]))
                w_list.append(w)
            else:
                # TODO: scale height of hills to ensure flat potential at boundary
                pass

            if self.minx[0] <= xi[0] <= sum(self.minx[0], chi1[0], self.cv_type[0]):
                center_out = self.minx[0] - diff(xi[0], self.minx[0], self.cv_type[0])
                dx_list.append(diff(self.grid[0], center_out, self.cv_type[0]))
                w_list.append(w)
            else:
                # TODO: scale height of hills to ensure flat potential at boundary
                pass

            for dx, W in zip(dx_list, w_list):
                epot = W * np.exp(-(dx * dx) / (2.0 * self.hill_var[0]))
                self.metapot[0] += epot
                self.bias[0][0] -= epot * dx / self.hill_var[0]

        else:
            # TODO: implement for 2D
            pass

    def _smooth_boundary_analytic(self, xi, w, center, local_pot, bias_force):
        """ensures linear profile at boundary

        see: Crespo et al. (2010), https://doi.org/10.1103/PhysRevE.81.055701

        Args:
            xi: Collective variable
        """
        chi1 = 3.0 * self.hill_std

        if self.ncoords == 1:
            if diff(self.maxx[0], chi1[0], self.cv_type[0]) <= center <= self.maxx[0]:
                center_out = self.maxx[0] + diff(self.maxx[0], center, self.cv_type[0])
                val = diff(xi[0], center_out, self.cv_type[0])
                epot = w * np.exp(-(val * val) / (2.0 * self.hill_var[0]))
                local_pot += epot
                bias_force[0] -= epot * val / self.hill_var[0]
            else:
                # TODO: scale height of hills to ensure flat potential at boundary
                pass

            if self.minx[0] <= center <= sum(self.minx[0], chi1[0], self.cv_type[0]):
                center_out = self.minx[0] - diff(center, self.minx[0], self.cv_type[0])
                val = diff(xi[0], center_out, self.cv_type[0])
                epot = w * np.exp(-(val * val) / (2.0 * self.hill_var[0]))
                local_pot += epot
                bias_force[0] -= epot * val / self.hill_var[0]
            else:
                # TODO: scale height of hills to ensure flat potential at boundary
                pass

        return local_pot, bias_force

    def write_restart(self, filename: str = "restart_wtm"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            pmf=self.pmf,
            force=self.bias,
            metapot=self.metapot,
            centers=self.center,
        )

    def restart(self, filename: str = "restart_wtm"):
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
        self.bias = data["force"]
        self.metapot = data["metapot"]
        self.center = data["centers"].tolist()

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}.npz!")

    def write_traj(self):
        """save trajectory for post-processing"""
        data = {
            "Epot [H]": self.epot,
            "T [K]": self.temp,
        }
        self._write_traj(data)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
