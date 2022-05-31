import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import diff
from ..units import *


class WTM(EnhancedSampling):
    """Well-Tempered Metadynamics
       
       see: Barducci et. al., Phys. rev. lett. (2008); https://doi.org/10.1103/PhysRevLett.100.020603

    An repulsive biasing potential is built by a superposition of Gaussian hills along the reaction coordinate.

    Args:
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CV (can be Bohr, Degree, or None)
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if None, hills are not scaled down (normal metadynamics)
        force_from_grid: forces are accumulated on grid for performance, 
                         if False, forces are calculated from sum of Gaussians in every step 
    """
    def __init__(
        self,
        hill_height: float,
        hill_std: list,
        *args,
        hill_drop_freq: int = 20,
        well_tempered_temp: float = 3000.0,
        force_from_grid: bool = True,
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
        self.update_freq = int(hill_drop_freq)
        self.well_tempered_temp = well_tempered_temp
        self.well_tempered = False if well_tempered_temp == None else True
        self.force_from_grid = force_from_grid

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

        bias_force += self.harmonic_walls(xi, delta_xi) #, self.hill_std)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

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
        self.pmf = -self.metapot * atomic_to_kJmol
        if self.well_tempered:
            self.pmf *= (
                self.equil_temp + self.well_tempered_temp
            ) / self.well_tempered_temp
        self.pmf -= self.pmf.min()

    def shared_bias(self):
        """TODO"""
        pass

    def get_wtm_force(self, xi: np.ndarray) -> list:
        """compute well-tempered metadynamics bias force from superpossiosion of gaussian hills

        Args:
            xi: state of collective variable

        Returns:
            bias_force: bias force from metadynamics
        """
        if self.the_md.step % self.update_freq == 0:
            self.center.append(xi[0]) if self.ncoords == 1 else self.center.append(xi)
            self._smooth_boundary(xi)

        is_in_bounds = (xi <= self.maxx).all() and (xi >= self.minx).all() 
        is_near_bounds = (
            xi - self.minx <= 3.0 * self.hill_std).any() or (
            self.maxx - xi <= 3.0 * self.hill_std).any()

        if is_in_bounds:
            bias_force = self._accumulate_wtm_force(xi)
            
        if not is_in_bounds or not self.force_from_grid or is_near_bounds:
            bias_force = self._analytic_wtm_force(xi)

        return bias_force

    def _accumulate_wtm_force(self, xi: np.ndarray) -> list:
        """accumulate metadynamics potential for free energy reweighting and its gradient on grid
        
        Args:
            xi: state of collective variable

        Returns:
            bias_force: bias force from metadynamics 
        """
        bink = self.get_index(xi)
        if self.the_md.step % self.update_freq == 0:
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

            else:
                # TODO: implement for 2D
                pass

        return [self.bias[i][bink[1], bink[0]] for i in range(self.ncoords)]

    def _analytic_wtm_force(self, xi: np.ndarray) -> list:
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
                
            dx = np.ma.array(
                diff(xi[0], np.asarray(self.center), self.cv_type[0])
            )
            dx[abs(dx) > 3 * self.hill_std[0]] = np.ma.masked

            # can get solw in long run, so only iterate over significant elements
            for val in np.nditer(dx.compressed(), flags=["zerosize_ok"]):
                    
                if self.well_tempered:
                    w = self.hill_height * np.exp(
                        -local_pot / (kB_in_atomic * self.well_tempered_temp)
                    )
                else:
                    w = self.hill_height

                epot = w * np.exp(-(val * val) / (2.0 * self.hill_var[0]))
                local_pot += epot
                bias_force[0] -= epot * val / self.hill_var[0]

        else:
            # TODO: implement for 2D
            pass
        
        return bias_force

    def _smooth_boundary(self, xi: np.ndarray):
        """smooth MtD potential at boundary by adding Gaussians outside of range(minx,maxx)

        Args:
            xi: collective variable
        """
        dminx = xi - self.minx  # > 0 if in bounds
        dmaxx = self.maxx - xi  # > 0 if in bounds

        if (dminx <= 3.0 * self.hill_std).all():
            self.center.append(self.minx[0]-dminx[0]) if self.ncoords == 1 else self.center.append(self.minx-dminx)
        elif (dmaxx <= 3.0 * self.hill_std).all():
            self.center.append(self.maxx[0]+dminx[0]) if self.ncoords == 1 else self.center.append(self.minx+dminx)
        elif (dminx <= 3.0 * self.hill_std).any():
            # TODO: implement for 2D
            pass
        elif (dmaxx <= 3.0 * self.hill_std).any():
            # TODO: implement for 2D
            pass

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
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

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
