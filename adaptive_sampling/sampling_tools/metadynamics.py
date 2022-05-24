import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import diff
from ..processing_tools.thermodynamic_integration import integrate


class MtD(EnhancedSampling):
    def __init__(
        self,
        hill_height: float,
        hill_std: list,
        *args,
        update_freq: int = 10,
        well_tempered_temp: float = 3000.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if int(update_freq) <= 0:
            raise ValueError(" >>> fatal error: Update interval has to be int > 0!")

        if hill_height <= 0:
            raise ValueError(" >>> fatal error: Gaussian height for MtD has to be > 0!")

        if well_tempered_temp is not None and well_tempered_temp <= 0:
            raise ValueError(
                " >>> fatal error: Effective temperature for Well-Tempered MtD has to be > 0!"
            )

        self.hill_height = hill_height / 2625.499639  # Hartree
        self.hill_std = self.unit_conversion_cv(np.asarray(hill_std))[0]
        self.hill_var = self.hill_std * self.hill_std
        self.update_freq = int(update_freq)
        self.well_tempered_temp = well_tempered_temp
        self.well_tempered = False if well_tempered_temp == None else True

        self.metapot = np.zeros_like(self.histogram)
        self.center = []

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)
        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        bias_force = np.zeros_like(md_state.forces)

        mtd_force = self.get_mtd_force(xi)
        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                bias_force += mtd_force[i] * delta_xi[i]

        bias_force += self.harmonic_walls(xi, delta_xi , self.hill_std)

        # correction for kinetics
        if self.kinetics:
            self.kinetics(delta_xi)

        if self.the_md.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj()

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"metapot {i}"] = self.metapot * 2625.499639  # kJ/mol

                self.write_output(output, filename="mtd.out")
                self.write_restart()

        return bias_force

    def get_pmf(self):
        self.pmf = -self.metapot * 2625.499639  # kJ/mol
        if self.well_tempered:
            self.pmf *= (
                self.equil_temp + self.well_tempered_temp
            ) / self.well_tempered_temp

    def shared_bias(self):
        """TODO"""
        pass

    def get_mtd_force(self, xi: np.ndarray) -> list:
        """compute metadynamics bias force from superpossiosion of gaussian hills

        args:
            xi: state of collective variable

        returns:
            bias: bias force
        """
        kB_a = 1.380648e-23 / 4.359744e-18
        if (xi <= self.maxx).all() and (xi >= self.minx).all():
            bink = self.get_index(xi)
            if self.the_md.step % self.update_freq == 0:
                if self.ncoords == 1:

                    self.center.append(xi[0])

                    if self.well_tempered:
                        w = self.hill_height * np.exp(
                            -self.metapot[bink[1], bink[0]]
                            / (kB_a * self.well_tempered_temp)
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

            bias = [self.bias[i][bink[1], bink[0]] for i in range(self.ncoords)]

        else:
            
            # compute bias from sum of gaussians if out of grid
            local_pot = 0.0
            bias = [0 for _ in range(self.ncoords)]
            if self.ncoords == 1:
                dx = np.ma.array(
                    diff(np.asarray(self.center[0]), xi[0], self.cv_type[0])
                )
                dx[abs(dx) > 3 * self.hill_std[0]] = np.ma.masked

                # can get solw in long run so only iterate over significant elements
                for val in np.nditer(dx.compressed(), flags=["zerosize_ok"]):
                    if self.well_tempered:
                        w = self.hill_height * np.exp(
                            -local_pot / (kB_a * self.well_tempered_temp)
                        )
                    else:
                        w = self.hill_height

                    epot = w * np.exp(-(val * val) / (2.0 * self.hill_var[0]))
                    local_pot += epot
                    bias[0] += epot * val / self.hill_var[0]

            else:
                # TODO: implement for 2D
                pass

        return bias

    def write_restart(self, filename: str = "restart_mtd"):
        """write restart file

        args:
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

    def restart(self, filename: str = "restart_mtd"):
        """restart from restart file

        args:
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
