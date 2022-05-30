import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var
from ..processing_tools.thermodynamic_integration import integrate
from ..units import *


class ABF(EnhancedSampling):
    """Adaptive Biasing Force Method
       see: Comer et. al., J. Phys. Chem. B (2015); https://doi.org/10.1021/jp506633n

    args:
        nfull: Number of force samples per bin where full bias is applied, 
               if nsamples < nfull the bias force is scaled down by nsamples/nfull
        md: Object of the MD Interface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
    """
    def __init__(self, *args, nfull: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.nfull = nfull
        self.var_force = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.bias)

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()

        (xi, delta_xi) = self.get_cv(**kwargs)

        bias_force = np.zeros_like(md_state.forces)

        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):

                # linear ramp function
                ramp = (
                    1.0
                    if self.histogram[bink[1], bink[0]] > self.nfull
                    else self.histogram[bink[1], bink[0]] / self.nfull
                )

                # inverse gradient v_i
                delta_xi_n = np.linalg.norm(delta_xi[i])
                v_i = delta_xi[i] / (delta_xi_n * delta_xi_n)

                # apply bias force
                force_sample = np.dot(
                    md_state.forces, v_i
                ) - kB_in_atomic * self.equil_temp * self.divergence_xi(
                    xi[i], self.cv_type[i]
                )

                (
                    self.bias[i][bink[1], bink[0]],
                    self.m2_force[i][bink[1], bink[0]],
                    self.var_force[i][bink[1], bink[0]],
                ) = welford_var(
                    self.histogram[bink[1], bink[0]],
                    self.bias[i][bink[1], bink[0]],
                    self.m2_force[i][bink[1], bink[0]],
                    force_sample,
                )

                bias_force -= ramp * self.bias[i][bink[1], bink[0]] * delta_xi[i]

        else:
            bias_force += self.harmonic_walls(xi, delta_xi)

        self.traj = np.append(self.traj, [xi], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        # correction for kinetics
        if self.kinetics:
            self.kinetics(delta_xi)

        if md_state.step % self.out_freq == 0:
            # write output

            if write_traj:
                self.write_traj()

            if write_output:
                self.get_pmf()
                output = {"hist": self.histogram, "free energy": self.pmf}
                for i in range(self.ncoords):
                    output[f"bias force {i}"] = self.bias[i]
                    output[f"var force {i}"] = self.var_force[i]

                self.write_output(output, filename="abf.out")
                self.write_restart()

        return bias_force

    def get_pmf(self, method: str = "trapezoid"):

        if self.ncoords == 1:
            self.pmf[0, :], _ = integrate(
                self.bias[0][0], self.dx, equil_temp=self.equil_temp, method=method
            )
            self.pmf *= atomic_to_kJmol
            self.pmf -= self.pmf.min()

        elif self.verbose:
            print(" >>> Info: On-the-fly integration only available for 1D coordinates")

    def shared_bias(self):
        """TODO"""
        pass

    def divergence_xi(self, xi, cv):

        div = np.zeros(self.ncoords)
        if cv.lower() == "x":
            pass
        elif cv.lower() == "distance":
            div[i] += 2.0 / xi[i]
        elif cv.lower() == "angle":
            div[i] += 1.0 / np.tan(xi[i])
        elif cv.lower() in ["torsion", "2d"]:
            pass  # div = 0
        else:
            raise ValueError(f" >>> Error: ABF not implemented for {cv}")

        return div

    def write_restart(self, filename: str = "restart_abf"):
        """write restart file

        args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            pmf=self.pmf,
            force=self.bias,
            var=self.var_force,
            m2=self.m2_force,
        )

    def restart(self, filename: str = "restart_abf"):
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
        self.var_force = data["var"]
        self.m2_force = data["m2"]

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
