import random, os, itertools, time
import numpy as np
from typing import Union
from .enhanced_sampling import EnhancedSampling
from .abf import ABF
from .utils import welford_var, combine_welford_stats, diff, cond_avg
from ..processing_tools.thermodynamic_integration import integrate
from ..units import *


class eABF(ABF, EnhancedSampling):
    """Extended-system adaptive biasing Force method

       see: Lesage et. al., J. Phys. Chem. B (2017); https://doi.org/10.1021/acs.jpcb.6b10055

    The collective variable is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particel is biased with the ABF algorithm.

    Args:
        ext_sigma: thermal width of coupling between collective and extended variable
        ext_mass: mass of extended variable in atomic units
        md: Object of the MDInterface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        nfull: Number of force samples per bin where full bias is applied,
               if nsamples < nfull the bias force is scaled down by nsamples/nfull
        friction: friction coefficient for Lagevin dynamics of the extended-system
        seed_in: random seed for Langevin dynamics of extended-system
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs
    """

    def __init__(
        self,
        ext_sigma: Union[float, list],
        ext_mass: Union[float, list],
        *args,
        friction: float = 1.0e-3,
        seed_in: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        ext_sigma = [ext_sigma] if not hasattr(ext_sigma, "__len__") else ext_sigma
        ext_mass = [ext_mass] if not hasattr(ext_mass, "__len__") else ext_mass

        (xi, _) = self.get_cv()

        # for dynamics of extended-system
        self.ext_sigma = self.unit_conversion_cv(np.asarray(ext_sigma))[0]
        self.ext_k = (kB_in_atomic * self.equil_temp) / (
            self.ext_sigma * self.ext_sigma
        )
        self.ext_mass = np.asarray(ext_mass)
        self.ext_hist = np.zeros_like(self.histogram)
        self.ext_forces = np.zeros(self.ncoords)
        self.ext_momenta = np.zeros(self.ncoords)
        self.ext_coords = np.copy(xi)
        self.correction_czar = np.zeros_like(self.bias)
        self.czar_force = np.zeros_like(self.bias)
        self.friction = friction
        self.ext_traj = np.copy(self.traj)

        # set random seed for langevin dynamics
        if type(seed_in) is int:
            random.seed(seed_in)
        else:
            try:
                random.setstate(seed_in)
            except:
                if self.verbose:
                    print(
                        "\n >>> Warning: The provided seed was neither an int nor a state of random!\n"
                    )

        # initialize extended system at target temp of MD simulation
        for i in range(self.ncoords):
            self.ext_momenta[i] = random.gauss(0.0, 1.0) * np.sqrt(
                self.equil_temp * self.ext_mass[i]
            )
            ttt = (np.power(self.ext_momenta, 2) / self.ext_mass).sum()
            ttt /= self.ncoords
            self.ext_momenta *= np.sqrt(self.equil_temp / (ttt * atomic_to_K))

    def step_bias(self, write_output: bool = True, write_traj: bool = True, **kwargs):

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)
        self._propagate()

        bias_force = self._extended_dynamics(xi, delta_xi)  # , self.ext_sigma)
        
        force_sample = [0 for _ in range(2 * self.ncoords)]

        bin_la = [-1, -1]
        if (self.ext_coords <= self.maxx).all() and (
            self.ext_coords >= self.minx
        ).all():

            bin_la = self.get_index(self.ext_coords)
            self.ext_hist[bin_la[1], bin_la[0]] += 1

            for i in range(self.ncoords):

                # linear ramp function
                ramp = (
                    1.0
                    if self.ext_hist[bin_la[1], bin_la[0]] > self.nfull
                    else self.ext_hist[bin_la[1], bin_la[0]] / self.nfull
                )

                # apply bias force on extended system
                force_sample[i] = self.ext_k[i] * diff(self.ext_coords[i], xi[i], self.cv_type[i])
                (
                    self.bias[i][bin_la[1], bin_la[0]],
                    self.m2_force[i][bin_la[1], bin_la[0]],
                    self.var_force[i][bin_la[1], bin_la[0]],
                ) = welford_var(
                    self.ext_hist[bin_la[1], bin_la[0]],
                    self.bias[i][bin_la[1], bin_la[0]],
                    self.m2_force[i][bin_la[1], bin_la[0]],
                    force_sample[i],
                )
                self.ext_forces -= ramp * self.bias[i][bin_la[1], bin_la[0]]

        # xi-conditioned accumulators for CZAR
        if (xi <= self.maxx).all() and (xi >= self.minx).all():

            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                force_sample[self.ncoords+i] = self.ext_k[i] * diff(
                    self.ext_coords[i], self.grid[i][bink[i]], self.cv_type[i]
                )
                self.correction_czar[i][bink[1], bink[0]] += force_sample[self.ncoords+i]

        if self.shared:            
            if not (self.ext_coords <= self.maxx).all() and (self.ext_coords >= self.minx).all():
                bin_la = [-1, -1]
            if not ((xi <= self.maxx).all() and (xi >= self.minx).all()):
                bink = [-1, -1]
            
            self.shared_bias(list(
                itertools.chain(*[
                    bin_la, 
                    bink, 
                    force_sample,
                ])), 
                **kwargs,
            )

        self._up_momenta()

        self.traj = np.append(self.traj, [xi], axis=0)
        self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)

        # correction for kinetics
        if self.kinetics:
            self._kinetics(delta_xi)

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
                    output[f"czar force {i}"] = self.czar_force[i]

                self.write_output(output, filename="eabf.out")
                self.write_restart()

        return bias_force

    def get_pmf(self, method: str = "trapezoid"):

        log_rho = np.log(
            self.histogram,
            out=np.zeros_like(self.histogram),
            where=(0 != self.histogram),
        )
        avg_force = cond_avg(self.correction_czar, self.histogram)

        if self.ncoords == 1:
            self.czar_force[0] = (
                -kB_in_atomic * self.equil_temp * np.gradient(log_rho[0], self.grid[0])
                + avg_force[0]
            )
            self.pmf[0, :], _ = integrate(
                self.czar_force[0][0],
                self.dx,
                equil_temp=self.equil_temp,
                method=method,
            )
            self.pmf *= atomic_to_kJmol
            self.pmf -= self.pmf.min()

        else:
            der_log_rho = np.gradient(log_rho, self.grid[1], self.grid[0])
            self.czar_force[0] = (
                -kB_in_atomic * self.equil_temp * der_log_rho[1] + avg_force[0]
            )
            self.czar_force[1] = (
                -kB_in_atomic * self.equil_temp * der_log_rho[0] + avg_force[1]
            )
            if self.verbose:
                print(
                    " >>> Info: On-the-fly integration only available for 1D coordinates"
                )

    def shared_bias(
        self, 
        force_sample,
        sync_interval: int=5,
        mw_file: str="shared_bias",
        n_trials: int=10,
    ):
        """syncs eABF bias with other walkers

        TODO: 2D collective variables

        Args:
            force_sample: force sample of current step
            sync_interval: number of steps between sychronisation
            mw_file: name of buffer file for shared-bias
            n_trials: number of attempts to access of buffer file before throwing an error
        """
        md_state = self.the_md.get_sampling_data()
        if md_state.step == 0:        
            if self.verbose:
                print(" >>> Info: Creating a new instance for shared-bias eABF.")
                print(" >>> Info: Data of local walker stored in `restart_eabf_local.npz`.")
            
            # create seperate restart file with local data only
            self._write_restart(
                filename="restart_eabf_local",
                hist=self.histogram,
                force=self.bias,
                var=self.var_force,
                m2=self.m2_force,
                ext_hist=self.ext_hist,
                czar_corr=self.correction_czar,
            )

            self.update_samples = np.zeros(shape=(sync_interval, len(force_sample)))
            
            if not os.path.isfile(mw_file+".npz"):
                if self.verbose:
                    print(f" >>> Info: Creating buffer file for shared-bias eABF: `{mw_file}.npz`.")
                self._write_restart(
                    filename=mw_file,
                    hist=self.histogram,
                    force=self.bias,
                    var=self.var_force,
                    m2=self.m2_force,
                    ext_hist=self.ext_hist,
                    czar_corr=self.correction_czar,
                )
                os.chmod(mw_file + ".npz", 0o444)
            elif self.verbose:
                print(f" >>> Info: Syncing with existing buffer file for shared-bias eABF: `{mw_file}.npz`.")
        
        count = md_state.step % sync_interval
        self.update_samples[count] = force_sample
            
        if count == sync_interval-1:
            
            hist = np.zeros_like(self.histogram)
            m2 = np.zeros_like(self.m2_force)
            var = np.zeros_like(self.var_force)
            bias = np.zeros_like(self.bias)
            ext_hist = np.zeros_like(self.ext_hist)
            czar_corr = np.zeros_like(self.correction_czar)

            for sample in self.update_samples:
                    
                bin_la = (int(sample[1]), int(sample[0]))
                bin_xi = (int(sample[3]), int(sample[2]))
                hist[bin_xi] += 1 if (np.asarray(bin_xi) >= 0).all() else 0
                ext_hist[bin_la] += 1 if (np.asarray(bin_la) >= 0).all() else 0
                for i in range(self.ncoords):
                    if (np.asarray(bin_xi) >= 0).all():
                        czar_corr[i][bin_xi] += sample[4+self.ncoords+i]
                    if (np.asarray(bin_la) >= 0).all():
                        (
                            bias[i][bin_la],
                            m2[i][bin_la],
                            var[i][bin_la],
                        ) = welford_var(
                            hist[bin_la],
                            bias[i][bin_la],
                            m2[i][bin_la],
                            sample[4+i],
                        )

            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):
                    
                    global_hist = np.zeros_like(self.histogram).flatten()
                    global_m2 = np.zeros_like(self.m2_force).flatten()
                    global_var = np.zeros_like(self.var_force).flatten()
                    global_bias = np.zeros_like(self.bias).flatten()
                    global_ext_hist = np.zeros_like(self.ext_hist).flatten()
                    global_czar_corr = np.zeros_like(self.correction_czar).flatten()

                    # grant write access only to one walker during sync
                    os.chmod(mw_file + ".npz", 0o666) 
                    shared_data = np.load(mw_file + ".npz")  

                    for i in range(len(hist.flatten())):
                        (
                            hist_i,
                            bias_i,
                            m2_i,
                            var_i,
                        ) = combine_welford_stats(
                            shared_data["hist"].flatten()[i], 
                            shared_data["force"].flatten()[i], 
                            shared_data["m2"].flatten()[i], 
                            hist.flatten()[i], 
                            bias.flatten()[i], 
                            m2.flatten()[i], 
                        )
                        global_hist[i] = hist_i
                        global_bias[i] = bias_i
                        global_m2[i] = m2_i
                        global_var[i] = var_i
                        global_ext_hist[i] = shared_data["ext_hist"].flatten()[i] + hist.flatten()[i]
                        global_czar_corr[i] = shared_data["czar_corr"].flatten()[i] + czar_corr.flatten()[i]
                        
                    self._write_restart(
                        filename=mw_file,
                        hist=global_hist.reshape(self.histogram.shape),
                        force=global_bias.reshape(self.bias.shape),
                        var=global_var.reshape(self.var_force.shape),
                        m2=global_m2.reshape(self.m2_force.shape),
                        ext_hist=global_ext_hist.reshape(self.histogram.shape),
                        czar_corr=global_czar_corr.reshape(self.correction_czar.shape),
                    )   

                    self.restart(filename=mw_file)
                    os.chmod(mw_file + ".npz", 0o444)  # other walkers can access again
                    
                    self.get_pmf()  # get new global pmf

                    # write data of local walker
                    local_data = np.load("restart_eabf_local.npz")  
                    for i in range(len(hist.flatten())):
                        (
                        hist_i,
                            bias_i,
                            m2_i,
                            var_i,
                        ) = combine_welford_stats(
                            local_data["hist"].flatten()[i], 
                            local_data["force"].flatten()[i], 
                            local_data["m2"].flatten()[i], 
                            hist.flatten()[i], 
                            bias.flatten()[i], 
                            m2.flatten()[i],
                        )
                            
                        # reusing arrays for global data
                        global_hist[i] = hist_i
                        global_bias[i] = bias_i
                        global_m2[i] = m2_i
                        global_var[i] = var_i
                        global_ext_hist[i] = local_data["ext_hist"].flatten()[i] + hist.flatten()[i]
                        global_czar_corr[i] = local_data["czar_corr"].flatten()[i] + czar_corr.flatten()[i]

                    self._write_restart(
                        filename="restart_eabf_local",
                        hist=global_hist.reshape(self.histogram.shape),
                        force=global_bias.reshape(self.bias.shape),
                        var=global_var.reshape(self.var_force.shape),
                        m2=global_m2.reshape(self.m2_force.shape),
                        ext_hist=global_ext_hist.reshape(self.histogram.shape),
                        czar_corr=global_czar_corr.reshape(self.correction_czar.shape),
                    )     
                    break                     

                elif trial < n_trials:
                    if self.verbose:
                        print(f" >>> Warning: Retry to open shared buffer file after {trial} failed attempts.")
                    time.sleep(0.1)
                else:
                    raise Exception(f" >>> Fatal Fatal: Failed to sync bias with `{mw_file}.npz`.")


    def _propagate(self, langevin: bool = True):
        """Propagate momenta/coords of extended variable in time with Velocity Verlet

        Args:
           langevin: Temperature control with langevin dynamics
        """
        if langevin:
            prefac = 2.0 / (2.0 + self.friction * self.the_md.dt)
            rand_push = np.sqrt(
                self.equil_temp * self.friction * self.the_md.dt * kB_in_atomic / 2.0e0
            )
            self.ext_rand_gauss = np.zeros(shape=(len(self.ext_momenta),), dtype=float)
            for atom in range(len(self.ext_rand_gauss)):
                self.ext_rand_gauss[atom] = random.gauss(0, 1)

            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
            self.ext_coords += (
                prefac * self.the_md.dt * self.ext_momenta / self.ext_mass
            )

        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
            self.ext_coords += self.the_md.dt * self.ext_momenta / self.ext_mass

    def _up_momenta(self, langevin: bool = True):
        """Update momenta of extended variables with Velocity Verlet

        Args:
            langevin: Temperature control with langevin dynamics
        """
        if langevin:
            prefac = (2.0e0 - self.friction * self.the_md.dt) / (
                2.0e0 + self.friction * self.the_md.dt
            )
            rand_push = np.sqrt(
                self.equil_temp * self.friction * self.the_md.dt * kB_in_atomic / 2.0e0
            )
            self.ext_momenta *= prefac
            self.ext_momenta += np.sqrt(self.ext_mass) * rand_push * self.ext_rand_gauss
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces
        else:
            self.ext_momenta -= 0.5e0 * self.the_md.dt * self.ext_forces

    def _extended_dynamics(
        self,
        xi: np.ndarray,
        delta_xi: np.ndarray,
        margin: np.ndarray = np.array([0.0, 0.0]),
    ) -> np.ndarray:
        """get forces due to coupling to extended system and confine extended variable to range of interest

        Args:
            xi: collective variable
            delta_xi: gradient of collective variable
            margin: inset from minx, maxx where bias starts

        Returns:
            bias_force:
        """
        bias_force = np.zeros_like(self.the_md.forces)

        for i in range(self.ncoords):
            # harmonic coupling of extended coordinate to reaction coordinate

            dxi = diff(self.ext_coords[i], xi[i], self.cv_type[i])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_force -= self.ext_k[i] * dxi * delta_xi[i]

            # harmonic walls for confinement to range of interest
            if self.ext_coords[i] > (self.maxx[i] - margin[i]):
                r = diff(self.maxx[i] - margin[i], self.ext_coords[i], self.cv_type[i])
                self.ext_forces[i] -= self.f_conf[i] * r

            elif self.ext_coords[i] < (self.minx[i] + margin[i]):
                r = diff(self.minx[i] + margin[i], self.ext_coords[i], self.cv_type[i])
                self.ext_forces[i] -= self.f_conf[i] * r

        return bias_force

    def write_restart(self, filename: str = "restart_abf"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            force=self.bias,
            var=self.var_force,
            m2=self.m2_force,
            ext_hist=self.ext_hist,
            czar_corr=self.correction_czar,
        )

    def restart(self, filename: str = "restart_abf"):
        """restart from restart file

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz")
        except:
            raise OSError(f" >>> fatal error: restart file {filename}.npz not found!")

        self.histogram = data["hist"]
        self.bias = data["force"]
        self.var_force = data["var"]
        self.m2_force = data["m2"]
        self.ext_hist = data["ext_hist"]
        self.correction_czar = data["czar_corr"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

    def _write_ext_traj(self):
        data = {}
        for i in range(self.ncoords):
            if self.cv_type[i] == "angle":
                self.ext_traj[:, i] *= DEGREES_per_RADIAN
            elif self.cv_type[i] == "distance":
                self.ext_traj[:, i] *= BOHR_to_ANGSTROM
            data[f"lambda{i}"] = self.ext_traj[:, i]
        return data

    def write_traj(self):
        """save trajectory for post-processing"""

        data = self._write_ext_traj()
        data["Epot [H]"] = self.epot
        data["T [K]"] = self.temp

        self._write_traj(data)

        # reset trajectories to save memory
        self.traj = np.array([self.traj[-1]])
        self.ext_traj = np.array([self.ext_traj[-1]])
        self.epot = [self.epot[-1]]
        self.temp = [self.temp[-1]]
