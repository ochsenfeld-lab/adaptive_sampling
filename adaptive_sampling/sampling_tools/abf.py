import os, time, itertools
import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var, combine_welford_stats
from ..processing_tools.thermodynamic_integration import integrate
from ..units import *


class ABF(EnhancedSampling):
    """Adaptive Biasing Force Method

       see: Comer et. al., J. Phys. Chem. B (2015); https://doi.org/10.1021/jp506633n

    Args:
        md: Object of the MD Interface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        nfull: Number of force samples per bin where full bias is applied,
               if nsamples < nfull the bias force is scaled down by nsamples/nfull
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
            force_sample = [0 for _ in range(self.ncoords)]
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
                force_sample[i] = np.dot(
                    md_state.forces, v_i
                ) - kB_in_atomic * self.equil_temp * self._divergence_xi(
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
                    force_sample[i],
                )

                bias_force -= ramp * self.bias[i][bink[1], bink[0]] * delta_xi[i]
            
            if self.shared:
                self.shared_bias(list(itertools.chain(*[bink, force_sample])), **kwargs)
        
        else:
            bias_force += self.harmonic_walls(xi, delta_xi)

        self.traj = np.append(self.traj, [xi], axis=0)
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

    def shared_bias(
        self, 
        force_sample,
        sync_interval: int=5,
        mw_file: str="shared_bias",
        n_trials: int=10,
    ):
        """Syncs ABF bias with buffer file. Excecuted if multiple_walker=True
        
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
                print(" >>> Info: Creating a new instance for shared-bias ABF.")
                print(" >>> Info: Data of local walker stored in `restart_abf_local.npz`.")
            
            # create seperate restart file with local data only
            self._write_restart(
                filename="restart_abf_local",
                hist=self.histogram,
                pmf=self.pmf,
                force=self.bias,
                var=self.var_force,
                m2=self.m2_force,
            )
            
            self.update_samples = np.zeros(shape=(sync_interval, len(force_sample)))
            self.update_samples[0] = force_sample
            
            if not os.path.isfile(mw_file+".npz"):
                if self.verbose:
                    print(f" >>> Info: Creating buffer file for shared-bias ABF: `{mw_file}.npz`.")
                self._write_restart(
                    filename=mw_file,
                    hist=self.histogram,
                    pmf=self.pmf,
                    force=self.bias,
                    var=self.var_force,
                    m2=self.m2_force,
                )
                os.chmod(mw_file + ".npz", 0o444)
            elif self.verbose:
                print(f" >>> Info: Syncing with existing buffer file for shared-bias ABF: `{mw_file}.npz`.")
        
        count = md_state.step % sync_interval
        self.update_samples[count] = force_sample
            
        if count == sync_interval-1:
                
            hist = np.zeros_like(self.histogram)
            m2 = np.zeros_like(self.m2_force)
            var = np.zeros_like(self.var_force)
            bias = np.zeros_like(self.bias)

            for sample in self.update_samples:
                bink = (int(sample[1]), int(sample[0]))
                hist[bink] += 1
                for i in range(self.ncoords):
                    (
                        bias[i][bink],
                        m2[i][bink],
                        var[i][bink],
                    ) = welford_var(
                        hist[bink],
                        bias[i][bink],
                        m2[i][bink],
                        sample[2+i],
                    )
                
            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):
                        
                    global_hist = np.zeros_like(self.histogram).flatten()
                    global_m2 = np.zeros_like(self.m2_force).flatten()
                    global_var = np.zeros_like(self.var_force).flatten()
                    global_bias = np.zeros_like(self.bias).flatten()
                    
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
                        
                    self._write_restart(
                        filename=mw_file,
                        hist=global_hist.reshape(self.histogram.shape),
                        pmf=self.pmf,
                        force=global_bias.reshape(self.bias.shape),
                        var=global_var.reshape(self.var_force.shape),
                        m2=global_m2.reshape(self.m2_force.shape),
                    )                        
                    self.restart(filename=mw_file)
                    os.chmod(mw_file + ".npz", 0o444)  # other walkers can access again
                    
                    self.get_pmf()  # get new global pmf

                    # write data of local walker
                    local_data = np.load("restart_abf_local.npz")  
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
                        
                        # overwriting global arrays with local data
                        global_hist[i] = hist_i
                        global_bias[i] = bias_i
                        global_m2[i] = m2_i
                        global_var[i] = var_i

                    # TODO: the local pmf is wrong (but can be recovered from the force)
                    self._write_restart(
                        filename="restart_abf_local",
                        hist=global_hist.reshape(self.histogram.shape),
                        pmf=np.zeros_like(self.pmf), 
                        force=global_bias.reshape(self.bias.shape),
                        var=global_var.reshape(self.var_force.shape),
                        m2=global_m2.reshape(self.m2_force.shape),
                    )     
                    break                     

                elif trial < n_trials:
                    if self.verbose:
                        print(f" >>> Warning: Retry to open shared buffer file after {trial} failed attempts.")
                    time.sleep(0.1)
                else:
                    raise Exception(f" >>> Fatal Fatal: Failed to sync bias with `{mw_file}.npz`.")                 


    def _divergence_xi(self, xi, cv):
        """Calculate divergence of collective variable"""
        if cv.lower() == "distance":
            div = 2.0 / xi
        elif cv.lower() == "angle":
            div = 1.0 / np.tan(xi)
        elif cv.lower() in ["torsion", "2d"]:
            div = 0.0
        else:
            raise NotImplementedError(f" >>> Fatal Error: ABF not implemented for `{cv}`.")

        return div

    def write_restart(self, filename: str = "restart_abf"):
        """write restart file

        Args:
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

        Args:
            filename: name of restart file
        """
        try:
            data = np.load(filename + ".npz")
        except:
            raise OSError(f" >>> Fatal Error: restart file `{filename}.npz` not found!")

        self.histogram = data["hist"]
        self.pmf = data["pmf"]
        self.bias = data["force"]
        self.var_force = data["var"]
        self.m2_force = data["m2"]

        if self.verbose:
            print(f" >>> Info: ABF restartet from `{filename}.npz`!")

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
