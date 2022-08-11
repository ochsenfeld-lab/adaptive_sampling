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

        if self._check_boundaries(xi):

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
        mw_file: str="../shared_bias",
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
                force=self.bias,
                m2=self.m2_force,
            )
            
            self.update_samples = np.zeros(shape=(sync_interval, len(force_sample)))
            
            if not os.path.isfile(mw_file+".npz"):
                if self.verbose:
                    print(f" >>> Info: Creating buffer file for shared-bias ABF: `{mw_file}.npz`.")
                self._write_restart(
                    filename=mw_file,
                    hist=self.histogram,
                    force=self.bias,
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
            bias = np.zeros_like(self.bias)

            for sample in self.update_samples:
                bink = (int(sample[1]), int(sample[0]))
                hist[bink] += 1
                for i in range(self.ncoords):
                    (
                        bias[i][bink],
                        m2[i][bink],
                        _,
                    ) = welford_var(
                        hist[bink],
                        bias[i][bink],
                        m2[i][bink],
                        sample[2+i],
                    )              
            # write data of local walker
            self._update_abf(
                "restart_abf_local", 
                hist, bias, m2,
            )             
            
            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):
                    
                    # grant write access only to local walker during sync
                    os.chmod(mw_file + ".npz", 0o666) 
                    self._update_abf(
                        mw_file, 
                        hist, bias, m2,
                    ) 
                    self.restart(filename=mw_file)
                    os.chmod(mw_file + ".npz", 0o444)                      
                    self.get_pmf()  
                    break                     

                elif trial < n_trials:
                    if self.verbose:
                        print(f" >>> Warning: Retry to open shared buffer file after {trial} failed attempts.")
                    time.sleep(0.1)
                else:
                    raise Exception(f" >>> Fatal Error: Failed to sync bias with `{mw_file}.npz`.")

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

    def _update_abf(
        self, 
        filename: str, 
        hist: np.ndarray, 
        bias: np.ndarray, 
        m2: np.ndarray,
    ):

        new_hist = np.zeros_like(self.histogram).flatten()
        new_m2 = np.zeros_like(self.m2_force).flatten()
        new_bias = np.zeros_like(self.bias).flatten() 

        with np.load(f"{filename}.npz") as local_data:
            for i in range(len(new_hist)):
                # overwriting global arrays with local data
                (
                    new_hist[i],
                    new_bias[i],
                    new_m2[i],
                    _,
                ) = combine_welford_stats(
                    local_data["hist"].flatten()[i], 
                    local_data["force"].flatten()[i], 
                    local_data["m2"].flatten()[i], 
                    hist.flatten()[i], 
                    bias.flatten()[i], 
                    m2.flatten()[i],
                )
                        
        self._write_restart(
            filename=filename,
            hist=new_hist.reshape(self.histogram.shape),
            force=new_bias.reshape(self.bias.shape),
            m2=new_m2.reshape(self.m2_force.shape),
        )

    def write_restart(self, filename: str = "restart_abf"):
        """write restart file

        Args:
            filename: name of restart file
        """
        self._write_restart(
            filename=filename,
            hist=self.histogram,
            force=self.bias,
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
        self.bias = data["force"]
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
