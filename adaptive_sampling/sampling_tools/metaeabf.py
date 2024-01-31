import os, time
import numpy as np
from .enhanced_sampling import EnhancedSampling
from .utils import welford_var, combine_welford_stats, diff
from .eabf import eABF
from .metadynamics import WTM
from ..units import *


class WTMeABF(eABF, WTM, EnhancedSampling):
    """Well-Tempered Metadynamics extended-system Adaptive Biasing Force method

       see: Fu et. al., J. Phys. Chem. Lett. (2018); https://doi.org/10.1021/acs.jpclett.8b01994

    The collective variable is coupled to a fictitious particle with an harmonic force.
    The dynamics of the fictitious particel is biased using a combination of ABF and Metadynamics.

    Args:
        ext_sigma: thermal width of coupling between collective and extended variable
        ext_mass: mass of extended variable in atomic units
        hill_height: height of Gaussian hills in kJ/mol
        hill_std: standard deviation of Gaussian hills in units of the CV (can be Bohr, Degree, or None)
        md: Object of the MD Inteface
        cv_def: definition of the Collective Variable (CV) (see adaptive_sampling.colvars)
                [["cv_type", [atom_indices], minimum, maximum, bin_width], [possible second dimension]]
        friction: friction coefficient for Lagevin dynamics of the extended-system
        seed_in: random seed for Langevin dynamics of extended-system
        nfull: Number of force samples per bin where full bias is applied,
               if nsamples < nfull the bias force is scaled down by nsamples/nfull
        hill_drop_freq: frequency of hill creation in steps
        well_tempered_temp: effective temperature for WTM, if None, hills are not scaled down (normal metadynamics)
        force_from_grid: forces are accumulated on grid for performance,
                         if False, forces are calculated from sum of Gaussians in every step
        equil_temp: equillibrium temperature of MD
        verbose: print verbose information
        kinetice: calculate necessary data to obtain kinetics of reaction
        f_conf: force constant for confinement of system to the range of interest in CV space
        output_freq: frequency in steps for writing outputs

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abf_forces = np.zeros_like(self.bias)

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

        md_state = self.the_md.get_sampling_data()
        (xi, delta_xi) = self.get_cv(**kwargs)

        if stabilize and len(self.traj)>0:
            self.stabilizer(xi, threshold=stabilizer_threshold)

        self._propagate()

        mtd_forces = self.get_wtm_force(self.ext_coords)
        bias_force = self._extended_dynamics(xi, delta_xi)  # , self.hill_std)
        force_sample = [0 for _ in range(2 * self.ncoords)]

        if self._check_boundaries(self.ext_coords):

            bin_la = self.get_index(self.ext_coords)
            self.ext_hist[bin_la[1], bin_la[0]] += 1

            for i in range(self.ncoords):

                # linear ramp function
                ramp = (
                    1.0
                    if self.ext_hist[bin_la[1], bin_la[0]] > self.nfull
                    else self.ext_hist[bin_la[1], bin_la[0]] / self.nfull
                )

                # apply bias force on extended variable
                force_sample[i] = self.ext_k[i] * diff(self.ext_coords[i], xi[i], self.cv_type[i])
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
                self.ext_forces -= (
                    ramp * self.abf_forces[i][bin_la[1], bin_la[0]] - mtd_forces[i]
                )

        # xi-conditioned accumulators for CZAR
        if self._check_boundaries(xi):
            bink = self.get_index(xi)
            self.histogram[bink[1], bink[0]] += 1

            for i in range(self.ncoords):
                force_sample[self.ncoords+i] = self.ext_k[i] * diff(
                    self.ext_coords[i], self.grid[i][bink[i]], self.cv_type[i]
                )
                self.correction_czar[i][bink[1], bink[0]] += force_sample[self.ncoords+i]

        # shared-bias eABF
        if self.shared:                        
            self.shared_bias(
                xi,
                force_sample, 
                **kwargs,
            )

        self.traj = np.append(self.traj, [xi], axis=0)
        self.ext_traj = np.append(self.ext_traj, [self.ext_coords], axis=0)
        self.temp.append(md_state.temp)
        self.epot.append(md_state.epot)
        self._up_momenta()

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
                for i in range(self.ncoords):
                    output[f"metaforce {i}"] = self.bias[i]
                    output[f"abf force {i}"] = self.abf_forces[i]
                    output[f"czar force {i}"] = self.czar_force[i]
                    # TODO: output variance of CZAR for error estimate
                    # output[f"var force {i}"] = self.var_force[i]
                output[f"metapot"] = self.metapot

                self.write_output(output, filename=output_file)
                self.write_restart(filename=restart_file)

        return bias_force

    def shared_bias(
        self, 
        xi,
        force_sample,
        sync_interval: int=50,
        mw_file: str="../shared_bias",
        local_file: str="restart_wtmeabf_local",
        n_trials: int=100,
    ):
        """syncs eABF bias with other walkers

        TODO: 2D collective variables

        Args:
            xi: state of the collective variable
            force_sample: force sample of current step
            sync_interval: number of steps between sychronisation
            mw_file: name of buffer file for shared-bias
            n_trials: number of attempts to access of buffer file before throwing an error
        """
        md_state = self.the_md.get_sampling_data()
        if md_state.step == 0:        
            if self.verbose:
                print(" >>> Info: Creating a new instance for shared-bias eABF.")
                print(f" >>> Info: Data of local walker stored in `{local_file}.npz`.")
            
            # create seperate restart file with local data only
            self._write_restart(
                filename=local_file,
                hist=self.histogram,
                force=self.bias,
                m2=self.m2_force,
                ext_hist=self.ext_hist,
                czar_corr=self.correction_czar,
                abf_force=self.abf_forces,
                center=self.center,
                metapot=self.metapot,
            )
            
            if sync_interval % self.hill_drop_freq != 0:
                raise ValueError(
                    " >>> Fatal Error: Sync interval for shared-bias WTM has to divisible through the frequency of hill creation!"
                )

            self.len_center_last_sync = len(self.center)
            self.update_samples = np.zeros(shape=(sync_interval, len(force_sample)))
            self.last_samples_xi = np.zeros(shape=(sync_interval, len(xi)))
            self.last_samples_la = np.zeros(shape=(sync_interval, len(self.ext_coords)))
            self.metapot_last_sync = np.copy(self.metapot)
            self.bias_last_sync = np.copy(self.bias)

            if not os.path.isfile(mw_file+".npz"):
                if self.verbose:
                    print(f" >>> Info: Creating buffer file for shared-bias WTM-eABF: `{mw_file}.npz`.")
                self._write_restart(
                    filename=mw_file,
                    hist=self.histogram,
                    force=self.bias,
                    m2=self.m2_force,
                    ext_hist=self.ext_hist,
                    czar_corr=self.correction_czar,
                    abf_force=self.abf_forces,
                    center=self.center,
                    metapot=self.metapot,
                )
                os.chmod(mw_file + ".npz", 0o444)
            elif self.verbose:
                print(f" >>> Info: Syncing with buffer file `{mw_file}.npz`.")
        
        # save new samples
        count = md_state.step % sync_interval
        self.update_samples[count] = force_sample
        self.last_samples_xi[count] = xi
        self.last_samples_la[count] = self.ext_coords

        if count == sync_interval-1:
            
            # calculate progress since last sync from new samples
            hist = np.zeros_like(self.histogram)
            m2 = np.zeros_like(self.m2_force)
            abf_forces = np.zeros_like(self.abf_forces)
            ext_hist = np.zeros_like(self.ext_hist)
            czar_corr = np.zeros_like(self.correction_czar)
            
            delta_bias = self.bias - self.bias_last_sync
            delta_metapot = self.metapot - self.metapot_last_sync
            new_center = self.center[self.len_center_last_sync:]            

            for i, sample in enumerate(self.update_samples):
                if self._check_boundaries(self.last_samples_la[i]):                   
                    bin_la = self.get_index(self.last_samples_la[i])
                    ext_hist[bin_la[1], bin_la[0]] += 1
                    for j in range(self.ncoords):
                        (
                            abf_forces[j][bin_la[1], bin_la[0]],
                            m2[j][bin_la[1], bin_la[0]],
                            _,
                        ) = welford_var(
                            ext_hist[bin_la[1], bin_la[0]],
                            abf_forces[j][bin_la[1], bin_la[0]],
                            m2[j][bin_la[1], bin_la[0]],
                            sample[j],
                        )
                
                if self._check_boundaries(self.last_samples_xi[i]):
                    bin_xi = self.get_index(self.last_samples_xi[i])
                    hist[bin_xi[1], bin_xi[0]] += 1 
                    for j in range(self.ncoords):
                        czar_corr[j][bin_xi[1], bin_xi[0]] += sample[self.ncoords+j]
            
            # add new samples to local restart
            self._update_wtmeabf(
                local_file,
                hist, 
                ext_hist,
                abf_forces,
                m2,
                czar_corr,
                delta_bias,
                delta_metapot,
                new_center,
            )

            trial = 0
            while trial < n_trials:
                trial += 1
                if not os.access(mw_file + ".npz", os.W_OK):
                    
                    # grant write access only to one walker during sync
                    os.chmod(mw_file + ".npz", 0o666) 
                    self._update_wtmeabf(
                        mw_file,
                        hist, 
                        ext_hist,
                        abf_forces,
                        m2,
                        czar_corr,
                        delta_bias,
                        delta_metapot,
                        new_center,
                    )
                    self.restart(filename=mw_file, restart_ext_sys=False)
                    os.chmod(mw_file + ".npz", 0o444)  # other walkers can access again

                    # recalculates `self.metapot` and `self.bias` to ensure convergence of WTM potential
                    self._update_metapot_from_centers() 
                    
                    self.get_pmf()  # get new global pmf
                    self.metapot_last_sync = np.copy(self.metapot)
                    self.bias_last_sync = np.copy(self.bias)
                    self.len_center_last_sync = len(self.center)
                    break                     

                elif trial < n_trials:
                    if self.verbose:
                        print(f" >>> Warning: Retry to open shared buffer file after {trial} failed attempts.")
                    time.sleep(0.1)
                else:
                    raise Exception(f" >>> Fatal Error: Failed to sync bias with `{mw_file}.npz`.")

    def _update_wtmeabf(
        self, 
        filename: str,
        hist: np.ndarray, 
        ext_hist: np.ndarray,
        abf_forces: np.ndarray,
        m2: np.ndarray,
        czar_corr: np.ndarray,
        delta_bias: np.ndarray,
        delta_metapot: np.ndarray,
        center: np.ndarray,
    ):
        with np.load(f"{filename}.npz") as data:

            new_hist = data["hist"] + hist      
            new_czar_corr = data["czar_corr"] + czar_corr
            new_metapot = data["metapot"] + delta_metapot
            new_bias = data["force"] + delta_bias
            
            new_m2 = np.zeros_like(self.m2_force).flatten()
            new_abf_forces = np.zeros_like(self.abf_forces).flatten()
            new_ext_hist = np.zeros_like(self.ext_hist).flatten()
            new_centers = np.append(data["center"], center)
            
            for i in range(len(hist.flatten())):
                (
                    new_ext_hist[i],
                    new_abf_forces[i],
                    new_m2[i],
                    _,
                ) = combine_welford_stats(
                    data["ext_hist"].flatten()[i], 
                    data["abf_force"].flatten()[i], 
                    data["m2"].flatten()[i], 
                    ext_hist.flatten()[i], 
                    abf_forces.flatten()[i], 
                    m2.flatten()[i], 
                )
                                    
        self._write_restart(
            filename=filename,
            hist=new_hist,
            force=new_bias,
            m2=new_m2.reshape(self.m2_force.shape),
            ext_hist=new_ext_hist.reshape(self.histogram.shape),
            czar_corr=new_czar_corr,
            abf_force=new_abf_forces.reshape(self.abf_forces.shape),
            center=new_centers,
            metapot=new_metapot,
        )                           

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
            force=self.bias,
            m2=self.m2_force,
            ext_hist=self.ext_hist,
            czar_corr=self.correction_czar,
            abf_force=self.abf_forces,
            center=self.center,
            metapot=self.metapot,
            ext_momenta=self.ext_momenta,
            ext_coords=self.ext_coords,
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
        self.bias = data["force"]
        self.m2_force = data["m2"]
        self.ext_hist = data["ext_hist"]
        self.correction_czar = data["czar_corr"]
        self.abf_forces = data["abf_force"]
        self.center = data["center"].tolist()
        self.metapot = data["metapot"]
        if restart_ext_sys:
            self.ext_momenta = data["ext_momenta"]
            self.ext_coords = data["ext_coords"]

        if self.verbose:
            print(f" >>> Info: Adaptive sampling restartet from {filename}!")

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
