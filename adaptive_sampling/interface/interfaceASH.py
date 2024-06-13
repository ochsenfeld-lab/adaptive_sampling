#!/usr/bin/env python
import os, time
import numpy as np
import random

from adaptive_sampling import units

try:
    import ash
except ImportError:
    import sys as _sys
    print(">>> adaptive-sampling: Module 'ash' not found, will not import 'InterfaceASH'", file=_sys.stderr)
    del _sys


class AshMD:
    """Class for biased MD using the ASH program 

    The ASH program: https://github.com/RagnarB83/ash

    Args:
        fragment: Fragment object from ASH
        calculator: Calculator object from ASH
        dt: timestep in fs
        thermostat: if True apply langevin thermostat
        target_temp: target temperature in Kelvin
        friction: friction constant for langevin thermostat in 1/fs
        seed: random seed for MD
        active_atoms: list of active atoms, if None, all atoms are active
        mute: mute Calculator output
        scratch_dir: directory where `calculator` input and output files are stored
    """
    def __init__(
        self,
        fragment:     object=None,
        calculator:   object=None,
        dt:           float=0.5e0,
        thermostat:   bool=True,
        target_temp:  float=298.15e0,
        friction:     float=1.0e-3,
        seed:         int=42,
        active_atoms: list=[],
        mute:         bool=True,
        scratch_dir:  str="scratch/",
    ):
        if fragment == None:
            raise ValueError(' >>> ERROR: AshMD needs fragment object!')
        if calculator == None:
            raise ValueError(' >>> ERROR: AshMD needs calculator object!')

        self.step = -1 # first iteration is 0th step
        self.molecule = fragment
        self.calculator = calculator
        self.mute = mute
        self.dt = dt
        self.dt_atomic = dt / units.atomic_to_fs # timestep in atomic units
        self.langevin = thermostat
        self.target_temp = target_temp
        self.friction = friction

        # molecule
        self.coords   = np.asarray(self.molecule.coords).flatten() / units.BOHR_to_ANGSTROM
        self.charge   = self.molecule.charge
        self.mult     = self.molecule.mult
        self.mass     = np.asarray(self.molecule.masses)
        self.masses   = np.repeat(self.mass, 3)
        self.natoms   = int(len(self.coords) / 3)
        self.ndegrees = 3.e0*self.natoms-6.e0
        self.forces   = np.zeros_like(self.coords)
        self.momenta  = np.zeros_like(self.coords)
        self.bias_forces = np.zeros_like(self.forces)
        self.biaspots = []

        # active and frozen atoms
        if len(active_atoms):
            self.active_atoms = np.asarray(active_atoms)
        else:
            self.active_atoms = np.arange(self.natoms)
        self.n_active = len(self.active_atoms)
        self.frozen_atoms = np.delete(np.arange(self.natoms), self.active_atoms)

        # Random Number Generator
        self.seed_in = seed    
        if type(seed) is int:
            random.seed(seed)
            print(" >>> INFO: The random number seed for AshMD was: %i" % (seed))
        else:
            try:
                random.setstate(seed)
            except:
                raise ValueError("The provided seed was neither an int nor a state of random")

        # md values
        self.epot   = 0.e0
        self.ekin   = 0.e0
        self.temp   = 0.e0
        self.vol    = 0.e0
        self.pres   = 0.e0
        self.dcd    = None
        self.time_per_step = []

        # self.calc() is run in `scratch_dir` to redirect input and output files of the `calculator`
        self.cwd = os.getcwd()
        self.scratch_dir = scratch_dir
        if not os.path.isdir(self.scratch_dir):
            os.mkdir(self.scratch_dir)
            

    def calc_init(
        self,
        init_momenta: str="random", 
        biaspots: list=[],
        restart_file: str=None, 
        init_temp: float=298.15e0,
    ):
        """Initial calculation of energy, forces, momenta

        Args:
            init_momenta: zero / random / read,
            biaspots: list of bias potentials from `adaptive_sampling.sampling_tools`
            restart_file: filename for restart file, if init_momenta='read'
            init_temp: initial temperature, if init_momenta='random'
        """
        self.calc()

        if hasattr(biaspots, "__len__"):
            self.biaspots = biaspots
        else:
            self.biaspots = [biaspots]

        # Init momenta
        if init_momenta.lower() == "zero":
            self.momenta = np.zeros(3*self.natoms)

        elif init_momenta.lower() == "random":
            self.momenta = np.zeros(3*self.natoms)
            for i in self.active_atoms:
                self.momenta[i*3+0] = random.gauss(0.0,1.0) * np.sqrt(init_temp*self.mass[i])
                self.momenta[i*3+1] = random.gauss(0.0,1.0) * np.sqrt(init_temp*self.mass[i])
                self.momenta[i*3+2] = random.gauss(0.0,1.0) * np.sqrt(init_temp*self.mass[i])

            TTT = (np.power(self.momenta, 2)/self.masses).sum()
            TTT /= (3.e0*self.n_active)
            self.momenta *= np.sqrt(init_temp/(TTT*units.atomic_to_K))

        elif init_momenta.lower() == "read":
            try:
                restart = np.load(restart_file, allow_pickle=True)
            except:
                raise ValueError(" >>> AshMD: Could not find restart_file!")
            self.momenta      = restart["momenta"]
            self.coords       = restart["coords"]
            self.active_atoms = restart["active"]
            self.n_active     = len(self.active_atoms)
            self.frozen_atoms = np.delete(np.arange(self.natoms), self.active_atoms)

            print(f' >>> AshMD: Restarted MD from {restart_file}!')

        else:
            print(' >>> AshMD: Illegal selection of init_momenta!')
       
        self.calc_etvp()


    def run(
        self, 
        nsteps: int=0, 
        out_freq: int=None,
        restart_freq: int=None,
        dcd_freq: int=None,
        remove_rotation: bool=False,
        prefix: str="ashMD",
        **kwargs,
    ):
        """Run MD simulation using an Verlocity Verlete integrator and langevin thermostat
        
        Args:
            nsteps: number of MD steps
            out_freq: frequency of writing outputs
            restart_freq: frequncy of writing restart file
            dcd_freq: frequency of writing coords to DCD trajectory
            remove_rotation: if True, remove center of mass translation and rotation
            prefix: prefix for output files
        """
        for step in range(nsteps):
            start_time = time.perf_counter()
            self.step += 1
            
            self.propagate()
            self.calc()
            
            if bool(self.biaspots):
                self.bias_forces = np.zeros_like(self.forces)
                for bias in self.biaspots:
                    self.bias_forces += bias.step_bias(**kwargs)
                self.forces += self.bias_forces

            if self.n_active != self.natoms:
                self.freeze_atoms()

            if remove_rotation:
                self.rem_com_rot()

            self.up_momenta()
            self.calc_etvp()

            if restart_freq!=None and step%restart_freq == 0:
                self.write_restart(prefix=prefix)
            
            if dcd_freq!=None and step%dcd_freq == 0:
                self.print_dcd(prefix=prefix)

            self.time_per_step.append(time.perf_counter() - start_time)
            if out_freq != None and step%out_freq == 0:
                self.print_energy(prefix=prefix)


    def calc(self):
        """ Calculation of energy, forces 
        Excecuted in `self.scratch_dir` to avoid crowding the input directory with input and output files of `calculator`
        """
        os.chdir(self.scratch_dir)
        self.molecule.replace_coords(self.molecule.elems, self.coords.reshape((self.natoms,3)) * units.BOHR_to_ANGSTROM)
        results = ash.Singlepoint(
            theory=self.calculator, 
            fragment=self.molecule, 
            charge=self.charge, 
            mult=self.mult, 
            Grad=True
        )
        self.forces = results.gradient.flatten() 
        self.epot = results.energy
        os.chdir(self.cwd)


    def calc_etvp(self):
        """ Calculation of kinetic energy, temperature, volume, and pressure """
        # Ekin
        self.ekin = (np.power(self.momenta, 2)/self.masses).sum()
        self.ekin /= 2.0

        # Temperature
        self.temp  = self.ekin*2.e0/(3.e0*self.n_active*units.kB_in_atomic)

        # Volume
        r0 = np.sqrt(5.e0/3.e0) * np.sqrt(np.sum(self.masses*np.power(self.coords, 2))/np.sum(self.mass))
        self.vol = 4.e0 * np.pi * np.power(r0,3) / 3.e0 # bohr ** 3

        # Pressure
        self.pres  = np.sum(self.momenta*self.momenta/(3.e0*self.masses)) # a.u.
        self.pres -= np.sum(self.coords*(self.forces-self.bias_forces)/(3.e0)) # bohr * a.u./bohr
        self.pres *= units.atomic_to_bar / self.vol


    def propagate(
        self, 
    ):
        """Propagate momenta/coords with Velocity Verlet

        Args:
           langevin: switch on temperature control using a langevin themostate
           friction: friction constant for the langevin thermostat 
        """
        
        if self.langevin:
            prefac    = 2.0 / (2.0 + self.friction*self.dt)
            rand_push = np.sqrt(self.target_temp*self.friction*self.dt*units.kB_in_atomic/2.0e0)
            self.rand_gauss= np.zeros(shape=(len(self.momenta),), dtype=float)
            for atom in self.active_atoms:
                self.rand_gauss[3*atom]   = random.gauss(0, 1)
                self.rand_gauss[3*atom+1] = random.gauss(0, 1)
                self.rand_gauss[3*atom+2] = random.gauss(0, 1)

            self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces
            self.coords  += prefac * self.dt_atomic * self.momenta / self.masses

        else:
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces
            self.coords  += self.dt_atomic * self.momenta / self.masses


    def up_momenta(
        self, 
    ):
        """ Update momenta with Velocity Verlet """

        if self.langevin==True:
            prefac = (2.0e0 - self.friction*self.dt)/(2.0e0 + self.friction*self.dt)
            rand_push = np.sqrt(self.target_temp*self.friction*self.dt*units.kB_in_atomic/2.0e0)
            self.momenta *= prefac
            self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces

        else:
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces


    def rescale_mom(self, temperature: float=None):
        """Rescales momenta to a certain temperature

        Args:
            temperature: temperature E_kin will be adjusted to
        """
        if temperature == None:
            return
        self.calc_etvp()
        self.momenta *= np.sqrt(temperature/self.temp)
      

    def freeze_atoms(self):
        """ Freeze self.frozen_atoms """
        self.momenta[3*self.frozen_atoms+0] = 0.e0
        self.momenta[3*self.frozen_atoms+1] = 0.e0
        self.momenta[3*self.frozen_atoms+2] = 0.e0
        self.forces[3*self.frozen_atoms+0] = 0.e0
        self.forces[3*self.frozen_atoms+1] = 0.e0
        self.forces[3*self.frozen_atoms+2] = 0.e0


    def rem_com_rot(self):
        """Remove center of mass rotation"""
        I_xx = 0.e0
        I_yy = 0.e0
        I_zz = 0.e0
        I_xy = 0.e0
        I_xz = 0.e0
        I_yz = 0.e0
        tot_ang_mom = np.array([0.e0,0.e0,0.e0])

        for i in range(self.natoms):
            c_x = self.coords[3*i+0]
            c_y = self.coords[3*i+1]
            c_z = self.coords[3*i+2]
            I_xx += self.mass[i] * (c_y*c_y+c_z*c_z)
            I_yy += self.mass[i] * (c_x*c_x+c_z*c_z)
            I_zz += self.mass[i] * (c_x*c_x+c_y*c_y)
            I_xy -= self.mass[i] * (c_x*c_y)
            I_xz -= self.mass[i] * (c_x*c_z)
            I_yz -= self.mass[i] * (c_y*c_z)
            tot_ang_mom[0] += c_y * self.momenta[3*i+2] - c_z * self.momenta[3*i+1]
            tot_ang_mom[1] += c_z * self.momenta[3*i+0] - c_x * self.momenta[3*i+2]
            tot_ang_mom[2] += c_x * self.momenta[3*i+1] - c_y * self.momenta[3*i+0]

        itens = np.array([[I_xx,I_xy,I_xz],[I_xy,I_yy,I_yz],[I_xz,I_yz,I_zz]])
        if np.linalg.matrix_rank(itens)!=3:
            return
        itens_inv = np.linalg.inv(itens)
        ang_vel = np.array([0.e0,0.e0,0.e0])
    
        for i in range(3):
            ang_vel[i] = itens_inv[i][0] * tot_ang_mom[0] + itens_inv[i][1] * tot_ang_mom[1] + itens_inv[i][2] * tot_ang_mom[2]
    
        v_ext = np.array([0.e0,0.e0,0.e0])

        for i in range(self.natoms):
            c_x = self.coords[3*i+0]
            c_y = self.coords[3*i+1]
            c_z = self.coords[3*i+2]
    
            v_ext[0] = ang_vel[1] * c_z - ang_vel[2] * c_y
            v_ext[1] = ang_vel[2] * c_x - ang_vel[0] * c_z
            v_ext[2] = ang_vel[0] * c_y - ang_vel[1] * c_x
    
            self.momenta[i*3+0] -= v_ext[0] * self.mass[i]
            self.momenta[i*3+1] -= v_ext[1] * self.mass[i]
            self.momenta[i*3+2] -= v_ext[2] * self.mass[i]


    def print_energy(self, prefix: str="ashMD"):
        """Print/add energy to file 
      
        Args:
            filename: name of output file
        """
        if self.step == 0:
            with open(f"{prefix}.out", "w") as f:
                f.write("# TimeStep[fs]  PotentialEnergy[Eh]    KineticEnergy[Eh]      TotalEnergy[Eh]      Temperature [K]    Wall time [s]      \n")

        wall_time = sum(self.time_per_step)
        with open(f"{prefix}.out", "a") as f:
            f.write(str("%14.4e %20.10e %20.10e %20.10e %20.10e %20.10e \n") % (
                self.step*self.dt,self.epot,self.ekin,self.epot+self.ekin, self.temp, wall_time)
            )
        self.time_per_step = []
    
    def write_restart(self, prefix: str="ashMD", write_xyz: bool=True, refpdb: str=None):
        '''Prints all necessary files for a restart
      
        Args:
           prefix (str): prefix for restart files
           refpdb (str): name of a reference pdb of the system
            
        Returns:
           -
        '''
        restart_data = {
          "momenta": self.momenta,
          "coords": self.coords,
          "step": self.step,
          "active": self.active_atoms,
          "seed": self.seed_in,
        }
        np.savez(prefix+"_restart", **restart_data)

        if write_xyz:
            with open(prefix+"_restart_geom.xyz", "w+") as f:
                f.write(str("%i\nTIME: %14.7f\n") % (self.natoms,self.step*self.dt))
                
                for i in range(self.natoms):
                    string = str("%s %20.10e %20.10e %20.10e\n") % (
                        self.molecule.elems[i],self.coords[3*i+0]*units.BOHR_to_ANGSTROM,self.coords[3*i+1]*units.BOHR_to_ANGSTROM,self.coords[3*i+2]*units.BOHR_to_ANGSTROM
                    )
                    f.write(string)
        
        if refpdb != None:
            try:
                import mdtraj
            except ImportError as e:
                raise ImportError(" >>> AshMD needs `mdtraj` to write pdb file") from e

            topology = mdtraj.load(refpdb).topology
            pdbout = mdtraj.formats.PDBTrajectoryFile(prefix+"_restart_geom.pdb", mode='w', force_overwrite=True)
            pdbout.write(units.BOHR_to_ANGSTROM*self.coords.reshape(self.natoms, 3), topology)
            pdbout.close()


    def print_dcd(self, prefix: str="ashMD"):
        """saves coordinates to binary dcd format
      
        Args:
            filename: name of dcd file
        """  
        import mdtraj
        if self.dcd == None:
            self.dcd = mdtraj.formats.DCDTrajectoryFile(f"{prefix}_traj.dcd", 'w', force_overwrite=True)
        self.dcd.write(units.BOHR_to_ANGSTROM * self.coords.reshape(self.natoms, 3))


    def get_sampling_data(self):
        """interface to adaptive sampling algorithms. see: https://github.com/ahulm/adaptive_sampling"""
        try:
            from adaptive_sampling.interface.sampling_data import SamplingData
            return SamplingData(self.mass, 
                                self.coords, 
                                self.forces, 
                                self.epot, 
                                self.temp, 
                                self.natoms, 
                                self.step, 
                                self.dt)
        
        except ImportError as e:
            raise NotImplementedError(" >>> AshMD: `get_sampling_data()` is missing `adaptive_sampling` package") from e

