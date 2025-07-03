#!/usr/bin/env python
import os, time
import numpy as np
import random

from adaptive_sampling import units


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
        barostat: if True apply Monte Carlo barostat
        target_pressure: target pressure in bar
        barostat_freq: frequency of barostat updates
        barostat_reporter: if not None, write barostat data to file
        seed: random seed for MD
        active_atoms: list of active atoms, if None, all atoms are active
        mute: mute Calculator output
        scratch_dir: directory where `calculator` input and output files are stored
    """

    def __init__(
        self,
        fragment: object = None,
        calculator: object = None,
        dt: float = 0.5e0,
        thermostat: bool = True,
        target_temp: float = 298.15e0,
        friction: float = 1.0e-3,
        barostat: bool = False,
        target_pressure: float = 1.0,
        barostat_freq: int = 25,
        barostat_reporter: str = "barostat.log",
        seed: int = 42,
        active_atoms: list = [],
        mute: bool = True,
        scratch_dir: str = "scratch/",
        **kwargs,
    ):
        if fragment == None:
            raise ValueError(" >>> ERROR: AshMD needs fragment object!")
        if calculator == None:
            raise ValueError(" >>> ERROR: AshMD needs calculator object!")

        self.step = -1  # first iteration is 0th step
        self.molecule = fragment
        self.calculator = calculator
        self.mute = mute
        self.dt = dt
        self.dt_atomic = dt / units.atomic_to_fs  # timestep in atomic units
        self.langevin = thermostat
        self.target_temp = target_temp
        self.friction = friction

        # molecule
        self.coords = (
            np.asarray(self.molecule.coords).flatten() / units.BOHR_to_ANGSTROM
        )
        self.charge = self.molecule.charge
        self.mult = self.molecule.mult
        self.mass = np.asarray(self.molecule.masses)
        self.masses = np.repeat(self.mass, 3)
        self.natoms = int(len(self.coords) / 3)
        self.ndegrees = 3.0e0 * self.natoms - 6.0e0
        self.forces = np.zeros_like(self.coords)
        self.momenta = np.zeros_like(self.coords)
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
                raise ValueError(
                    "The provided seed was neither an int nor a state of random"
                )

        # md values
        self.epot = 0.0e0
        self.ekin = 0.0e0
        self.temp = 0.0e0
        self.vol = 0.0e0
        self.pres = 0.0e0
        self.dcd = None
        self.time_per_step = []

        # self.calc() is run in `scratch_dir` to redirect input and output files of the `calculator`
        self.cwd = os.getcwd()
        self.scratch_dir = scratch_dir
        if not os.path.isdir(self.scratch_dir):
            os.mkdir(self.scratch_dir)

        # barostat
        self.apply_barostat = barostat
        if self.apply_barostat:
            self.barostat = MonteCarloBarostatASH(
                the_md=self,
                frequency=barostat_freq,
                target_pressure=target_pressure,
                barostat_reporter=barostat_reporter,
                **kwargs,
            )


    def calc_init(
        self,
        init_momenta: str = "random",
        biaspots: list = [],
        restart_file: str = None,
        init_temp: float = 298.15e0,
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
            self.momenta = np.zeros(3 * self.natoms)

        elif init_momenta.lower() == "random":
            self.momenta = np.zeros(3 * self.natoms)
            for i in self.active_atoms:
                self.momenta[i * 3 + 0] = random.gauss(0.0, 1.0) * np.sqrt(
                    init_temp * self.mass[i]
                )
                self.momenta[i * 3 + 1] = random.gauss(0.0, 1.0) * np.sqrt(
                    init_temp * self.mass[i]
                )
                self.momenta[i * 3 + 2] = random.gauss(0.0, 1.0) * np.sqrt(
                    init_temp * self.mass[i]
                )

            TTT = (np.power(self.momenta, 2) / self.masses).sum()
            TTT /= 3.0e0 * self.n_active
            self.momenta *= np.sqrt(init_temp / (TTT * units.atomic_to_K))

        elif init_momenta.lower() == "read":
            try:
                restart = np.load(restart_file, allow_pickle=True)
            except:
                raise ValueError(" >>> AshMD: Could not find restart_file!")
            self.momenta = restart["momenta"]
            self.coords = restart["coords"]
            self.active_atoms = restart["active"]
            self.n_active = len(self.active_atoms)
            self.frozen_atoms = np.delete(np.arange(self.natoms), self.active_atoms)

            print(f" >>> AshMD: Restarted MD from {restart_file}!")

        else:
            print(" >>> AshMD: Illegal selection of init_momenta!")

        self.calc_etvp()

    def run(
        self,
        nsteps: int = 0,
        out_freq: int = None,
        restart_freq: int = None,
        dcd_freq: int = None,
        remove_rotation: bool = False,
        prefix: str = "ashMD_production",
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
        for _ in range(nsteps):
            start_time = time.perf_counter()
            self.step += 1

            if self.apply_barostat:
                self.barostat.updateState()

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

            if restart_freq != None and self.step % restart_freq == 0:
                self.write_restart(prefix=prefix)

            if dcd_freq != None and self.step % dcd_freq == 0:
                self.print_dcd(prefix=prefix)

            self.time_per_step.append(time.perf_counter() - start_time)
            if out_freq != None and self.step % out_freq == 0:
                self.print_energy(prefix=prefix)

    def heat(
        self,
        nsteps: int = 0,
        start_temp: float = 1.0,
        target_temp: float = 300.0,
        stages: int = 100,
        out_freq: int = None,
        restart_freq: int = None,
        dcd_freq: int = None,
        remove_rotation: bool = False,
        prefix: str = "ashMD_heating",
        **kwargs,
    ):
        """Heating of MD simulation from current temperature to target temperature
        The temperature is slowly increased by rescaling momenta to the new temperature over discrete stages

        Args:
            nsteps: number of MD steps
            target_temp: target temperature
            start_temp: initial temperature
            stages: number of equilibrium stages (number of times momenta a rescaled to new temperatures)
            out_freq: frequency of writing outputs
            restart_freq: frequncy of writing restart file
            dcd_freq: frequency of writing coords to DCD trajectory
            remove_rotation: if True, remove center of mass translation and rotation
            prefix: prefix for output files
        """
        self.rescale_mom(temperature=start_temp)
        temp_diff = target_temp - start_temp
        deltaT_per_stage = temp_diff / stages
        steps_per_stage = int(nsteps / stages)
        self.target_temp = start_temp
        for _ in range(stages):
            self.run(
                nsteps=steps_per_stage,
                out_freq=out_freq,
                restart_freq=restart_freq,
                dcd_freq=dcd_freq,
                remove_rotation=remove_rotation,
                prefix=prefix,
                **kwargs,
            )
            self.target_temp = self.target_temp + deltaT_per_stage
            self.rescale_mom(temperature=self.target_temp)

    def calc(self):
        """Calculation of energy, forces
        Excecuted in `self.scratch_dir` to avoid crowding the input directory with input and output files of `calculator`
        """
        from ash import Singlepoint

        os.chdir(self.scratch_dir)
        self.molecule.replace_coords(
            self.molecule.elems,
            self.coords.reshape((self.natoms, 3)) * units.BOHR_to_ANGSTROM,
        )
        results = Singlepoint(
            theory=self.calculator,
            fragment=self.molecule,
            charge=self.charge,
            mult=self.mult,
            Grad=True,
        )
        self.forces = results.gradient.flatten()
        self.epot = results.energy
        os.chdir(self.cwd)

    def calc_etvp(self):
        """Calculation of kinetic energy, temperature, volume, and pressure"""
        # Ekin
        self.ekin = (np.power(self.momenta, 2) / self.masses).sum() / 2.0

        # Temperature
        self.temp = self.ekin * 2.0e0 / (3.0e0 * self.n_active * units.kB_in_atomic)

        # Volume
        r0 = np.sqrt(5.0e0 / 3.0e0) * np.sqrt(
            np.sum(self.masses * np.power(self.coords, 2)) / np.sum(self.mass)
        )
        self.vol = 4.0e0 * np.pi * np.power(r0, 3) / 3.0e0  # bohr ** 3

        # Pressure
        self.pres = np.sum(self.momenta * self.momenta / (3.0e0 * self.masses))  # a.u.
        self.pres -= np.sum(
            self.coords * (self.forces - self.bias_forces) / (3.0e0)
        )  # bohr * a.u./bohr
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
            prefac = 2.0 / (2.0 + self.friction * self.dt)
            rand_push = np.sqrt(
                self.target_temp * self.friction * self.dt * units.kB_in_atomic / 2.0e0
            )
            self.rand_gauss = np.zeros(shape=(len(self.momenta),), dtype=float)
            for atom in self.active_atoms:
                self.rand_gauss[3 * atom] = random.gauss(0, 1)
                self.rand_gauss[3 * atom + 1] = random.gauss(0, 1)
                self.rand_gauss[3 * atom + 2] = random.gauss(0, 1)

            self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces
            self.coords += prefac * self.dt_atomic * self.momenta / self.masses

        else:
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces
            self.coords += self.dt_atomic * self.momenta / self.masses

    def up_momenta(
        self,
    ):
        """Update momenta with Velocity Verlet"""

        if self.langevin:
            prefac = (2.0e0 - self.friction * self.dt) / (
                2.0e0 + self.friction * self.dt
            )
            rand_push = np.sqrt(
                self.target_temp * self.friction * self.dt * units.kB_in_atomic / 2.0e0
            )
            self.momenta *= prefac
            self.momenta += np.sqrt(self.masses) * rand_push * self.rand_gauss
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces

        else:
            self.momenta -= 0.5e0 * self.dt_atomic * self.forces

    def rescale_mom(self, temperature: float = None):
        """Rescales momenta to a certain temperature

        Args:
            temperature: temperature E_kin will be adjusted to
        """
        if temperature == None:
            return
        temp = (np.power(self.momenta, 2) / self.masses).sum() / (
            3.0e0 * self.n_active * units.kB_in_atomic
        )
        self.momenta *= np.sqrt(temperature / temp)

    def freeze_atoms(self):
        """Freeze self.frozen_atoms"""
        self.momenta[3 * self.frozen_atoms + 0] = 0.0e0
        self.momenta[3 * self.frozen_atoms + 1] = 0.0e0
        self.momenta[3 * self.frozen_atoms + 2] = 0.0e0
        self.forces[3 * self.frozen_atoms + 0] = 0.0e0
        self.forces[3 * self.frozen_atoms + 1] = 0.0e0
        self.forces[3 * self.frozen_atoms + 2] = 0.0e0

    def rem_com_rot(self):
        """Remove center of mass rotation"""
        I_xx = 0.0e0
        I_yy = 0.0e0
        I_zz = 0.0e0
        I_xy = 0.0e0
        I_xz = 0.0e0
        I_yz = 0.0e0
        tot_ang_mom = np.array([0.0e0, 0.0e0, 0.0e0])

        for i in range(self.natoms):
            c_x = self.coords[3 * i + 0]
            c_y = self.coords[3 * i + 1]
            c_z = self.coords[3 * i + 2]
            I_xx += self.mass[i] * (c_y * c_y + c_z * c_z)
            I_yy += self.mass[i] * (c_x * c_x + c_z * c_z)
            I_zz += self.mass[i] * (c_x * c_x + c_y * c_y)
            I_xy -= self.mass[i] * (c_x * c_y)
            I_xz -= self.mass[i] * (c_x * c_z)
            I_yz -= self.mass[i] * (c_y * c_z)
            tot_ang_mom[0] += (
                c_y * self.momenta[3 * i + 2] - c_z * self.momenta[3 * i + 1]
            )
            tot_ang_mom[1] += (
                c_z * self.momenta[3 * i + 0] - c_x * self.momenta[3 * i + 2]
            )
            tot_ang_mom[2] += (
                c_x * self.momenta[3 * i + 1] - c_y * self.momenta[3 * i + 0]
            )

        itens = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])
        if np.linalg.matrix_rank(itens) != 3:
            return
        itens_inv = np.linalg.inv(itens)
        ang_vel = np.array([0.0e0, 0.0e0, 0.0e0])

        for i in range(3):
            ang_vel[i] = (
                itens_inv[i][0] * tot_ang_mom[0]
                + itens_inv[i][1] * tot_ang_mom[1]
                + itens_inv[i][2] * tot_ang_mom[2]
            )

        v_ext = np.array([0.0e0, 0.0e0, 0.0e0])

        for i in range(self.natoms):
            c_x = self.coords[3 * i + 0]
            c_y = self.coords[3 * i + 1]
            c_z = self.coords[3 * i + 2]

            v_ext[0] = ang_vel[1] * c_z - ang_vel[2] * c_y
            v_ext[1] = ang_vel[2] * c_x - ang_vel[0] * c_z
            v_ext[2] = ang_vel[0] * c_y - ang_vel[1] * c_x

            self.momenta[i * 3 + 0] -= v_ext[0] * self.mass[i]
            self.momenta[i * 3 + 1] -= v_ext[1] * self.mass[i]
            self.momenta[i * 3 + 2] -= v_ext[2] * self.mass[i]

    def print_energy(self, prefix: str = "ashMD"):
        """Print/add energy to file

        Args:
            filename: name of output file
        """
        if self.step == 0:
            with open(f"{prefix}.out", "w") as f:
                f.write(
                    "# TimeStep[fs]  PotentialEnergy[Eh]    KineticEnergy[Eh]      TotalEnergy[Eh]      Temperature [K]    Wall time [s]      \n"
                )

        wall_time = sum(self.time_per_step)
        with open(f"{prefix}.out", "a") as f:
            f.write(
                str("%14.4e %20.10e %20.10e %20.10e %20.10e %20.10e \n")
                % (
                    self.step * self.dt,
                    self.epot,
                    self.ekin,
                    self.epot + self.ekin,
                    self.temp,
                    wall_time,
                )
            )
        self.time_per_step = []

    def write_restart(
        self, prefix: str = "ashMD", write_xyz: bool = True, refpdb: str = None
    ):
        """Prints all necessary files for a restart

        Args:
           prefix (str): prefix for restart files
           refpdb (str): name of a reference pdb of the system

        Returns:
           -
        """
        restart_data = {
            "momenta": self.momenta,
            "coords": self.coords,
            "step": self.step,
            "active": self.active_atoms,
            "seed": self.seed_in,
        }
        np.savez(prefix + "_restart", **restart_data)

        if write_xyz:
            with open(prefix + "_restart_geom.xyz", "w+") as f:
                f.write(str("%i\nTIME: %14.7f\n") % (self.natoms, self.step * self.dt))

                for i in range(self.natoms):
                    string = str("%s %20.10e %20.10e %20.10e\n") % (
                        self.molecule.elems[i],
                        self.coords[3 * i + 0] * units.BOHR_to_ANGSTROM,
                        self.coords[3 * i + 1] * units.BOHR_to_ANGSTROM,
                        self.coords[3 * i + 2] * units.BOHR_to_ANGSTROM,
                    )
                    f.write(string)

        if refpdb != None:
            try:
                import mdtraj
            except ImportError as e:
                raise ImportError(" >>> AshMD needs `mdtraj` to write pdb file") from e

            topology = mdtraj.load(refpdb).topology
            pdbout = mdtraj.formats.PDBTrajectoryFile(
                prefix + "_restart_geom.pdb", mode="w", force_overwrite=True
            )
            pdbout.write(
                units.BOHR_to_ANGSTROM * self.coords.reshape(self.natoms, 3), topology
            )
            pdbout.close()

    def print_dcd(self, prefix: str = "ashMD"):
        """saves coordinates to binary dcd format

        Args:
            filename: name of dcd file
        """
        import mdtraj

        if self.dcd == None:
            self.dcd = mdtraj.formats.DCDTrajectoryFile(
                f"{prefix}_traj.dcd", "w", force_overwrite=True
            )
        self.dcd.write(units.BOHR_to_ANGSTROM * self.coords.reshape(self.natoms, 3))

    def get_sampling_data(self):
        """interface to adaptive sampling algorithms. see: https://github.com/ahulm/adaptive_sampling"""
        try:
            from adaptive_sampling.interface.sampling_data import SamplingData

            return SamplingData(
                self.mass,
                self.coords,
                self.forces,
                self.epot,
                self.temp,
                self.natoms,
                self.step,
                self.dt,
            )

        except ImportError as e:
            raise NotImplementedError(
                " >>> AshMD: `get_sampling_data()` is missing `adaptive_sampling` package"
            ) from e


class MonteCarloBarostatASH:
    """Monte Carlo Barostat for OpenMMTheory in AshMD

    Ref: https://doi.org/10.1016/j.cplett.2003.12.039

    Requires one additional energy evaluation every `self.frequency` steps to update the periodic box size,
    and two additional energy evaluations per pressure evaluation if `pressure_from_finite_difference` is True.

    Note, that only the time average of the instanteneous pressure can be expected to equal the pressure applied by the barostat.
    Fluctuations around the average pressure typically are extremely large and very long simulations are required to calculate it accurately!
    Hence, it is recomended to instead monitore the density of the system, as provided in `self.density` in g/cm^3.
    
    Args:
        the_md: AshMD object
        target_pressure: target pressure in bar
        frequency: frequency of barostat updates
        pressure_from_finite_difference: if True, more accurate pressure, else calculate pressure from system forces, neglecting periodic boundary conditions 
        barostat_reporter: if not None, write barostat data to file
        indexCenterMol: index of the molecule that is moved to the center of the periodic box
        useParmed: if True, use `parmed` to update periodic box vectors in OpenMMTheory, else use `Amberfiles`
    """
    
    AVOGADRO = 6.02214076e23
    BAR2ATM = 0.986923
    DALTON2GRAMM = 1.6605402e-24

    def __init__(
        self, 
        the_md: AshMD, 
        target_pressure: float=1.0, 
        frequency: int=None, 
        pressure_from_finite_difference: bool=False, 
        barostat_reporter: str=None,
        indexCenterMol: int=0,
        useParmed: bool=False,
    ):
        if not hasattr(the_md, "calculator") or not hasattr(the_md.calculator, "mm_theory"):
            raise ValueError(" >>> ERROR: AshMD needs a valid calculator with mm_theory")
        if the_md.calculator.mm_theory_name != "OpenMMTheory":
            raise ValueError(" >>> ERROR: `MonteCarloBarostatASH` only available for OpenMMTheory")
        #if not the_md.calculator.mm_theory.Periodic:
        #    raise ValueError(" >>> ERROR: A barostat cannot be used with a non-periodic system")
        if not the_md.langevin:
            raise ValueError(" >>> ERROR: A barostat requires a thermostat to be applied, use `the_md.thermostat=True`")

        self.the_md = the_md
        self.target_pressure = target_pressure * (self.AVOGADRO*1e-25) # convert Bar to kJ/mol/nm^2
        self.frequency = frequency
        self.pressure_from_finite_difference = pressure_from_finite_difference
        self.indexCenterMol = indexCenterMol
        self.useParmed = useParmed

        self.box = self.getPeriodicBoxVectors() # nm
        self.volume = self.getVolume(self.box)  # nm^3
        self.density = self.the_md.mass.sum() / self.volume * self.DALTON2GRAMM / 1.e-21 # g/cm^3
        self.volumeScale = 0.01*self.volume
        self.numAttempted = 0
        self.numAccepted = 0
        self.count_ekin = 0.0
        self.avg_ekin = 0.0
        self.m2_ekin = 0.0
        self.var_ekin = 0.0

        # get the individual molecule's
        self.simulation = the_md.calculator.mm_theory.create_simulation()       
        self.molecules = [list(mol) for mol in self.simulation.context.getMolecules()]
        self.nMolecules = len(self.molecules)
        self.wrapMoleculesToPeriodicBox(self.box, molOrigin=self.indexCenterMol)

        # barostat reporter
        self.barostat_reporter = barostat_reporter
        if self.barostat_reporter:
            with open(self.barostat_reporter, "w") as f:
                f.write(f"{'# TimeStep[fs]':20s} \t {'Volume[nm^3]':20s} \t {'Density[g/cm^3]':20s} \t {'Pressure[Bar]':20s} \t {'MC crit[kJ/mol]':20s} \t {'Accepted?':20s}\n")
    
    def setPeriodicBoxVectors(self, newBox, useParmed: bool = False):
        """modify periodic box vectors of the OpenMMTheory object in AshMD
        See: https://github.com/RagnarB83/ash/blob/81c4f952e7908bfe2e4714836308d0f792a85f4c/ash/interfaces/interface_OpenMM.py#L923

        WARNING: this method only supports `OpenMMTheory` using CHARMM or Amber files

        Args:
            newBox: new periodic box vectors in nm, shape (3, 3)
            useParmed: if True, use `parmed` for box vector update in OpenMMTheory using amber, else use `Amberfiles`
        """
        import openmm
        the_mm = self.the_md.calculator.mm_theory

        # update periodic box vectors in OpenMMTheory
        gromacs = hasattr(the_mm, 'grofile')
        if gromacs:
            raise ValueError(" >>> ERROR: `MonteCarloBarostatASH` does not support Gromacs files for OpenMMTheory")

        charmm = hasattr(the_mm, 'psffile')
        amber = hasattr(the_mm, 'prmtop')        
        the_mm.set_periodics_before_system_creation(
            PBCvectors=newBox * 10.0,       # in Angstrom 
            pdb_pbc_vectors=None,           # not used, if PBCvectors are set
            periodic_cell_dimensions=None,  # not used, if PBCvectors are set
            CHARMMfiles=charmm,             # it True, OpenMMTheory uses CHARMMfiles for system creation
            Amberfiles=amber,               # if True, OpenMMTheory uses AMBERfiles for system cration
            use_parmed=useParmed,           # if True, OpenMMTheory uses Parmed for system creation
        )

        # get parameters of OpenMM system
        if the_mm.nonbondedMethod_PBC == 'PME':
            nonb_method_PBC=openmm.app.PME
        elif the_mm.nonbondedMethod_PBC == 'Ewald':
            nonb_method_PBC=openmm.app.Ewald
        elif the_mm.nonbondedMethod_PBC == 'LJPME':
            nonb_method_PBC=openmm.app.LJPME
        elif the_mm.nonbondedMethod_PBC == 'CutoffPeriodic':
            nonb_method_PBC=openmm.app.CutoffPeriodic

        for force in the_mm.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                dispersion_correction = force.getUseDispersionCorrection()
                PMEParameters = force.getPMEParameters()

        # create new system with updated box vectors (identical to system creaction in init of OpenMMTheory)
        if charmm:
           the_mm.system = self.forcefield.createSystem(
                the_mm.params, 
                nonbondedMethod=nonb_method_PBC,
                constraints=the_mm.autoconstraints,
                hydrogenMass=the_mm.hydrogenmass,
                rigidWater=the_mm.rigidwater, 
                ewaldErrorTolerance=the_mm.ewalderrortolerance,
                nonbondedCutoff=the_mm.periodic_nonbonded_cutoff * openmm.unit.angstroms,
                switchDistance=the_mm.switching_function_distance * openmm.unit.angstroms
            )
                
        elif amber:
            the_mm.system = the_mm.forcefield.createSystem(
                nonbondedMethod=nonb_method_PBC,
                constraints=the_mm.autoconstraints,
                hydrogenMass=the_mm.hydrogenmass,
                rigidWater=the_mm.rigidwater, 
                ewaldErrorTolerance=the_mm.ewalderrortolerance,
                nonbondedCutoff=the_mm.periodic_nonbonded_cutoff * openmm.unit.angstroms
            )
        else:
            raise ValueError(" >>> ERROR: `MonteCarloBarostatASH` only supports CHARMM or AMBER files for OpenMMTheory")

        for force in the_mm.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(dispersion_correction)
                if nonb_method_PBC == openmm.app.PME:
                    force.setPMEParameters(*PMEParameters)      

    def getPeriodicBoxVectors(self) -> np.array:
        """Get the periodic box vectors in nm"""
        return np.asarray(self.the_md.calculator.mm_theory.get_PBC_vectors()) / 10.0 # nm
    
    def getVolume(self, boxVectors) -> float:
        """Calculate the volume of the periodic box"""
        return np.prod(np.linalg.norm(boxVectors, axis=1))

    def updateState(self):
        """Update the state of the barostat
        This method should be called in every MD step, such that the periodic box size is updated every `self.frequency` steps  
        """
        self.updateAvgKinEnergy(tau=self.frequency)
        if self.the_md.step%self.frequency != 0:
            return
        
        # current potential energy
        internalEnergy = np.copy(self.the_md.epot) * units.atomic_to_kJmol # kJ/mol
        
        # modify the periodic box size 
        deltaVolume = self.volumeScale*np.random.uniform(-1.,1.)
        newVolume = self.volume+deltaVolume
        lengthScale = np.cbrt(newVolume/self.volume)       

        # scale coordinates and Box
        newBox = self.box*lengthScale   
        coordsNew = self.scaleCoords(lengthScale) # nm

        # calculte energy of new box
        self.setPeriodicBoxVectors(newBox, useParmed=self.useParmed)
        self.wrapMoleculesToPeriodicBox(newBox, molOrigin=self.indexCenterMol) 
        os.chdir(self.the_md.scratch_dir)
        self.the_md.molecule.replace_coords(
            self.the_md.molecule.elems,
            coordsNew.reshape((self.the_md.natoms, 3)) * 10.0, # nm to Angstrom
        )
        from ash import Singlepoint
        results = Singlepoint(
            theory=self.the_md.calculator,
            fragment=self.the_md.molecule,
            charge=self.the_md.charge,
            mult=self.the_md.mult,
            Grad=False,
        )
        os.chdir(self.the_md.cwd)
        finalEnergy = results.energy * units.atomic_to_kJmol  # kJ/mol

        # Monte Carlo acceptance criterion
        kT = units.kB_in_atomic * units.atomic_to_kJmol * self.the_md.target_temp
        w = finalEnergy-internalEnergy
        w += self.target_pressure*deltaVolume
        w -= self.nMolecules*kT*np.log(newVolume/self.volume)
        
        if w > 0 and np.random.uniform(0,1) > np.exp(-w/kT):
            # reject the step and restore original state of self.the_md
            self.setPeriodicBoxVectors(self.box, useParmed=self.useParmed)
            self.wrapMoleculesToPeriodicBox(self.box, molOrigin=self.indexCenterMol)
            self.the_md.molecule.replace_coords(
                self.the_md.molecule.elems,
                self.the_md.coords.reshape((self.the_md.natoms, 3)) * units.BOHR_to_ANGSTROM, 
            )
            accepted = False
        else:
            # accept the step
            self.the_md.coords = coordsNew / units.BOHR_to_NANOMETER
            self.numAccepted += 1
            accepted = True

        # update box, volume, density and pressure
        self.box = self.getPeriodicBoxVectors()  # nm
        self.volume = self.getVolume(self.getPeriodicBoxVectors())
        self.density = self.the_md.mass.sum() / self.volume * self.DALTON2GRAMM / 1.e-21
        self.pressure = self.computeCurrentPressure(numeric=self.pressure_from_finite_difference)
        if self.barostat_reporter:
            with open(self.barostat_reporter, "a") as f:
                f.write(
                    str("%20.2f \t %20.10e \t %20.10e \t %20.10e \t %20.10e \t %1d\n")
                    % (self.the_md.step * self.the_md.dt, self.volume, self.density, self.pressure, w, int(accepted))
                )

        # adjust acceptance rate
        self.numAttempted += 1        
        if self.numAttempted >= 10:
            if (self.numAccepted < 0.25*self.numAttempted):
                self.volumeScale /= 1.1
                self.numAttempted = 0
                self.numAccepted = 0
            elif (self.numAccepted > 0.75*self.numAttempted):
                self.volumeScale = np.min([self.volumeScale*1.1, self.volume*0.3])                
                self.numAttempted = 0
                self.numAccepted = 0
    
    def wrapMoleculesToPeriodicBox(self, box, molOrigin: int=0):
        """Wrap coordinates of molecules in `self.the_md` to periodic range
        
        Args:
            box: periodic box vectors in nm with origin at (0,0,0), shape (3, 3)
            molOrigin: index of the molecule to center in the periodic box
        """
        box_au = np.copy(box) / units.BOHR_to_NANOMETER 
        coordsNew = np.copy(self.the_md.coords).reshape((self.the_md.natoms, 3))
        
        # center molOrigin in periodic box
        molOrigin = self.molecules[molOrigin]
        coordsMolOrigin = coordsNew[molOrigin].sum(axis=0) / float(len(molOrigin))
        diff = coordsMolOrigin - np.array([box_au[0,0], box_au[1,1], box_au[2,2]])/2.0
        coordsNew -= diff

        # wrap coordinates 
        for molAtoms in self.molecules:
            center = (coordsNew[molAtoms]).sum(axis=0) / float(len(molAtoms))
            diff  = box_au[2]*np.floor(center[2]/box_au[2][2])
            diff += box_au[1]*np.floor((center[1]-diff[1])/box_au[1][1])
            diff += box_au[0]*np.floor((center[0]-diff[0])/box_au[0][0])
            coordsNew[molAtoms] -= diff
        self.the_md.coords = coordsNew.flatten() 

    def scaleCoords(self, lengthScale: float) -> np.array:
        """move center of molecules according to volume change
        """        
        coordsNew = np.copy(self.the_md.coords).reshape((self.the_md.natoms,3)) * units.BOHR_to_NANOMETER
        for molAtoms in self.molecules:
            center = (coordsNew[molAtoms]).sum(axis=0) / float(len(molAtoms))
            centerNew = center*lengthScale
            coordsNew[molAtoms] += centerNew - center
        return coordsNew.flatten()

    def updateAvgKinEnergy(self, tau=None) -> float:
        """Exponentially decaying running average of kinetic energy

        Args:
            tau: decay time of the running average, if None, use conventional running average
        """
        from adaptive_sampling.sampling_tools.utils import welford_var
        self.count_ekin += 1
        mol_ekin = self.computeMolecularKineticEnergy(components=1) # kJ/mol
        self.avg_ekin, self.m2_ekin, self.var_ekin = welford_var(
            self.count_ekin, self.avg_ekin, self.m2_ekin, mol_ekin, tau
        )
        return self.avg_ekin
    
    def computeMolecularKineticEnergy(self, components: int=1) -> float:
        """Calculate the molecular kinetic energy of the system
        
        Args:
            components: number of components to return
                1: total kinetic energy
                3: kinetic energy per Cartesian component (x, y, z)
                6: kinetic energy per Cartesian component and cross terms (xx, yy, zz, xy, xz, yz)
        
        Returns:
            ekin: kinetic energy in kJ/mol, ether total (float) or components (np.array)
        """
        ekin = np.zeros(6)
        momenta = np.copy(self.the_md.momenta).reshape((self.the_md.natoms, 3)) 
        for molAtoms in self.molecules:
            molMass = self.the_md.mass[molAtoms].sum()
            molMom = (momenta[molAtoms]).sum(axis=0) / float(len(molAtoms)) # molecular momentum
            if components == 1:
                ekin[0] += 0.5 * np.dot(molMom, molMom) / molMass
            elif components == 3 or components == 6:
                ekin[:3] += 0.5 * np.square(molMom) / molMass
                if components == 6:
                    ekin[3] += 0.5 * molMom[1] * molMom[0] / molMass
                    ekin[4] += 0.5 * molMom[2] * molMom[0] / molMass
                    ekin[5] += 0.5 * molMom[2] * molMom[1] / molMass
            else:
                raise ValueError(" >>> ERROR: `components` must be 1, 3, or 6 for kinetic energy calculation")
        
        ekin *= units.atomic_to_kJmol
        
        if components == 1:
            return ekin[0]
        elif components == 3:  
            return ekin[:3] 
        else: 
            return ekin

    def computeCurrentPressure(self, numeric: bool=False) -> float:
        """ Calculate instantaneous pressure from from virial equation

        Note, that only the time average of the instanteneous pressure can be expected to equal pressure applied bz the barostat.
        Fluctuations arounf the average are commonly extremely large and vary long simulations are required to calculate tha average accurately!

        Args:
            numeric: if True, use finite difference to calculate the derivative of potential energy with respect to volume as done by OpenMM
                The analytic formulation is fast, but neglects accounting for the periodic boundary conditions!
        """ 
        box = self.getPeriodicBoxVectors() # nm
        volume = self.getVolume(box)       

        if numeric:
            deltaV = 1e-3
            coordsOld = np.copy(self.the_md.coords) * units.BOHR_to_NANOMETER
        
            # Compute the first energy.
            scale1 = 1.0-deltaV
            coordsNew = self.scaleCoords(scale1) 
            newBox = box*scale1
            self.setPeriodicBoxVectors(newBox, useParmed=self.useParmed)
            os.chdir(self.the_md.scratch_dir)
            self.the_md.molecule.replace_coords(
                self.the_md.molecule.elems,
                coordsNew.reshape((self.the_md.natoms, 3)) * 10.0,
            )
            from ash import Singlepoint
            results = Singlepoint(
                theory=self.the_md.calculator,
                fragment=self.the_md.molecule,
                charge=self.the_md.charge,
                mult=self.the_md.mult,
                Grad=False,
            )
            energy1 = results.energy * units.atomic_to_kJmol
       
            # Compute the second energy.
            scale2 = 1.0+deltaV
            coordsNew = self.scaleCoords(scale2)
            newBox = box*scale2
            self.setPeriodicBoxVectors(newBox, useParmed=self.useParmed)
            self.the_md.molecule.replace_coords(
                self.the_md.molecule.elems,
                coordsNew.reshape((self.the_md.natoms, 3)) * 10.0,
            )
            results = Singlepoint(
                theory=self.the_md.calculator,
                fragment=self.the_md.molecule,
                charge=self.the_md.charge,
                mult=self.the_md.mult,
                Grad=False,
            )
            energy2 = results.energy * units.atomic_to_kJmol
            os.chdir(self.the_md.cwd)

            # Restore the context to its original state.
            self.setPeriodicBoxVectors(box, useParmed=self.useParmed)
            self.the_md.molecule.replace_coords(
                self.the_md.molecule.elems,
                coordsOld.reshape((self.the_md.natoms, 3)) * 10.0,
            )
        
        # kinetic pressure term expected to average to 2NkT/2 
        pressure = (2.0/3.0)*self.avg_ekin/volume 

        # potential pressure term -dU/dV = sum(x*f)/Vd 
        if numeric:
            deltaVolume = volume*(scale1*scale1*scale1 - scale2*scale2*scale2)
            pressure -= (energy1-energy2)/deltaVolume
        else:
            # neglecting periodic boundary conditions!
            pressure -= np.sum(self.the_md.coords * self.the_md.forces * units.atomic_to_kJmol)/(3.*volume)
        return pressure / (self.AVOGADRO*1e-25) # convert kJ/mol/nm^2 to Bar
         