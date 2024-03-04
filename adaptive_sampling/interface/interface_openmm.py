from ..units import atomic_to_kJmol, BOHR_to_ANGSTROM, BOHR_to_NANOMETER, kB_in_atomic
from openmm import *
from openmm.app import *
from openmm import unit
import numpy as np

from .sampling_data import SamplingData

class AdaptiveSamplingOpenMM():
    """Performes biased simulations with OpenMM

    Args:
        positions: atomic positions from OpenMM
        topology: molecular topology from OpenMM
        system: system object from OpenMM
        dt: MD timestep in fs
        equil_temp: equilibrium temperature for langevin dynamics
        langevin_damping: friction factor for langevin dynamics
        cv_atoms: indices of atoms that are involfed in Collective Variable
        platform: OpenMM Platform type. 'CPU', 'CUDA' of 'OpenCL'
    """
    def __init__(
        self, 
        positions: object,
        topology: object,
        system: object,
        dt: float = 2.0,
        equil_temp: float=300.0,
        langevin_damping: float=1.0,
        cv_atoms: list=[],
        platform: str='CPU',
    ):
        self.topology = topology
        self.system = system

        # MD stuff
        self.equil_temp = equil_temp
        self.step = 0
        self.coords = np.asarray(positions._value)
        self.forces = np.zeros_like(self.coords)
        self.ekin = None
        self.epot = None
        self.temp = None
        self.dt = dt 

        # Prepare OpenMM simulation object
        self.integrator = LangevinIntegrator(
            equil_temp * unit.kelvin, 
            langevin_damping / unit.picoseconds, 
            dt * unit.femtosecond,
        )

        self.natoms = len(self.coords)
        self.mass = []
        for i in range(self.natoms):
            self.mass.append(float(system.getParticleMass(i)._value))
        self.mass = np.asarray(self.mass)

        self.cv_atoms = np.sort(cv_atoms)
        if not len(self.cv_atoms):
            self.cv_atoms = [i for i in range(self.natoms)]

        # create bias force on cv_atoms
        self.bias_force = CustomExternalForce("fx*x+fy*y+fz*z")
        self.bias_force.addPerParticleParameter("fx")
        self.bias_force.addPerParticleParameter("fy")
        self.bias_force.addPerParticleParameter("fz")  
        for i in self.cv_atoms:
            self.bias_force.addParticle(i, [0.0, 0.0, 0.0])  
        self.system.addForce(self.bias_force)

        # create OpenMM simulation object
        self.platform = Platform.getPlatformByName(platform)
        self.simulation = Simulation(
            self.topology, 
            self.system, 
            self.integrator, 
            self.platform,
        )
        self.simulation.context.setPositions(positions)
        self.simulation.context.setVelocitiesToTemperature(self.equil_temp)
        self.simulation.step(0)
        self.get_state()
    
    def restart(self, checkpoint: str):
        """ Restart from OpenMM checkpoint file
        """
        self.simulation.loadCheckpoint(checkpoint)

    def set_sampling_algorithm(self, sampling_algorithm: object):
        """ Set sampling algorithm for MD
        """
        self.the_bias = sampling_algorithm
    
    def run(
        self, 
        nsteps: int=1, 
        update_bias_freq: int=1, 
        **kwargs
    ):
        """ perform MD steps

        Args:
            nsteps: number of MD steps
            update_bias_freq: frequency of updates of bias force
        """
        for i in range(nsteps):
            
            # update bias force
            bias_force = self.the_bias.step_bias(**kwargs).reshape((self.natoms,3))
            for i, idx in enumerate(self.cv_atoms):
                bias_force[idx] *= atomic_to_kJmol / BOHR_to_NANOMETER
                self.bias_force.setParticleParameters(
                    i, 
                    idx, 
                    bias_force[idx] * (unit.kilojoules / (unit.nanometer*unit.mole)),
                )
            self.bias_force.updateParametersInContext(self.simulation.context)
            
            # run MD
            self.simulation.step(update_bias_freq)
            self.step += update_bias_freq
            self.get_state()
    
    def get_state(self):
        """ Get current state of simulation
        """
        state = self.simulation.context.getState(
            getPositions=True,
            getForces=True,
            getEnergy=True,
        )
        self.ekin = state.getKineticEnergy()._value / atomic_to_kJmol
        self.epot = state.getPotentialEnergy()._value / atomic_to_kJmol
        self.coords = np.asarray(state.getPositions()._value).flatten() / BOHR_to_NANOMETER 
        self.forces = np.asarray(state.getForces()._value).flatten() / atomic_to_kJmol * BOHR_to_ANGSTROM
        self.temp = self._get_temp()

    def _get_temp(self):
        return self.ekin*2.e0/(3.e0*self.natoms*kB_in_atomic)
    
    def get_sampling_data(self):
        """interface to adaptive_sampling"""
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