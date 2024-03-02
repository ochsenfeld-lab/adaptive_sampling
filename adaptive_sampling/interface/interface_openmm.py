from ..units import atomic_to_kJmol, BOHR_to_ANGSTROM, kB_in_atomic
from openmm import *
from openmm.app import *
from openmm import unit
import numpy as np

from .sampling_data import SamplingData

class AdaptiveSamplingOpenMM():
    def __init__(
        self, 
        positions: object,
        topology: object,
        system: object,
        dt: float = 1.0,
        equil_temp: float=300.0,
        langevin_damping: float=1.0,
        langevin_timestep: float=2.0,
        cv_atoms: list=None,
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
        self.units_coversion_force = atomic_to_kJmol / (10.0 / BOHR_to_ANGSTROM)

        # Prepare OpenMM simulation object
        self.integrator = LangevinIntegrator(
            equil_temp * unit.kelvin, 
            langevin_damping / unit.picoseconds, 
            langevin_timestep * unit.femtosecond,
        )

        self.natoms = len(self.coords)
        self.mass = []
        for i in range(self.natoms):
            self.mass.append(float(system.getParticleMass(i)._value))
        self.mass = np.asarray(self.mass)

        self.cv_atoms = cv_atoms
        if self.cv_atoms == None:
            self.cv_atoms = [i for i in range(self.natoms)]

        # create bias force on cv_atoms
        self.bias_force = CustomExternalForce("-fx*x-fy*y-fz*z")
        self.bias_force.addPerParticleParameter("fx")
        self.bias_force.addPerParticleParameter("fy")
        self.bias_force.addPerParticleParameter("fz")  
        for i in self.cv_atoms:
            self.bias_force.addParticle(i, [0.0,0.0,0.0])  
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
                bias_force[idx] *= self.units_coversion_force
                self.bias_force.setParticleParameters(
                    i, 
                    idx, 
                    bias_force[idx] * (unit.kilojoule/(unit.nanometer*unit.mole)),
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
        self.coords = np.asarray(state.getPositions()._value).flatten() * 10.0 / BOHR_to_ANGSTROM 
        self.forces = np.asarray(state.getForces()._value).flatten() / self.units_coversion_force
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