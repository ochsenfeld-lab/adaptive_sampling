from ..units import atomic_to_kJmol, BOHR_to_ANGSTROM, BOHR_to_NANOMETER, kB_in_atomic
import openmm
from openmm import unit
import numpy as np

from .sampling_data import SamplingData


class AdaptiveSamplingOpenMM:
    """Perform enhanced sampling with OpenMM

    Args:
        positions: atomic positions from OpenMM (given as `Quantity` object)
        topology: molecular topology object from OpenMM
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
        equil_temp: float = 300.0,
        langevin_damping: float = 1.0,
        cv_atoms: list = [],
        calc_energy: bool = True,
        platform: str = "CPU",
    ):
        import openmm
        from openmm import unit

        self.topology = topology  # OpenMM topology object
        self.system = system  # OpenMM system object

        # MD stuff
        self.calcEnergy = calc_energy  # if OpenMM should return the Energy, this may slow down the simulation
        self.coords = np.asarray(positions._value)
        self.forces = np.zeros_like(self.coords)
        self.ekin = None  # kinetic energy
        self.epot = None  # potential energy
        self.temp = None  # temperature
        self.step = 0  # current md step

        self.equil_temp = (
            equil_temp if hasattr(equil_temp, "unit") else equil_temp * unit.kelvin
        )
        self.langevin_damping = (
            langevin_damping
            if hasattr(langevin_damping, "unit")
            else langevin_damping / unit.picosecond
        )
        self.dt = (
            dt if not hasattr(dt, "unit") else dt._value
        )  # dt without unit for sampling algorithms
        dt_with_unit = (
            dt if hasattr(dt, "unit") else dt * unit.femtosecond
        )  # dt with unit for openmm integrator

        self.natoms = len(self.coords)
        self.mass = []
        for i in range(self.natoms):
            self.mass.append(float(system.getParticleMass(i)._value))
        self.mass = np.asarray(self.mass)

        # list of atoms that participate in CV
        if len(cv_atoms):
            self.cv_atoms = np.sort(cv_atoms)
        else:
            self.cv_atoms = [i for i in range(self.natoms)]

        # Setup OpenMM Integrator object
        self.integrator = openmm.LangevinIntegrator(
            self.equil_temp,
            self.langevin_damping,
            dt_with_unit,
        )

        # Create bias force on cv_atoms
        self.bias_force = openmm.CustomExternalForce("fx*x+fy*y+fz*z")
        self.bias_force.addPerParticleParameter("fx")
        self.bias_force.addPerParticleParameter("fy")
        self.bias_force.addPerParticleParameter("fz")
        for i in self.cv_atoms:
            self.bias_force.addParticle(i, [0.0, 0.0, 0.0])
        self.system.addForce(self.bias_force)

        # create OpenMM simulation object
        self.platform = openmm.Platform.getPlatformByName(platform)
        self.simulation = openmm.app.simulation.Simulation(
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
        """Restart from OpenMM checkpoint file"""
        self.simulation.loadCheckpoint(checkpoint)

    def set_sampling_algorithm(self, sampling_algorithm: object):
        """Set sampling algorithm for MD

        Args:
            sampling_algorithm: sampling algorithm from `adaptive_sampling.sampling_tools`
        """
        self.the_bias = sampling_algorithm

    def run(self, nsteps: int = 1, update_bias_freq: int = 1, **kwargs):
        """perform MD steps

        Args:
            nsteps: number of MD steps
            update_bias_freq: frequency of updates of bias force
        """
        from openmm import unit

        for i in range(int(nsteps / update_bias_freq)):

            # update bias force
            if hasattr(self, "the_bias"):
                bias_force = self.the_bias.step_bias(**kwargs).reshape((self.natoms, 3))
                for i, idx in enumerate(self.cv_atoms):
                    bias_force[idx] *= atomic_to_kJmol / BOHR_to_NANOMETER
                    self.bias_force.setParticleParameters(
                        i,
                        idx,
                        bias_force[idx]
                        * (unit.kilojoules / (unit.nanometer * unit.mole)),
                    )
                self.bias_force.updateParametersInContext(self.simulation.context)
            else:
                import logging

                logging.warning(
                    " >>> AdaptiveSamplingOpenMM: No sampling algorithm specified!"
                )

            # run MD for update_bias_freq steps
            self.simulation.step(update_bias_freq)
            self.step += 1  # update_bias_freq
            self.get_state()

    def get_state(self):
        """Get current state of simulation"""
        state = self.simulation.context.getState(
            getPositions=True,
            getForces=True,
            getEnergy=self.calcEnergy,
        )
        if self.calcEnergy:
            self.ekin = state.getKineticEnergy()._value / atomic_to_kJmol
            self.epot = state.getPotentialEnergy()._value / atomic_to_kJmol
            self.temp = self._get_temperature()
        else:
            self.ekin = 0.0
            self.epot = 0.0
            self.temp = 0.0
        self.coords = (
            np.asarray(state.getPositions()._value).flatten() / BOHR_to_NANOMETER
        )
        self.forces = (
            np.asarray(state.getForces()._value).flatten()
            / atomic_to_kJmol
            * BOHR_to_ANGSTROM
        )

    def _get_temperature(self) -> float:
        return self.ekin * 2.0e0 / (3.0e0 * self.natoms * kB_in_atomic)

    def get_sampling_data(self) -> object:
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
