import os
import numpy as np
from ..units import *


class PLUMED_CV:
    """Calculate CV from plumed

    Args:

    """

    def __init__(
        self,
        cv_def: str = None,
        natoms: int = None,
        scratch_dir: str = "PLUMED",
    ):
        if not cv_def:
            raise ValueError(" >>> PLUMED: cv_def missing for plumed interface")
        if not natoms:
            raise ValueError(" >>> PLUMED: natoms missing for plumed interface")

        try:
            import plumed
        except ImportError:
            print("Found no plumed library. Install via: pip install plumed")

        self.natoms = natoms
        self.masses = np.ones(natoms)
        self.step = 0

        self.cwd = os.getcwd()
        self.scratch_dir = scratch_dir
        if not os.path.isdir(self.scratch_dir):
            os.mkdir(self.scratch_dir)

        # init PLUMED
        os.chdir(self.scratch_dir)
        self.plumedobj = plumed.Plumed()
        self.plumedobj.cmd("setMDEngine", "python")
        self.plumedobj.cmd("setTimestep", 1.0)
        self.plumedobj.cmd("setNatoms", self.natoms)
        self.plumedobj.cmd("setLogFile", "plumed.log")
        self.plumedobj.cmd("init")
        self.plumedobj.cmd(
            "readInputLine",
            "UNITS LENGTH={} ENERGY={} TIME={}".format("A", "kj/mol", "ps"),
        )

        # setup colvar
        inputlines = cv_def.splitlines()
        for line in inputlines:
            self.plumedobj.cmd("readInputLine", f"cv1: {line}")

        # write outpur to file
        self.plumedobj.cmd("readInputLine", "FLUSH STRIDE=1")
        self.plumedobj.cmd("readInputLine", "PRINT ARG=cv1 FILE=COLVAR STRIDE=1")
        self.plumedobj.cmd(
            "readInputLine", "DUMPDERIVATIVES ARG=cv1 FILE=GRADIENT STRIDE=1"
        )
        os.chdir(self.cwd)

    def calc(self, z: np.array):
        """Calc plumed CV and bias force"""
        z = z.reshape((int(len(z.flatten()) / 3), 3)) / 10
        z = z.astype(np.float64)
        self.step += 1

        # run plumed in scratch dir
        os.chdir(self.scratch_dir)
        self.plumedobj.cmd("setStep", self.step)
        self.plumedobj.cmd("setMasses", np.array(self.masses))
        self.plumedobj.cmd("setBox", np.zeros((3, 3)))
        self.plumedobj.cmd("setVirial", np.zeros((3, 3)))
        self.plumedobj.cmd("setPositions", z)
        self.plumedobj.cmd("setForces", np.zeros_like(z, dtype=np.float64))
        self.plumedobj.cmd("calc")
        self.plumedobj.finalize()

        self.cv = np.loadtxt("./COLVAR")[1]
        self.gradient = np.loadtxt("./GRADIENT")
        self.gradient = self.gradient[:, 2].reshape((int(len(self.gradient) / 3), 3))

        # clean up
        for file in os.listdir("."):
            if os.path.isfile(file) and file.startswith("bck"):
                os.remove(file)
        os.chdir(self.cwd)

        return self.cv
