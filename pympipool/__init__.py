import os
from ._version import get_versions
from pympipool.mpi.executor import PyMPIExecutor

try:  # The PyFluxExecutor requires flux-core to be installed.
    from pympipool.flux.executor import PyFluxExecutor

    flux_installed = "FLUX_URI" in os.environ
except ImportError:
    flux_installed = False
    pass

try:  # The PySlurmExecutor requires the srun command to be available.
    from pympipool.slurm.executor import PySlurmExecutor

    slurm_installed = True
except ImportError:
    slurm_installed = False
    pass


__version__ = get_versions()["version"]
del get_versions


class Executor:
    def __new__(
        cls,
        max_workers,
        cores_per_worker=1,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
    ):
        if flux_installed:
            return PyFluxExecutor(
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                init_function=init_function,
                cwd=cwd,
                sleep_interval=sleep_interval,
            )
        elif slurm_installed:
            return PySlurmExecutor(
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                init_function=init_function,
                cwd=cwd,
                sleep_interval=sleep_interval,
            )
        else:
            return PyMPIExecutor(
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                init_function=init_function,
                cwd=cwd,
                sleep_interval=sleep_interval,
            )
