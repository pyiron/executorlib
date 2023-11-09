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
    """
    The pympipool.Executor leverages either the message passing interface (MPI), the SLURM workload manager or preferable
    the flux framework for distributing python functions within a given resource allocation. In contrast to the
    mpi4py.futures.MPIPoolExecutor the pympipool.Executor can be executed in a serial python process and does not
    require the python script to be executed with MPI. It is even possible to execute the pympipool.Executor directly in
    an interactive Jupyter notebook.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel
        cores_per_worker (int): number of MPI cores to be used for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_worker (int): number of GPUs per worker - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed
        sleep_interval (float): synchronization interval - default 0.1

    Examples:
        ```
        >>> import numpy as np
        >>> from pympipool import Executor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> def init_k():
        >>>     return {"k": 3}
        >>>
        >>> with Executor(cores=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        max_workers=1,
        cores_per_worker=1,
        threads_per_core=1,
        gpus_per_worker=0,
        oversubscribe=False,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
        executor=None,
    ):
        # Use __new__() instead of __init__(). This function is only implemented to enable auto-completion.
        pass

    def __new__(
        cls,
        max_workers=1,
        cores_per_worker=1,
        threads_per_core=1,
        gpus_per_worker=0,
        oversubscribe=False,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
        executor=None,
    ):
        """
        Instead of returning a pympipool.Executor object this function returns either a pympipool.mpi.PyMPIExecutor,
        pympipool.slurm.PySlurmExecutor or pympipool.flux.PyFluxExecutor depending on which backend is available. The
        pympipool.flux.PyFluxExecutor is the preferred choice while the pympipool.mpi.PyMPIExecutor is primarily used
        for development and testing. The pympipool.flux.PyFluxExecutor requires flux-core from the flux-framework to be
        installed and in addition flux-sched to enable GPU scheduling. Finally, the pympipool.slurm.PySlurmExecutor
        requires the SLURM workload manager to be installed on the system.

        Args:
            max_workers (int): defines the number workers which can execute functions in parallel
            cores_per_worker (int): number of MPI cores to be used for each function call
            threads_per_core (int): number of OpenMP threads to be used for each function call
            gpus_per_worker (int): number of GPUs per worker - defaults to 0
            oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
            init_function (None): optional function to preset arguments for functions which are submitted later
            cwd (str/None): current working directory where the parallel python task is executed
            sleep_interval (float): synchronization interval - default 0.1
        """
        if flux_installed:
            if oversubscribe:
                raise ValueError(
                    "Oversubscribing is not supported for the pympipool.flux.PyFLuxExecutor backend."
                    "Please use oversubscribe=False instead of oversubscribe=True."
                )
            return PyFluxExecutor(
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                threads_per_core=threads_per_core,
                gpus_per_worker=gpus_per_worker,
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
            if threads_per_core != 1:
                raise ValueError(
                    "Thread based parallelism is not supported for the pympipool.mpi.PyMPIExecutor backend."
                    "Please use threads_per_core=1 instead of threads_per_core="
                    + str(threads_per_core)
                    + "."
                )
            if gpus_per_worker != 0:
                raise ValueError(
                    "GPU assignment is not supported for the pympipool.mpi.PyMPIExecutor backend."
                    "Please use gpus_per_worker=0 instead of gpus_per_worker="
                    + str(gpus_per_worker)
                    + "."
                )
            return PyMPIExecutor(
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                init_function=init_function,
                cwd=cwd,
                sleep_interval=sleep_interval,
            )
