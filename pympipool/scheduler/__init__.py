import os
import shutil
from typing import Optional
from pympipool.scheduler.local import (
    PyLocalExecutor,
    PyLocalStepExecutor,
)
from pympipool.scheduler.slurm import (
    PySlurmExecutor,
    PySlurmStepExecutor,
)
from pympipool.shared.interface import SLURM_COMMAND
from pympipool.shared.inputcheck import (
    check_command_line_argument_lst,
    check_gpus_per_worker,
    check_threads_per_core,
    check_oversubscribe,
    check_executor,
    check_init_function,
    check_pmi,
    validate_backend,
    validate_number_of_cores,
)

try:  # The PyFluxExecutor requires flux-core to be installed.
    from pympipool.scheduler.flux import (
        PyFluxExecutor,
        PyFluxStepExecutor,
    )

    flux_installed = "FLUX_URI" in os.environ
except ImportError:
    flux_installed = False
    pass

# The PySlurmExecutor requires the srun command to be available.
slurm_installed = shutil.which(SLURM_COMMAND) is not None


def create_executor(
    max_workers: int = 1,
    max_cores: int = 1,
    cores_per_worker: int = 1,
    threads_per_core: int = 1,
    gpus_per_worker: int = 0,
    oversubscribe: bool = False,
    cwd: Optional[str] = None,
    executor=None,
    hostname_localhost: bool = False,
    backend: str = "auto",
    block_allocation: bool = False,
    init_function: Optional[callable] = None,
    command_line_argument_lst: list[str] = [],
    pmi: Optional[str] = None,
):
    """
    Instead of returning a pympipool.Executor object this function returns either a pympipool.mpi.PyMPIExecutor,
    pympipool.slurm.PySlurmExecutor or pympipool.flux.PyFluxExecutor depending on which backend is available. The
    pympipool.flux.PyFluxExecutor is the preferred choice while the pympipool.mpi.PyMPIExecutor is primarily used
    for development and testing. The pympipool.flux.PyFluxExecutor requires flux-core from the flux-framework to be
    installed and in addition flux-sched to enable GPU scheduling. Finally, the pympipool.slurm.PySlurmExecutor
    requires the SLURM workload manager to be installed on the system.

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        max_cores (int): defines the number cores which can be used in parallel
        cores_per_worker (int): number of MPI cores to be used for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_worker (int): number of GPUs per worker - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and SLURM only) - default False
        cwd (str/None): current working directory where the parallel python task is executed
        executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an Executor
                                      running on a different compute node within the same allocation. And in principle
                                      any computer should be able to resolve that their own hostname points to the same
                                      address as localhost. Still MacOS >= 12 seems to disable this look up for security
                                      reasons. So on MacOS it is required to set this option to true
        backend (str): Switch between the different backends "flux", "local" or "slurm". Alternatively, when "auto"
                       is selected (the default) the available backend is determined automatically.
        block_allocation (boolean): To accelerate the submission of a series of python functions with the same
                                    resource requirements, pympipool supports block allocation. In this case all
                                    resources have to be defined on the executor, rather than during the submission
                                    of the individual function.
        init_function (None): optional function to preset arguments for functions which are submitted later
        command_line_argument_lst (list): Additional command line arguments for the srun call (SLURM only)
        pmi (str): PMI interface to use (OpenMPI v5 requires pmix) default is None (Flux only)

    """
    max_cores = validate_number_of_cores(max_cores=max_cores, max_workers=max_workers)
    check_init_function(block_allocation=block_allocation, init_function=init_function)
    backend = validate_backend(
        backend=backend, flux_installed=flux_installed, slurm_installed=slurm_installed
    )
    check_pmi(backend=backend, pmi=pmi)
    executor_kwargs = {
        "cores": cores_per_worker,
        "hostname_localhost": hostname_localhost,
        "cwd": cwd,
    }
    if backend == "flux":
        check_oversubscribe(oversubscribe=oversubscribe)
        check_command_line_argument_lst(
            command_line_argument_lst=command_line_argument_lst
        )
        executor_kwargs["threads_per_core"] = threads_per_core
        executor_kwargs["gpus_per_core"] = int(gpus_per_worker / cores_per_worker)
        executor_kwargs["executor"] = executor
        executor_kwargs["pmi"] = pmi
        if block_allocation:
            executor_kwargs["init_function"] = init_function
            return PyFluxExecutor(
                max_workers=int(max_cores / cores_per_worker),
                executor_kwargs=executor_kwargs,
            )
        else:
            return PyFluxStepExecutor(
                max_cores=max_cores,
                executor_kwargs=executor_kwargs,
            )
    elif backend == "slurm":
        check_executor(executor=executor)
        executor_kwargs["threads_per_core"] = threads_per_core
        executor_kwargs["gpus_per_core"] = int(gpus_per_worker / cores_per_worker)
        executor_kwargs["command_line_argument_lst"] = command_line_argument_lst
        executor_kwargs["oversubscribe"] = oversubscribe
        if block_allocation:
            executor_kwargs["init_function"] = init_function
            return PySlurmExecutor(
                max_workers=int(max_cores / cores_per_worker),
                executor_kwargs=executor_kwargs,
            )
        else:
            return PySlurmStepExecutor(
                max_cores=max_cores,
                executor_kwargs=executor_kwargs,
            )
    else:  # backend="local"
        check_threads_per_core(threads_per_core=threads_per_core)
        check_gpus_per_worker(gpus_per_worker=gpus_per_worker)
        check_command_line_argument_lst(
            command_line_argument_lst=command_line_argument_lst
        )
        check_executor(executor=executor)
        executor_kwargs["oversubscribe"] = oversubscribe
        if block_allocation:
            executor_kwargs["init_function"] = init_function
            return PyLocalExecutor(
                max_workers=int(max_cores / cores_per_worker),
                executor_kwargs=executor_kwargs,
            )
        else:
            return PyLocalStepExecutor(
                max_cores=max_cores,
                executor_kwargs=executor_kwargs,
            )
