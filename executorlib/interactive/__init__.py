from typing import Optional

from executorlib.interactive.executor import (
    InteractiveExecutor,
    InteractiveStepExecutor,
)
from executorlib.shared.inputcheck import (
    check_command_line_argument_lst,
    check_executor,
    check_gpus_per_worker,
    check_init_function,
    check_nested_flux_executor,
    check_oversubscribe,
    check_pmi,
    check_threads_per_core,
    validate_number_of_cores,
)
from executorlib.shared.spawner import (
    SLURM_COMMAND,
    MpiExecSpawner,
    SrunSpawner,
)

try:  # The PyFluxExecutor requires flux-core to be installed.
    from executorlib.interactive.flux import FluxPythonSpawner
except ImportError:
    pass


def create_executor(
    max_workers: int = 1,
    backend: str = "local",
    max_cores: int = 1,
    cores_per_worker: int = 1,
    threads_per_core: int = 1,
    gpus_per_worker: int = 0,
    cwd: Optional[str] = None,
    openmpi_oversubscribe: bool = False,
    slurm_cmd_args: list[str] = [],
    flux_executor=None,
    flux_executor_pmi_mode: Optional[str] = None,
    flux_executor_nesting: bool = False,
    hostname_localhost: bool = False,
    block_allocation: bool = False,
    init_function: Optional[callable] = None,
):
    """
    Instead of returning a executorlib.Executor object this function returns either a executorlib.mpi.PyMPIExecutor,
    executorlib.slurm.PySlurmExecutor or executorlib.flux.PyFluxExecutor depending on which backend is available. The
    executorlib.flux.PyFluxExecutor is the preferred choice while the executorlib.mpi.PyMPIExecutor is primarily used
    for development and testing. The executorlib.flux.PyFluxExecutor requires flux-core from the flux-framework to be
    installed and in addition flux-sched to enable GPU scheduling. Finally, the executorlib.slurm.PySlurmExecutor
    requires the SLURM workload manager to be installed on the system.

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        max_cores (int): defines the number cores which can be used in parallel
        backend (str): Switch between the different backends "flux", "local" or "slurm". The default is "local".
        cores_per_worker (int): number of MPI cores to be used for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_worker (int): number of GPUs per worker - defaults to 0
        cwd (str/None): current working directory where the parallel python task is executed
        openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and SLURM only) - default False
        slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
        flux_executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        flux_executor_pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None (Flux only)
        flux_executor_nesting (bool): Provide hierarchically nested Flux job scheduler inside the submitted function.
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an Executor
                                      running on a different compute node within the same allocation. And in principle
                                      any computer should be able to resolve that their own hostname points to the same
                                      address as localhost. Still MacOS >= 12 seems to disable this look up for security
                                      reasons. So on MacOS it is required to set this option to true
        block_allocation (boolean): To accelerate the submission of a series of python functions with the same
                                    resource requirements, executorlib supports block allocation. In this case all
                                    resources have to be defined on the executor, rather than during the submission
                                    of the individual function.
        init_function (None): optional function to preset arguments for functions which are submitted later
    """
    max_cores = validate_number_of_cores(max_cores=max_cores, max_workers=max_workers)
    check_init_function(block_allocation=block_allocation, init_function=init_function)
    if flux_executor is not None and backend != "flux":
        backend = "flux"
    check_pmi(backend=backend, pmi=flux_executor_pmi_mode)
    executor_kwargs = {
        "cores": cores_per_worker,
        "hostname_localhost": hostname_localhost,
        "cwd": cwd,
    }
    if backend == "flux":
        check_oversubscribe(oversubscribe=openmpi_oversubscribe)
        check_command_line_argument_lst(command_line_argument_lst=slurm_cmd_args)
        executor_kwargs["threads_per_core"] = threads_per_core
        executor_kwargs["gpus_per_core"] = int(gpus_per_worker / cores_per_worker)
        executor_kwargs["flux_executor"] = flux_executor
        executor_kwargs["flux_executor_pmi_mode"] = flux_executor_pmi_mode
        executor_kwargs["flux_executor_nesting"] = flux_executor_nesting
        if block_allocation:
            executor_kwargs["init_function"] = init_function
            return InteractiveExecutor(
                max_workers=int(max_cores / cores_per_worker),
                executor_kwargs=executor_kwargs,
                spawner=FluxPythonSpawner,
            )
        else:
            return InteractiveStepExecutor(
                max_cores=max_cores,
                executor_kwargs=executor_kwargs,
                spawner=FluxPythonSpawner,
            )
    elif backend == "slurm":
        check_executor(executor=flux_executor)
        check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
        executor_kwargs["threads_per_core"] = threads_per_core
        executor_kwargs["gpus_per_core"] = int(gpus_per_worker / cores_per_worker)
        executor_kwargs["slurm_cmd_args"] = slurm_cmd_args
        executor_kwargs["openmpi_oversubscribe"] = openmpi_oversubscribe
        if block_allocation:
            executor_kwargs["init_function"] = init_function
            return InteractiveExecutor(
                max_workers=int(max_cores / cores_per_worker),
                executor_kwargs=executor_kwargs,
                spawner=SrunSpawner,
            )
        else:
            return InteractiveStepExecutor(
                max_cores=max_cores,
                executor_kwargs=executor_kwargs,
                spawner=SrunSpawner,
            )
    else:  # backend="local"
        check_threads_per_core(threads_per_core=threads_per_core)
        check_gpus_per_worker(gpus_per_worker=gpus_per_worker)
        check_command_line_argument_lst(command_line_argument_lst=slurm_cmd_args)
        check_executor(executor=flux_executor)
        check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
        executor_kwargs["openmpi_oversubscribe"] = openmpi_oversubscribe
        if block_allocation:
            executor_kwargs["init_function"] = init_function
            return InteractiveExecutor(
                max_workers=int(max_cores / cores_per_worker),
                executor_kwargs=executor_kwargs,
                spawner=MpiExecSpawner,
            )
        else:
            return InteractiveStepExecutor(
                max_cores=max_cores,
                executor_kwargs=executor_kwargs,
                spawner=MpiExecSpawner,
            )
