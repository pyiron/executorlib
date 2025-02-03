from typing import Callable, Optional, Union

from executorlib.interactive.shared import (
    InteractiveExecutor,
    InteractiveStepExecutor,
)
from executorlib.interactive.slurm import SrunSpawner
from executorlib.interactive.slurm import (
    validate_max_workers as validate_max_workers_slurm,
)
from executorlib.standalone.inputcheck import (
    check_command_line_argument_lst,
    check_executor,
    check_flux_log_files,
    check_gpus_per_worker,
    check_init_function,
    check_nested_flux_executor,
    check_oversubscribe,
    check_pmi,
    validate_number_of_cores,
)
from executorlib.standalone.interactive.spawner import MpiExecSpawner

try:  # The PyFluxExecutor requires flux-base to be installed.
    from executorlib.interactive.flux import FluxPythonSpawner
    from executorlib.interactive.flux import (
        validate_max_workers as validate_max_workers_flux,
    )
except ImportError:
    pass


def create_executor(
    max_workers: Optional[int] = None,
    backend: str = "local",
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    flux_executor=None,
    flux_executor_pmi_mode: Optional[str] = None,
    flux_executor_nesting: bool = False,
    flux_log_files: bool = False,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
) -> Union[InteractiveStepExecutor, InteractiveExecutor]:
    """
    Instead of returning a executorlib.Executor object this function returns either a executorlib.mpi.PyMPIExecutor,
    executorlib.slurm.PySlurmExecutor or executorlib.flux.PyFluxExecutor depending on which backend is available. The
    executorlib.flux.PyFluxExecutor is the preferred choice while the executorlib.mpi.PyMPIExecutor is primarily used
    for development and testing. The executorlib.flux.PyFluxExecutor requires flux-base from the flux-framework to be
    installed and in addition flux-sched to enable GPU scheduling. Finally, the executorlib.slurm.PySlurmExecutor
    requires the SLURM workload manager to be installed on the system.

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        backend (str): Switch between the different backends "flux", "local" or "slurm". The default is "local".
        max_cores (int): defines the number cores which can be used in parallel
        cache_directory (str, optional): The directory to store cache files. Defaults to "cache".
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - threads_per_core (int): number of OpenMP threads to be used for each function call
                              - gpus_per_core (int): number of GPUs per worker - defaults to 0
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and
                                                              SLURM only) - default False
                              - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
        flux_executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        flux_executor_pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None (Flux only)
        flux_executor_nesting (bool): Provide hierarchically nested Flux job scheduler inside the submitted function.
        flux_log_files (bool, optional): Write flux stdout and stderr files. Defaults to False.
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
    if resource_dict is None:
        resource_dict = {}
    if flux_executor is not None and backend != "flux_allocation":
        backend = "flux_allocation"
    if backend == "flux_allocation":
        check_init_function(
            block_allocation=block_allocation, init_function=init_function
        )
        check_pmi(backend=backend, pmi=flux_executor_pmi_mode)
        resource_dict["cache_directory"] = cache_directory
        resource_dict["hostname_localhost"] = hostname_localhost
        check_oversubscribe(
            oversubscribe=resource_dict.get("openmpi_oversubscribe", False)
        )
        check_command_line_argument_lst(
            command_line_argument_lst=resource_dict.get("slurm_cmd_args", [])
        )
        return create_flux_allocation_executor(
            max_workers=max_workers,
            max_cores=max_cores,
            cache_directory=cache_directory,
            resource_dict=resource_dict,
            flux_executor=flux_executor,
            flux_executor_pmi_mode=flux_executor_pmi_mode,
            flux_executor_nesting=flux_executor_nesting,
            flux_log_files=flux_log_files,
            hostname_localhost=hostname_localhost,
            block_allocation=block_allocation,
            init_function=init_function,
        )
    elif backend == "slurm_allocation":
        check_pmi(backend=backend, pmi=flux_executor_pmi_mode)
        check_executor(executor=flux_executor)
        check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
        check_flux_log_files(flux_log_files=flux_log_files)
        return create_slurm_allocation_executor(
            max_workers=max_workers,
            max_cores=max_cores,
            cache_directory=cache_directory,
            resource_dict=resource_dict,
            hostname_localhost=hostname_localhost,
            block_allocation=block_allocation,
            init_function=init_function,
        )
    elif backend == "local":
        check_pmi(backend=backend, pmi=flux_executor_pmi_mode)
        check_executor(executor=flux_executor)
        check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
        check_flux_log_files(flux_log_files=flux_log_files)
        return create_local_executor(
            max_workers=max_workers,
            max_cores=max_cores,
            cache_directory=cache_directory,
            resource_dict=resource_dict,
            hostname_localhost=hostname_localhost,
            block_allocation=block_allocation,
            init_function=init_function,
        )
    else:
        raise ValueError(
            "The supported backends are slurm_allocation, slurm_submission, flux_allocation, flux_submission and local."
        )


def create_flux_allocation_executor(
    max_workers: Optional[int] = None,
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    flux_executor=None,
    flux_executor_pmi_mode: Optional[str] = None,
    flux_executor_nesting: bool = False,
    flux_log_files: bool = False,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
) -> Union[InteractiveStepExecutor, InteractiveExecutor]:
    check_init_function(block_allocation=block_allocation, init_function=init_function)
    check_pmi(backend="flux_allocation", pmi=flux_executor_pmi_mode)
    if resource_dict is None:
        resource_dict = {}
    cores_per_worker = resource_dict.get("cores", 1)
    resource_dict["cache_directory"] = cache_directory
    resource_dict["hostname_localhost"] = hostname_localhost
    check_oversubscribe(oversubscribe=resource_dict.get("openmpi_oversubscribe", False))
    check_command_line_argument_lst(
        command_line_argument_lst=resource_dict.get("slurm_cmd_args", [])
    )
    if "openmpi_oversubscribe" in resource_dict:
        del resource_dict["openmpi_oversubscribe"]
    if "slurm_cmd_args" in resource_dict:
        del resource_dict["slurm_cmd_args"]
    resource_dict["flux_executor"] = flux_executor
    resource_dict["flux_executor_pmi_mode"] = flux_executor_pmi_mode
    resource_dict["flux_executor_nesting"] = flux_executor_nesting
    resource_dict["flux_log_files"] = flux_log_files
    if block_allocation:
        resource_dict["init_function"] = init_function
        max_workers = validate_number_of_cores(
            max_cores=max_cores,
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
            set_local_cores=False,
        )
        validate_max_workers_flux(
            max_workers=max_workers,
            cores=cores_per_worker,
            threads_per_core=resource_dict.get("threads_per_core", 1),
        )
        return InteractiveExecutor(
            max_workers=max_workers,
            executor_kwargs=resource_dict,
            spawner=FluxPythonSpawner,
        )
    else:
        return InteractiveStepExecutor(
            max_cores=max_cores,
            max_workers=max_workers,
            executor_kwargs=resource_dict,
            spawner=FluxPythonSpawner,
        )


def create_slurm_allocation_executor(
    max_workers: Optional[int] = None,
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
) -> Union[InteractiveStepExecutor, InteractiveExecutor]:
    check_init_function(block_allocation=block_allocation, init_function=init_function)
    if resource_dict is None:
        resource_dict = {}
    cores_per_worker = resource_dict.get("cores", 1)
    resource_dict["cache_directory"] = cache_directory
    resource_dict["hostname_localhost"] = hostname_localhost
    if block_allocation:
        resource_dict["init_function"] = init_function
        max_workers = validate_number_of_cores(
            max_cores=max_cores,
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
            set_local_cores=False,
        )
        validate_max_workers_slurm(
            max_workers=max_workers,
            cores=cores_per_worker,
            threads_per_core=resource_dict.get("threads_per_core", 1),
        )
        return InteractiveExecutor(
            max_workers=max_workers,
            executor_kwargs=resource_dict,
            spawner=SrunSpawner,
        )
    else:
        return InteractiveStepExecutor(
            max_cores=max_cores,
            max_workers=max_workers,
            executor_kwargs=resource_dict,
            spawner=SrunSpawner,
        )


def create_local_executor(
    max_workers: Optional[int] = None,
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
) -> Union[InteractiveStepExecutor, InteractiveExecutor]:
    check_init_function(block_allocation=block_allocation, init_function=init_function)
    if resource_dict is None:
        resource_dict = {}
    cores_per_worker = resource_dict.get("cores", 1)
    resource_dict["cache_directory"] = cache_directory
    resource_dict["hostname_localhost"] = hostname_localhost

    check_gpus_per_worker(gpus_per_worker=resource_dict.get("gpus_per_core", 0))
    check_command_line_argument_lst(
        command_line_argument_lst=resource_dict.get("slurm_cmd_args", [])
    )
    if "threads_per_core" in resource_dict:
        del resource_dict["threads_per_core"]
    if "gpus_per_core" in resource_dict:
        del resource_dict["gpus_per_core"]
    if "slurm_cmd_args" in resource_dict:
        del resource_dict["slurm_cmd_args"]
    if block_allocation:
        resource_dict["init_function"] = init_function
        return InteractiveExecutor(
            max_workers=validate_number_of_cores(
                max_cores=max_cores,
                max_workers=max_workers,
                cores_per_worker=cores_per_worker,
                set_local_cores=True,
            ),
            executor_kwargs=resource_dict,
            spawner=MpiExecSpawner,
        )
    else:
        return InteractiveStepExecutor(
            max_cores=max_cores,
            max_workers=max_workers,
            executor_kwargs=resource_dict,
            spawner=MpiExecSpawner,
        )
