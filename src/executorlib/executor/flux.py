from typing import Callable, Optional, Union

from executorlib.executor.base import BaseExecutor
from executorlib.standalone.inputcheck import (
    check_command_line_argument_lst,
    check_init_function,
    check_log_obj_size,
    check_oversubscribe,
    check_plot_dependency_graph,
    check_pmi,
    check_refresh_rate,
    check_restart_limit,
    check_wait_on_shutdown,
    validate_number_of_cores,
)
from executorlib.standalone.validate import (
    validate_resource_dict,
    validate_resource_dict_with_optional_keys,
)
from executorlib.task_scheduler.interactive.blockallocation import (
    BlockAllocationTaskScheduler,
)
from executorlib.task_scheduler.interactive.dependency import DependencyTaskScheduler
from executorlib.task_scheduler.interactive.onetoone import OneProcessTaskScheduler


class FluxJobExecutor(BaseExecutor):
    """
    The executorlib.FluxJobExecutor leverages either the message passing interface (MPI), the SLURM workload manager or
    preferable the flux framework for distributing python functions within a given resource allocation. In contrast to
    the mpi4py.futures.MPIPoolExecutor the executorlib.FluxJobExecutor can be executed in a serial python process and
    does not require the python script to be executed with MPI. It is even possible to execute the
    executorlib.FluxJobExecutor directly in an interactive Jupyter notebook.

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        max_cores (int): defines the number cores which can be used in parallel
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              * cores (int): number of MPI cores to be used for each function call
                              * threads_per_core (int): number of OpenMP threads to be used for each function call
                              * gpus_per_core (int): number of GPUs per worker - defaults to 0
                              * cwd (str): current working directory where the parallel python task is executed
                              * cache_key (str): Rather than using the internal hashing of executorlib the user can
                                                 provide an external cache_key to identify tasks on the file system.
                              * num_nodes (int): number of compute nodes used for the evaluation of the Python function.
                              * exclusive (bool): boolean flag to reserve exclusive access to selected compute nodes -
                                                  do not allow other tasks to use the same compute node.
                              * error_log_file (str): path to the error log file, primarily used to merge the log of
                                                      multiple tasks in one file.
                              * run_time_limit (int): the maximum time the execution of the submitted Python function is
                                                      allowed to take in seconds.
                              * priority (int): the queuing system priority assigned to a given Python function to
                                                influence the scheduling.
                              *`slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
        pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None
        flux_executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        flux_executor_nesting (bool): Provide hierarchically nested Flux job scheduler inside the submitted function.
        flux_log_files (bool, optional): Write flux stdout and stderr files. Defaults to False.
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        block_allocation (boolean): To accelerate the submission of a series of python functions with the same resource
                                    requirements, executorlib supports block allocation. In this case all resources have
                                    to be defined on the executor, rather than during the submission of the individual
                                    function.
        init_function (None): optional function to preset arguments for functions which are submitted later
        disable_dependencies (boolean): Disable resolving future objects during the submission.
        refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
        plot_dependency_graph (bool): Plot the dependencies of multiple future objects without executing them. For
                                      debugging purposes and to get an overview of the specified dependencies.
        plot_dependency_graph_filename (str): Name of the file to store the plotted graph in.
        export_workflow_filename (str): Name of the file to store the exported workflow graph in.
        log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.
        wait (bool): Whether to wait for the completion of all tasks before shutting down the executor.
        restart_limit (int): The maximum number of restarting worker processes.
        openmpi_oversubscribe (bool): adds the `--oversubscribe` command flag (OpenMPI and SLURM) - default False

    Examples:
        ```
        >>> import numpy as np
        >>> from executorlib import FluxJobExecutor
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
        >>> with FluxJobExecutor(max_workers=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        cache_directory: Optional[str] = None,
        max_cores: Optional[int] = None,
        resource_dict: Optional[dict] = None,
        pmi_mode: Optional[str] = None,
        flux_executor=None,
        flux_executor_nesting: bool = False,
        flux_log_files: bool = False,
        hostname_localhost: Optional[bool] = None,
        block_allocation: bool = False,
        init_function: Optional[Callable] = None,
        disable_dependencies: bool = False,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        plot_dependency_graph_filename: Optional[str] = None,
        export_workflow_filename: Optional[str] = None,
        log_obj_size: bool = False,
        wait: bool = True,
        restart_limit: int = 0,
        openmpi_oversubscribe: bool = False,
    ):
        """
        The executorlib.FluxJobExecutor leverages either the message passing interface (MPI), the SLURM workload manager
         or preferable the flux framework for distributing python functions within a given resource allocation. In
         contrast to the mpi4py.futures.MPIPoolExecutor the executorlib.FluxJobExecutor can be executed in a serial
         python process and does not require the python script to be executed with MPI. It is even possible to execute
         the executorlib.FluxJobExecutor directly in an interactive Jupyter notebook.

        Args:
            max_workers (int): for backwards compatibility with the standard library, max_workers also defines the
                               number of cores which can be used in parallel - just like the max_cores parameter. Using
                               max_cores is recommended, as computers have a limited number of compute cores.
            cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
            max_cores (int): defines the number cores which can be used in parallel
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                                  * cores (int): number of MPI cores to be used for each function call
                                  * threads_per_core (int): number of OpenMP threads to be used for each function call
                                  * gpus_per_core (int): number of GPUs per worker - defaults to 0
                                  * cwd (str): current working directory where the parallel python task is executed
                                  * cache_key (str): Rather than using the internal hashing of executorlib the user can
                                                      provide an external cache_key to identify tasks on the file system.
                                  * num_nodes (int): number of compute nodes used for the evaluation of the Python
                                                     function.
                                  * exclusive (bool): boolean flag to reserve exclusive access to selected compute nodes
                                                      - do not allow other tasks to use the same compute node.
                                  * error_log_file (str): path to the error log file, primarily used to merge the log of
                                                          multiple tasks in one file.
                                  * run_time_limit (int): the maximum time the execution of the submitted Python
                                                        function is allowed to take in seconds.
                                  * priority (int): the queuing system priority assigned to a given Python function to
                                                    influence the scheduling.
                                  * slurm_cmd_args (list): Additional command line arguments for the srun call.
            pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None
            flux_executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
            flux_executor_nesting (bool): Provide hierarchically nested Flux job scheduler inside the submitted function.
            flux_log_files (bool, optional): Write flux stdout and stderr files. Defaults to False.
            hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
            block_allocation (boolean): To accelerate the submission of a series of python functions with the same
                                        resource requirements, executorlib supports block allocation. In this case all
                                        resources have to be defined on the executor, rather than during the submission
                                        of the individual function.
            init_function (None): optional function to preset arguments for functions which are submitted later
            disable_dependencies (boolean): Disable resolving future objects during the submission.
            refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
            plot_dependency_graph (bool): Plot the dependencies of multiple future objects without executing them. For
                                          debugging purposes and to get an overview of the specified dependencies.
            plot_dependency_graph_filename (str): Name of the file to store the plotted graph in.
            export_workflow_filename (str): Name of the file to store the exported workflow graph in.
            log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.
            wait (bool): Whether to wait for the completion of all tasks before shutting down the executor.
            validator (callable): A function to validate the resource_dict.
            restart_limit (int): The maximum number of restarting worker processes.
            openmpi_oversubscribe (bool): adds the `--oversubscribe` command flag (OpenMPI and SLURM) - default False

        """
        default_resource_dict: dict = {
            "cores": 1,
            "threads_per_core": 1,
            "gpus_per_core": 0,
            "cwd": None,
            "openmpi_oversubscribe": openmpi_oversubscribe,
            "slurm_cmd_args": [],
        }
        if resource_dict is None:
            resource_dict = {}
        validate_resource_dict(resource_dict=resource_dict)
        resource_dict.update(
            {k: v for k, v in default_resource_dict.items() if k not in resource_dict}
        )
        check_restart_limit(
            restart_limit=restart_limit, block_allocation=block_allocation
        )
        if not disable_dependencies:
            super().__init__(
                executor=DependencyTaskScheduler(
                    executor=create_flux_executor(
                        max_workers=max_workers,
                        cache_directory=cache_directory,
                        max_cores=max_cores,
                        executor_kwargs=resource_dict,
                        pmi_mode=pmi_mode,
                        flux_executor=flux_executor,
                        flux_executor_nesting=flux_executor_nesting,
                        flux_log_files=flux_log_files,
                        hostname_localhost=hostname_localhost,
                        block_allocation=block_allocation,
                        init_function=init_function,
                        log_obj_size=log_obj_size,
                        wait=wait,
                        restart_limit=restart_limit,
                    ),
                    max_cores=max_cores,
                    refresh_rate=refresh_rate,
                    plot_dependency_graph=plot_dependency_graph,
                    plot_dependency_graph_filename=plot_dependency_graph_filename,
                    export_workflow_filename=export_workflow_filename,
                    validator=validate_resource_dict,
                )
            )
        else:
            check_plot_dependency_graph(plot_dependency_graph=plot_dependency_graph)
            check_refresh_rate(refresh_rate=refresh_rate)
            super().__init__(
                executor=create_flux_executor(
                    max_workers=max_workers,
                    cache_directory=cache_directory,
                    max_cores=max_cores,
                    executor_kwargs=resource_dict,
                    pmi_mode=pmi_mode,
                    flux_executor=flux_executor,
                    flux_executor_nesting=flux_executor_nesting,
                    flux_log_files=flux_log_files,
                    hostname_localhost=hostname_localhost,
                    block_allocation=block_allocation,
                    init_function=init_function,
                    log_obj_size=log_obj_size,
                    wait=wait,
                    validator=validate_resource_dict,
                    restart_limit=restart_limit,
                )
            )


class FluxClusterExecutor(BaseExecutor):
    """
    The executorlib.FluxClusterExecutor leverages either the message passing interface (MPI), the SLURM workload manager
    or preferable the flux framework for distributing python functions within a given resource allocation. In contrast
    to the mpi4py.futures.MPIPoolExecutor the executorlib.FluxClusterExecutor can be executed in a serial python process
    and does not require the python script to be executed with MPI. It is even possible to execute the
    executorlib.FluxClusterExecutor directly in an interactive Jupyter notebook.

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        max_cores (int): defines the number cores which can be used in parallel
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              * cores (int): number of MPI cores to be used for each function call
                              * threads_per_core (int): number of OpenMP threads to be used for each function call
                              * gpus_per_core (int): number of GPUs per worker - defaults to 0
                              * cwd (str): current working directory where the parallel python task is executed
                              * cache_key (str): Rather than using the internal hashing of executorlib the user can
                                                 provide an external cache_key to identify tasks on the file system.
                              * num_nodes (int): number of compute nodes used for the evaluation of the Python function.
                              * exclusive (bool): boolean flag to reserve exclusive access to selected compute nodes -
                                                  do not allow other tasks to use the same compute node.
                              * error_log_file (str): path to the error log file, primarily used to merge the log of
                                                      multiple tasks in one file.
                              * run_time_limit (int): the maximum time the execution of the submitted Python function is
                                                      allowed to take in seconds.
                              * priority (int): the queuing system priority assigned to a given Python function to
                                                influence the scheduling.
                              *`slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
        pysqa_config_directory (str, optional): path to the pysqa config directory (only for pysqa based backend).
        pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        block_allocation (boolean): To accelerate the submission of a series of python functions with the same resource
                                    requirements, executorlib supports block allocation. In this case all resources have
                                    to be defined on the executor, rather than during the submission of the individual
                                    function.
        init_function (None): optional function to preset arguments for functions which are submitted later
        disable_dependencies (boolean): Disable resolving future objects during the submission.
        refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
        plot_dependency_graph (bool): Plot the dependencies of multiple future objects without executing them. For
                                      debugging purposes and to get an overview of the specified dependencies.
        plot_dependency_graph_filename (str): Name of the file to store the plotted graph in.
        export_workflow_filename (str): Name of the file to store the exported workflow graph in.
        log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.
        wait (bool): Whether to wait for the completion of all tasks before shutting down the executor.
        openmpi_oversubscribe (bool): adds the `--oversubscribe` command flag (OpenMPI and SLURM) - default False

    Examples:
        ```
        >>> import numpy as np
        >>> from executorlib import FluxClusterExecutor
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
        >>> with FluxClusterExecutor(max_workers=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        cache_directory: Optional[str] = None,
        max_cores: Optional[int] = None,
        resource_dict: Optional[dict] = None,
        pysqa_config_directory: Optional[str] = None,
        pmi_mode: Optional[str] = None,
        hostname_localhost: Optional[bool] = None,
        block_allocation: bool = False,
        init_function: Optional[Callable] = None,
        disable_dependencies: bool = False,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        plot_dependency_graph_filename: Optional[str] = None,
        export_workflow_filename: Optional[str] = None,
        log_obj_size: bool = False,
        wait: bool = True,
        openmpi_oversubscribe: bool = False,
    ):
        """
        The executorlib.FluxClusterExecutor leverages either the message passing interface (MPI), the SLURM workload
        manager or preferable the flux framework for distributing python functions within a given resource allocation.
        In contrast to the mpi4py.futures.MPIPoolExecutor the executorlib.FluxClusterExecutor can be executed in a
        serial python process and does not require the python script to be executed with MPI. It is even possible to
        execute the executorlib.FluxClusterExecutor directly in an interactive Jupyter notebook.

        Args:
            max_workers (int): for backwards compatibility with the standard library, max_workers also defines the
                               number of cores which can be used in parallel - just like the max_cores parameter. Using
                               max_cores is recommended, as computers have a limited number of compute cores.
            cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
            max_cores (int): defines the number cores which can be used in parallel
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                                  * cores (int): number of MPI cores to be used for each function call
                                  * threads_per_core (int): number of OpenMP threads to be used for each function call
                                  * gpus_per_core (int): number of GPUs per worker - defaults to 0
                                  * cwd (str): current working directory where the parallel python task is executed
                                  * cache_key (str): Rather than using the internal hashing of executorlib the user can
                                                      provide an external cache_key to identify tasks on the file system.
                                  * num_nodes (int): number of compute nodes used for the evaluation of the Python
                                                     function.
                                  * exclusive (bool): boolean flag to reserve exclusive access to selected compute nodes
                                                      - do not allow other tasks to use the same compute node.
                                  * error_log_file (str): path to the error log file, primarily used to merge the log of
                                                          multiple tasks in one file.
                                  * run_time_limit (int): the maximum time the execution of the submitted Python
                                                        function is allowed to take in seconds.
                                  * priority (int): the queuing system priority assigned to a given Python function to
                                                    influence the scheduling.
                                  * slurm_cmd_args (list): Additional command line arguments for the srun call.
            pysqa_config_directory (str, optional): path to the pysqa config directory (only for pysqa based backend).
            pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None
            hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
            block_allocation (boolean): To accelerate the submission of a series of python functions with the same
                                        resource requirements, executorlib supports block allocation. In this case all
                                        resources have to be defined on the executor, rather than during the submission
                                        of the individual function.
            init_function (None): optional function to preset arguments for functions which are submitted later
            disable_dependencies (boolean): Disable resolving future objects during the submission.
            refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
            plot_dependency_graph (bool): Plot the dependencies of multiple future objects without executing them. For
                                          debugging purposes and to get an overview of the specified dependencies.
            plot_dependency_graph_filename (str): Name of the file to store the plotted graph in.
            export_workflow_filename (str): Name of the file to store the exported workflow graph in.
            log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.
            wait (bool): Whether to wait for the completion of all tasks before shutting down the executor.
            openmpi_oversubscribe (bool): adds the `--oversubscribe` command flag (OpenMPI and SLURM) - default False

        """
        default_resource_dict: dict = {
            "cores": 1,
            "threads_per_core": 1,
            "gpus_per_core": 0,
            "cwd": None,
            "openmpi_oversubscribe": openmpi_oversubscribe,
            "slurm_cmd_args": [],
            "run_time_limit": None,
        }
        if resource_dict is None:
            resource_dict = {}
        validate_resource_dict_with_optional_keys(resource_dict=resource_dict)
        resource_dict.update(
            {k: v for k, v in default_resource_dict.items() if k not in resource_dict}
        )
        check_log_obj_size(log_obj_size=log_obj_size)
        if not plot_dependency_graph:
            import pysqa  # noqa

            if block_allocation:
                from executorlib.task_scheduler.interactive.spawner_pysqa import (
                    create_pysqa_block_allocation_scheduler,
                )

                super().__init__(
                    executor=create_pysqa_block_allocation_scheduler(
                        max_cores=max_cores,
                        cache_directory=cache_directory,
                        hostname_localhost=hostname_localhost,
                        log_obj_size=log_obj_size,
                        pmi_mode=pmi_mode,
                        init_function=init_function,
                        max_workers=max_workers,
                        executor_kwargs=resource_dict,
                        pysqa_config_directory=pysqa_config_directory,
                        backend="flux",
                        validator=validate_resource_dict_with_optional_keys,
                    )
                )
            else:
                from executorlib.task_scheduler.file.task_scheduler import (
                    create_file_executor,
                )

                super().__init__(
                    executor=create_file_executor(
                        max_workers=max_workers,
                        backend="flux",
                        max_cores=max_cores,
                        cache_directory=cache_directory,
                        executor_kwargs=resource_dict,
                        flux_executor=None,
                        pmi_mode=pmi_mode,
                        flux_executor_nesting=False,
                        flux_log_files=False,
                        pysqa_config_directory=pysqa_config_directory,
                        hostname_localhost=hostname_localhost,
                        block_allocation=block_allocation,
                        init_function=init_function,
                        disable_dependencies=disable_dependencies,
                        wait=wait,
                        refresh_rate=refresh_rate,
                        validator=validate_resource_dict_with_optional_keys,
                    )
                )
        else:
            super().__init__(
                executor=DependencyTaskScheduler(
                    executor=create_flux_executor(
                        max_workers=max_workers,
                        cache_directory=cache_directory,
                        max_cores=max_cores,
                        executor_kwargs=resource_dict,
                        pmi_mode=None,
                        flux_executor=None,
                        flux_executor_nesting=False,
                        flux_log_files=False,
                        hostname_localhost=hostname_localhost,
                        block_allocation=block_allocation,
                        init_function=init_function,
                    ),
                    max_cores=max_cores,
                    refresh_rate=refresh_rate,
                    plot_dependency_graph=plot_dependency_graph,
                    plot_dependency_graph_filename=plot_dependency_graph_filename,
                    export_workflow_filename=export_workflow_filename,
                    validator=validate_resource_dict,
                )
            )


def create_flux_executor(
    max_workers: Optional[int] = None,
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    executor_kwargs: Optional[dict] = None,
    pmi_mode: Optional[str] = None,
    flux_executor=None,
    flux_executor_nesting: bool = False,
    flux_log_files: bool = False,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
    log_obj_size: bool = False,
    wait: bool = True,
    validator: Callable = validate_resource_dict,
    restart_limit: int = 0,
) -> Union[OneProcessTaskScheduler, BlockAllocationTaskScheduler]:
    """
    Create a flux executor

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the
                           number of cores which can be used in parallel - just like the max_cores parameter. Using
                           max_cores is recommended, as computers have a limited number of compute cores.
        max_cores (int): defines the number cores which can be used in parallel
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        executor_kwargs (dict): A dictionary of arguments required by the executor. With the following keys:
                              * cores (int): number of MPI cores to be used for each function call
                              * threads_per_core (int): number of OpenMP threads to be used for each function call
                              * gpus_per_core (int): number of GPUs per worker - defaults to 0
                              * cwd (str): current working directory where the parallel python task is executed
                              * cache_key (str): Rather than using the internal hashing of executorlib the user can
                                                 provide an external cache_key to identify tasks on the file system.
                              * num_nodes (int): number of compute nodes used for the evaluation of the Python function.
                              * exclusive (bool): boolean flag to reserve exclusive access to selected compute nodes -
                                                  do not allow other tasks to use the same compute node.
                              * error_log_file (str): path to the error log file, primarily used to merge the log of
                                                      multiple tasks in one file.
                              * run_time_limit (int): the maximum time the execution of the submitted Python function is
                                                      allowed to take in seconds.
                              * priority (int): the queuing system priority assigned to a given Python function to
                                                influence the scheduling.
                              *`slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
        pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None
        flux_executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        flux_executor_nesting (bool): Provide hierarchically nested Flux job scheduler inside the submitted function.
        flux_log_files (bool, optional): Write flux stdout and stderr files. Defaults to False.
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                  context of an HPC cluster this essential to be able to communicate to an
                                  Executor running on a different compute node within the same allocation. And
                                  in principle any computer should be able to resolve that their own hostname
                                  points to the same address as localhost. Still MacOS >= 12 seems to disable
                                  this look up for security reasons. So on MacOS it is required to set this
                                  option to true
        block_allocation (boolean): To accelerate the submission of a series of python functions with the same
                                    resource requirements, executorlib supports block allocation. In this case all
                                    resources have to be defined on the executor, rather than during the submission
                                    of the individual function.
        init_function (None): optional function to preset arguments for functions which are submitted later
        log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.
        wait (bool): Whether to wait for the completion of all tasks before shutting down the executor.
        restart_limit (int): The maximum number of restarting worker processes.

    Returns:
        InteractiveStepExecutor/ InteractiveExecutor
    """
    from executorlib.task_scheduler.interactive.spawner_flux import (
        FluxPythonSpawner,
        validate_max_workers,
    )

    if executor_kwargs is None:
        executor_kwargs = {}
    cores_per_worker = executor_kwargs.get("cores", 1)
    executor_kwargs["cache_directory"] = cache_directory
    executor_kwargs["hostname_localhost"] = hostname_localhost
    executor_kwargs["log_obj_size"] = log_obj_size
    check_init_function(block_allocation=block_allocation, init_function=init_function)
    check_pmi(backend="flux_allocation", pmi=pmi_mode)
    check_oversubscribe(
        oversubscribe=executor_kwargs.get("openmpi_oversubscribe", False)
    )
    check_command_line_argument_lst(
        command_line_argument_lst=executor_kwargs.get("slurm_cmd_args", [])
    )
    check_wait_on_shutdown(wait_on_shutdown=wait)
    if "openmpi_oversubscribe" in executor_kwargs:
        del executor_kwargs["openmpi_oversubscribe"]
    if "slurm_cmd_args" in executor_kwargs:
        del executor_kwargs["slurm_cmd_args"]
    executor_kwargs["pmi_mode"] = pmi_mode
    executor_kwargs["flux_executor"] = flux_executor
    executor_kwargs["flux_executor_nesting"] = flux_executor_nesting
    executor_kwargs["flux_log_files"] = flux_log_files
    if block_allocation:
        executor_kwargs["init_function"] = init_function
        max_workers = validate_number_of_cores(
            max_cores=max_cores,
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
            set_local_cores=False,
        )
        validate_max_workers(
            max_workers=max_workers,
            cores=cores_per_worker,
            threads_per_core=executor_kwargs.get("threads_per_core", 1),
        )
        return BlockAllocationTaskScheduler(
            max_workers=max_workers,
            executor_kwargs=executor_kwargs,
            spawner=FluxPythonSpawner,
            validator=validator,
            restart_limit=restart_limit,
        )
    else:
        return OneProcessTaskScheduler(
            max_cores=max_cores,
            max_workers=max_workers,
            executor_kwargs=executor_kwargs,
            spawner=FluxPythonSpawner,
            validator=validator,
        )
