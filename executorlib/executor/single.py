from typing import Callable, Optional, Union

from executorlib.executor.base import BaseExecutor
from executorlib.standalone.inputcheck import (
    check_command_line_argument_lst,
    check_gpus_per_worker,
    check_init_function,
    check_plot_dependency_graph,
    check_refresh_rate,
    validate_number_of_cores,
)
from executorlib.standalone.interactive.spawner import MpiExecSpawner
from executorlib.task_scheduler.interactive.blockallocation import (
    BlockAllocationTaskScheduler,
)
from executorlib.task_scheduler.interactive.dependency import DependencyTaskScheduler
from executorlib.task_scheduler.interactive.onetoone import OneProcessTaskScheduler


class SingleNodeExecutor(BaseExecutor):
    """
    The executorlib.SingleNodeExecutor leverages either the message passing interface (MPI), the SLURM workload manager
    or preferable the flux framework for distributing python functions within a given resource allocation. In contrast
    to the mpi4py.futures.MPIPoolExecutor the executorlib.SingleNodeExecutor can be executed in a serial python process
    and does not require the python script to be executed with MPI. It is even possible to execute the
    executorlib.SingleNodeExecutor directly in an interactive Jupyter notebook.

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        max_cores (int): defines the number cores which can be used in parallel
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - threads_per_core (int): number of OpenMP threads to be used for each function call
                              - gpus_per_core (int): number of GPUs per worker - defaults to 0
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and
                                                              SLURM only) - default False
                              - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
                              - error_log_file (str): Name of the error log file to use for storing exceptions raised
                                                      by the Python functions submitted to the Executor.
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
        log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.

    Examples:
        ```
        >>> import numpy as np
        >>> from executorlib import SingleNodeExecutor
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
        >>> with SingleNodeExecutor(max_workers=2, init_function=init_k) as p:
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
        hostname_localhost: Optional[bool] = None,
        block_allocation: bool = False,
        init_function: Optional[Callable] = None,
        disable_dependencies: bool = False,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        plot_dependency_graph_filename: Optional[str] = None,
        log_obj_size: bool = False,
    ):
        """
        The executorlib.SingleNodeExecutor leverages either the message passing interface (MPI), the SLURM workload
        manager or preferable the flux framework for distributing python functions within a given resource allocation.
        In contrast to the mpi4py.futures.MPIPoolExecutor the executorlib.SingleNodeExecutor can be executed in a serial
        python process and does not require the python script to be executed with MPI. It is even possible to execute
        the executorlib.SingleNodeExecutor directly in an interactive Jupyter notebook.

        Args:
            max_workers (int): for backwards compatibility with the standard library, max_workers also defines the
                               number of cores which can be used in parallel - just like the max_cores parameter. Using
                               max_cores is recommended, as computers have a limited number of compute cores.
            cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
            max_cores (int): defines the number cores which can be used in parallel
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                                  - cores (int): number of MPI cores to be used for each function call
                                  - threads_per_core (int): number of OpenMP threads to be used for each function call
                                  - gpus_per_core (int): number of GPUs per worker - defaults to 0
                                  - cwd (str/None): current working directory where the parallel python task is executed
                                  - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI
                                                                  and SLURM only) - default False
                                  - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM
                                                           only)
                                  - error_log_file (str): Name of the error log file to use for storing exceptions
                                                          raised by the Python functions submitted to the Executor.
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
            log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.

        """
        default_resource_dict: dict = {
            "cores": 1,
            "threads_per_core": 1,
            "gpus_per_core": 0,
            "cwd": None,
            "openmpi_oversubscribe": False,
            "slurm_cmd_args": [],
        }
        if resource_dict is None:
            resource_dict = {}
        resource_dict.update(
            {k: v for k, v in default_resource_dict.items() if k not in resource_dict}
        )
        if not disable_dependencies:
            super().__init__(
                executor=DependencyTaskScheduler(
                    executor=create_single_node_executor(
                        max_workers=max_workers,
                        cache_directory=cache_directory,
                        max_cores=max_cores,
                        resource_dict=resource_dict,
                        hostname_localhost=hostname_localhost,
                        block_allocation=block_allocation,
                        init_function=init_function,
                        log_obj_size=log_obj_size,
                    ),
                    max_cores=max_cores,
                    refresh_rate=refresh_rate,
                    plot_dependency_graph=plot_dependency_graph,
                    plot_dependency_graph_filename=plot_dependency_graph_filename,
                )
            )
        else:
            check_plot_dependency_graph(plot_dependency_graph=plot_dependency_graph)
            check_refresh_rate(refresh_rate=refresh_rate)
            super().__init__(
                executor=create_single_node_executor(
                    max_workers=max_workers,
                    cache_directory=cache_directory,
                    max_cores=max_cores,
                    resource_dict=resource_dict,
                    hostname_localhost=hostname_localhost,
                    block_allocation=block_allocation,
                    init_function=init_function,
                    log_obj_size=log_obj_size,
                )
            )


class TestClusterExecutor(BaseExecutor):
    """
    The executorlib.api.TestClusterExecutor is designed to test the file based communication used in the
    SlurmClusterExecutor and the FluxClusterExecutor locally. It is not recommended for production use, rather use the
    SingleNodeExecutor.

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        max_cores (int): defines the number cores which can be used in parallel
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - threads_per_core (int): number of OpenMP threads to be used for each function call
                              - gpus_per_core (int): number of GPUs per worker - defaults to 0
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - error_log_file (str): Name of the error log file to use for storing exceptions raised
                                                      by the Python functions submitted to the Executor.
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
        log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.

    Examples:
        ```
        >>> import numpy as np
        >>> from executorlib.api import TestClusterExecutor
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
        >>> with TestClusterExecutor(max_workers=2, init_function=init_k) as p:
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
        hostname_localhost: Optional[bool] = None,
        block_allocation: bool = False,
        init_function: Optional[Callable] = None,
        disable_dependencies: bool = False,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        plot_dependency_graph_filename: Optional[str] = None,
        log_obj_size: bool = False,
    ):
        """
        The executorlib.api.TestClusterExecutor is designed to test the file based communication used in the
        SlurmClusterExecutor and the FluxClusterExecutor locally. It is not recommended for production use, rather use
        the SingleNodeExecutor.

        Args:
            max_workers (int): for backwards compatibility with the standard library, max_workers also defines the
                               number of cores which can be used in parallel - just like the max_cores parameter. Using
                               max_cores is recommended, as computers have a limited number of compute cores.
            cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
            max_cores (int): defines the number cores which can be used in parallel
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                                  - cores (int): number of MPI cores to be used for each function call
                                  - threads_per_core (int): number of OpenMP threads to be used for each function call
                                  - gpus_per_core (int): number of GPUs per worker - defaults to 0
                                  - cwd (str/None): current working directory where the parallel python task is executed
                                  - error_log_file (str): Name of the error log file to use for storing exceptions
                                                          raised by the Python functions submitted to the Executor.
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
            log_obj_size (bool): Enable debug mode which reports the size of the communicated objects.

        """
        default_resource_dict: dict = {
            "cores": 1,
            "threads_per_core": 1,
            "gpus_per_core": 0,
            "cwd": None,
            "openmpi_oversubscribe": False,
        }
        if resource_dict is None:
            resource_dict = {}
        resource_dict.update(
            {k: v for k, v in default_resource_dict.items() if k not in resource_dict}
        )
        if not plot_dependency_graph:
            from executorlib.task_scheduler.file.subprocess_spawner import (
                execute_in_subprocess,
            )
            from executorlib.task_scheduler.file.task_scheduler import (
                create_file_executor,
            )

            super().__init__(
                executor=create_file_executor(
                    max_workers=max_workers,
                    backend=None,
                    max_cores=max_cores,
                    cache_directory=cache_directory,
                    resource_dict=resource_dict,
                    flux_executor=None,
                    pmi_mode=None,
                    flux_executor_nesting=False,
                    flux_log_files=False,
                    pysqa_config_directory=None,
                    hostname_localhost=hostname_localhost,
                    block_allocation=block_allocation,
                    init_function=init_function,
                    disable_dependencies=disable_dependencies,
                    execute_function=execute_in_subprocess,
                )
            )
        else:
            super().__init__(
                executor=DependencyTaskScheduler(
                    executor=create_single_node_executor(
                        max_workers=max_workers,
                        cache_directory=cache_directory,
                        max_cores=max_cores,
                        resource_dict=resource_dict,
                        hostname_localhost=hostname_localhost,
                        block_allocation=block_allocation,
                        init_function=init_function,
                        log_obj_size=log_obj_size,
                    ),
                    max_cores=max_cores,
                    refresh_rate=refresh_rate,
                    plot_dependency_graph=plot_dependency_graph,
                    plot_dependency_graph_filename=plot_dependency_graph_filename,
                )
            )


def create_single_node_executor(
    max_workers: Optional[int] = None,
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[Callable] = None,
    log_obj_size: bool = False,
) -> Union[OneProcessTaskScheduler, BlockAllocationTaskScheduler]:
    """
    Create a single node executor

    Args:
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the
                           number of cores which can be used in parallel - just like the max_cores parameter. Using
                           max_cores is recommended, as computers have a limited number of compute cores.
        max_cores (int): defines the number cores which can be used in parallel
        cache_directory (str, optional): The directory to store cache files. Defaults to "executorlib_cache".
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - threads_per_core (int): number of OpenMP threads to be used for each function call
                              - gpus_per_core (int): number of GPUs per worker - defaults to 0
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI
                                                              and SLURM only) - default False
                              - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM
                                                       only)
                              - error_log_file (str): Name of the error log file to use for storing exceptions raised
                                                      by the Python functions submitted to the Executor.
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

    Returns:
        InteractiveStepExecutor/ InteractiveExecutor
    """
    if resource_dict is None:
        resource_dict = {}
    cores_per_worker = resource_dict.get("cores", 1)
    resource_dict["cache_directory"] = cache_directory
    resource_dict["hostname_localhost"] = hostname_localhost
    resource_dict["log_obj_size"] = log_obj_size

    check_init_function(block_allocation=block_allocation, init_function=init_function)
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
        return BlockAllocationTaskScheduler(
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
        return OneProcessTaskScheduler(
            max_cores=max_cores,
            max_workers=max_workers,
            executor_kwargs=resource_dict,
            spawner=MpiExecSpawner,
        )
