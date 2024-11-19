from concurrent.futures import Future
from typing import Any, Callable, Dict, Optional

from executorlib.base.executor import ExecutorBase
from executorlib.interactive.shared import (
    InteractiveExecutor,
    InteractiveStepExecutor,
    execute_tasks_with_dependencies,
)
from executorlib.standalone.inputcheck import (
    check_command_line_argument_lst,
    check_executor,
    check_gpus_per_worker,
    check_init_function,
    check_nested_flux_executor,
    check_oversubscribe,
    check_pmi,
    validate_number_of_cores,
)
from executorlib.standalone.interactive.spawner import (
    MpiExecSpawner,
    SrunSpawner,
)
from executorlib.standalone.plot import (
    draw,
    generate_nodes_and_edges,
    generate_task_hash,
)
from executorlib.standalone.thread import RaisingThread

try:  # The PyFluxExecutor requires flux-base to be installed.
    from executorlib.interactive.flux import FluxPythonSpawner
except ImportError:
    pass


class ExecutorWithDependencies(ExecutorBase):
    """
    ExecutorWithDependencies is a class that extends ExecutorBase and provides functionality for executing tasks with
    dependencies.

    Args:
        refresh_rate (float, optional): The refresh rate for updating the executor queue. Defaults to 0.01.
        plot_dependency_graph (bool, optional): Whether to generate and plot the dependency graph. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _future_hash_dict (Dict[str, Future]): A dictionary mapping task hash to future object.
        _task_hash_dict (Dict[str, Dict]): A dictionary mapping task hash to task dictionary.
        _generate_dependency_graph (bool): Whether to generate the dependency graph.

    """

    def __init__(
        self,
        *args: Any,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_cores=kwargs.get("max_cores", None))
        executor = create_executor(*args, **kwargs)
        self._set_process(
            RaisingThread(
                target=execute_tasks_with_dependencies,
                kwargs={
                    # Executor Arguments
                    "future_queue": self._future_queue,
                    "executor_queue": executor._future_queue,
                    "executor": executor,
                    "refresh_rate": refresh_rate,
                },
            )
        )
        self._future_hash_dict = {}
        self._task_hash_dict = {}
        self._generate_dependency_graph = plot_dependency_graph

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        resource_dict: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Future:
        """
        Submits a task to the executor.

        Args:
            fn (callable): The function to be executed.
            *args: Variable length argument list.
            resource_dict (dict, optional): A dictionary of resources required by the task. Defaults to {}.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Future: A future object representing the result of the task.

        """
        if not self._generate_dependency_graph:
            f = super().submit(fn, *args, resource_dict=resource_dict, **kwargs)
        else:
            f = Future()
            f.set_result(None)
            task_dict = {
                "fn": fn,
                "args": args,
                "kwargs": kwargs,
                "future": f,
                "resource_dict": resource_dict,
            }
            task_hash = generate_task_hash(
                task_dict=task_dict,
                future_hash_inverse_dict={
                    v: k for k, v in self._future_hash_dict.items()
                },
            )
            self._future_hash_dict[task_hash] = f
            self._task_hash_dict[task_hash] = task_dict
        return f

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """
        Exit method called when exiting the context manager.

        Args:
            exc_type: The type of the exception.
            exc_val: The exception instance.
            exc_tb: The traceback object.

        """
        super().__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)
        if self._generate_dependency_graph:
            node_lst, edge_lst = generate_nodes_and_edges(
                task_hash_dict=self._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in self._future_hash_dict.items()
                },
            )
            return draw(node_lst=node_lst, edge_lst=edge_lst)


def create_executor(
    max_workers: Optional[int] = None,
    backend: str = "local",
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    resource_dict: Optional[dict] = None,
    flux_executor=None,
    flux_executor_pmi_mode: Optional[str] = None,
    flux_executor_nesting: bool = False,
    hostname_localhost: Optional[bool] = None,
    block_allocation: bool = False,
    init_function: Optional[callable] = None,
):
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
    check_init_function(block_allocation=block_allocation, init_function=init_function)
    if flux_executor is not None and backend != "flux_allocation":
        backend = "flux_allocation"
    check_pmi(backend=backend, pmi=flux_executor_pmi_mode)
    cores_per_worker = resource_dict["cores"]
    resource_dict["cache_directory"] = cache_directory
    resource_dict["hostname_localhost"] = hostname_localhost
    if backend == "flux_allocation":
        check_oversubscribe(oversubscribe=resource_dict["openmpi_oversubscribe"])
        check_command_line_argument_lst(
            command_line_argument_lst=resource_dict["slurm_cmd_args"]
        )
        del resource_dict["openmpi_oversubscribe"]
        del resource_dict["slurm_cmd_args"]
        resource_dict["flux_executor"] = flux_executor
        resource_dict["flux_executor_pmi_mode"] = flux_executor_pmi_mode
        resource_dict["flux_executor_nesting"] = flux_executor_nesting
        if block_allocation:
            resource_dict["init_function"] = init_function
            return InteractiveExecutor(
                max_workers=validate_number_of_cores(
                    max_cores=max_cores,
                    max_workers=max_workers,
                    cores_per_worker=cores_per_worker,
                    set_local_cores=False,
                ),
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
    elif backend == "slurm_allocation":
        check_executor(executor=flux_executor)
        check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
        if block_allocation:
            resource_dict["init_function"] = init_function
            return InteractiveExecutor(
                max_workers=validate_number_of_cores(
                    max_cores=max_cores,
                    max_workers=max_workers,
                    cores_per_worker=cores_per_worker,
                    set_local_cores=False,
                ),
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
    elif backend == "local":
        check_executor(executor=flux_executor)
        check_nested_flux_executor(nested_flux_executor=flux_executor_nesting)
        check_gpus_per_worker(gpus_per_worker=resource_dict["gpus_per_core"])
        check_command_line_argument_lst(
            command_line_argument_lst=resource_dict["slurm_cmd_args"]
        )
        del resource_dict["threads_per_core"]
        del resource_dict["gpus_per_core"]
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
    else:
        raise ValueError(
            "The supported backends are slurm_allocation, slurm_submission, flux_allocation, flux_submission and local."
        )
