from typing import Optional

from executorlib.interactive import create_executor
from executorlib.interactive.dependencies import ExecutorWithDependencies
from executorlib.shared.inputcheck import (
    check_plot_dependency_graph as _check_plot_dependency_graph,
)
from executorlib.shared.inputcheck import (
    check_refresh_rate as _check_refresh_rate,
)
from executorlib.shell.executor import SubprocessExecutor
from executorlib.shell.interactive import ShellExecutor

from ._version import get_versions

__version__ = get_versions()["version"]
__all__ = [
    SubprocessExecutor,
    ShellExecutor,
]


try:
    from executorlib.cache.executor import FileExecutor

    __all__ += [FileExecutor]
except ImportError:
    pass


class Executor:
    """
    The executorlib.Executor leverages either the message passing interface (MPI), the SLURM workload manager or
    preferable the flux framework for distributing python functions within a given resource allocation. In contrast to
    the mpi4py.futures.MPIPoolExecutor the executorlib.Executor can be executed in a serial python process and does not
    require the python script to be executed with MPI. It is even possible to execute the executorlib.Executor directly
    in an interactive Jupyter notebook.

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
        conda_environment_name (str): name of the conda environment to initialize
        conda_environment_path (str): path of the conda environment to initialize
        executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        backend (str): Switch between the different backends "flux", "local" or "slurm". Alternatively, when "auto"
                       is selected (the default) the available backend is determined automatically.
        block_allocation (boolean): To accelerate the submission of a series of python functions with the same resource
                                    requirements, executorlib supports block allocation. In this case all resources have
                                    to be defined on the executor, rather than during the submission of the individual
                                    function.
        init_function (None): optional function to preset arguments for functions which are submitted later
        command_line_argument_lst (list): Additional command line arguments for the srun call (SLURM only)
        pmi (str): PMI interface to use (OpenMPI v5 requires pmix) default is None (Flux only)
        disable_dependencies (boolean): Disable resolving future objects during the submission.
        refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
        plot_dependency_graph (bool): Plot the dependencies of multiple future objects without executing them. For
                                      debugging purposes and to get an overview of the specified dependencies.

    Examples:
        ```
        >>> import numpy as np
        >>> from executorlib import Executor
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
        max_workers: int = 1,
        max_cores: int = 1,
        cores_per_worker: int = 1,
        threads_per_core: int = 1,
        gpus_per_worker: int = 0,
        oversubscribe: bool = False,
        cwd: Optional[str] = None,
        conda_environment_name: Optional[str] = None,
        conda_environment_path: Optional[str] = None,
        executor=None,
        hostname_localhost: bool = False,
        backend: str = "auto",
        block_allocation: bool = True,
        init_function: Optional[callable] = None,
        command_line_argument_lst: list[str] = [],
        pmi: Optional[str] = None,
        disable_dependencies: bool = False,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
    ):
        # Use __new__() instead of __init__(). This function is only implemented to enable auto-completion.
        pass

    def __new__(
        cls,
        max_workers: int = 1,
        max_cores: int = 1,
        cores_per_worker: int = 1,
        threads_per_core: int = 1,
        gpus_per_worker: int = 0,
        oversubscribe: bool = False,
        cwd: Optional[str] = None,
        conda_environment_name: Optional[str] = None,
        conda_environment_path: Optional[str] = None,
        executor=None,
        hostname_localhost: bool = False,
        backend: str = "auto",
        block_allocation: bool = False,
        init_function: Optional[callable] = None,
        command_line_argument_lst: list[str] = [],
        pmi: Optional[str] = None,
        nested_flux_executor: bool = False,
        disable_dependencies: bool = False,
        refresh_rate: float = 0.01,
        plot_dependency_graph: bool = False,
    ):
        """
        Instead of returning a executorlib.Executor object this function returns either a executorlib.mpi.PyMPIExecutor,
        executorlib.slurm.PySlurmExecutor or executorlib.flux.PyFluxExecutor depending on which backend is available. The
        executorlib.flux.PyFluxExecutor is the preferred choice while the executorlib.mpi.PyMPIExecutor is primarily used
        for development and testing. The executorlib.flux.PyFluxExecutor requires flux-core from the flux-framework to be
        installed and in addition flux-sched to enable GPU scheduling. Finally, the executorlib.slurm.PySlurmExecutor
        requires the SLURM workload manager to be installed on the system.

        Args:
            max_workers (int): for backwards compatibility with the standard library, max_workers also defines the
                               number of cores which can be used in parallel - just like the max_cores parameter. Using
                               max_cores is recommended, as computers have a limited number of compute cores.
            max_cores (int): defines the number cores which can be used in parallel
            cores_per_worker (int): number of MPI cores to be used for each function call
            threads_per_core (int): number of OpenMP threads to be used for each function call
            gpus_per_worker (int): number of GPUs per worker - defaults to 0
            oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and SLURM only) - default False
            cwd (str/None): current working directory where the parallel python task is executed
            conda_environment_name (str): name of the conda environment to initialize
            conda_environment_path (str): path of the conda environment to initialize
            executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
            hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
            backend (str): Switch between the different backends "flux", "local" or "slurm". Alternatively, when "auto"
                           is selected (the default) the available backend is determined automatically.
            block_allocation (boolean): To accelerate the submission of a series of python functions with the same
                                        resource requirements, executorlib supports block allocation. In this case all
                                        resources have to be defined on the executor, rather than during the submission
                                        of the individual function.
            init_function (None): optional function to preset arguments for functions which are submitted later
            command_line_argument_lst (list): Additional command line arguments for the srun call (SLURM only)
            pmi (str): PMI interface to use (OpenMPI v5 requires pmix) default is None (Flux only)
            nested_flux_executor (bool): Provide hierarchically nested Flux job scheduler inside the submitted function.
            disable_dependencies (boolean): Disable resolving future objects during the submission.
            refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
            plot_dependency_graph (bool): Plot the dependencies of multiple future objects without executing them. For
                                          debugging purposes and to get an overview of the specified dependencies.

        """
        if not disable_dependencies:
            return ExecutorWithDependencies(
                max_workers=max_workers,
                max_cores=max_cores,
                cores_per_worker=cores_per_worker,
                threads_per_core=threads_per_core,
                gpus_per_worker=gpus_per_worker,
                oversubscribe=oversubscribe,
                cwd=cwd,
                conda_environment_name=conda_environment_name,
                conda_environment_path=conda_environment_path,
                executor=executor,
                hostname_localhost=hostname_localhost,
                backend=backend,
                block_allocation=block_allocation,
                init_function=init_function,
                command_line_argument_lst=command_line_argument_lst,
                pmi=pmi,
                nested_flux_executor=nested_flux_executor,
                refresh_rate=refresh_rate,
                plot_dependency_graph=plot_dependency_graph,
            )
        else:
            _check_plot_dependency_graph(plot_dependency_graph=plot_dependency_graph)
            _check_refresh_rate(refresh_rate=refresh_rate)
            return create_executor(
                max_workers=max_workers,
                max_cores=max_cores,
                cores_per_worker=cores_per_worker,
                threads_per_core=threads_per_core,
                gpus_per_worker=gpus_per_worker,
                oversubscribe=oversubscribe,
                cwd=cwd,
                conda_environment_name=conda_environment_name,
                conda_environment_path=conda_environment_path,
                executor=executor,
                hostname_localhost=hostname_localhost,
                backend=backend,
                block_allocation=block_allocation,
                init_function=init_function,
                command_line_argument_lst=command_line_argument_lst,
                pmi=pmi,
                nested_flux_executor=nested_flux_executor,
            )
