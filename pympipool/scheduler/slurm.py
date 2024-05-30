from pympipool.shared.executorbase import (
    execute_parallel_tasks,
    execute_separate_tasks,
    ExecutorBroker,
    ExecutorSteps,
)
from pympipool.shared.interface import SrunInterface
from pympipool.shared.thread import RaisingThread


class PySlurmExecutor(ExecutorBroker):
    """
    The pympipool.slurm.PySlurmExecutor leverages the srun command to distribute python tasks within a SLURM queuing
    system allocation. In analogy to the pympipool.flux.PyFluxExecutor it provides the option to specify the number of
    threads per worker as well as the number of GPUs per worker in addition to specifying the number of cores per
    worker.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel
        executor_kwargs (dict): keyword arguments for the executor

    Examples:

        >>> import numpy as np
        >>> from pympipool.scheduler.slurm import PySlurmExecutor
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
        >>> with PySlurmExecutor(max_workers=2, executor_kwargs={"init_function": init_k}) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
    """

    def __init__(
        self,
        max_workers: int = 1,
        executor_kwargs: dict = {},
    ):
        super().__init__()
        executor_kwargs["future_queue"] = self._future_queue
        executor_kwargs["interface_class"] = SrunInterface
        self._set_process(
            process=[
                RaisingThread(
                    target=execute_parallel_tasks,
                    kwargs=executor_kwargs,
                )
                for _ in range(max_workers)
            ],
        )


class PySlurmStepExecutor(ExecutorSteps):
    """
    The pympipool.slurm.PySlurmStepExecutor leverages the srun command to distribute python tasks within a SLURM queuing
    system allocation. In analogy to the pympipool.flux.PyFluxExecutor it provides the option to specify the number of
    threads per worker as well as the number of GPUs per worker in addition to specifying the number of cores per
    worker.

    Args:
        max_cores (int): defines the number cores which can be used in parallel
        executor_kwargs (dict): keyword arguments for the executor

    Examples:

        >>> import numpy as np
        >>> from pympipool.scheduler.slurm import PySlurmStepExecutor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> with PySlurmStepExecutor(max_cores=2) as p:
        >>>     fs = p.submit(calc, 2, j=4, k=3, resource_dict={"cores": 2})
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
    """

    def __init__(
        self,
        max_cores: int = 1,
        executor_kwargs: dict = {},
    ):
        super().__init__()
        executor_kwargs["future_queue"] = self._future_queue
        executor_kwargs["interface_class"] = SrunInterface
        executor_kwargs["max_cores"] = max_cores
        self._set_process(
            RaisingThread(
                target=execute_separate_tasks,
                kwargs=executor_kwargs,
            )
        )
