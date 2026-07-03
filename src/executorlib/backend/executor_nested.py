from concurrent.futures import Future
from typing import Callable, Optional

from executorlib.standalone.interactive.arguments import (
    get_future_objects_from_input,
)


class BackendExecutor:
    def __init__(self):
        self._tasks_dict = {}
        self._future_dict = {}

    @property
    def tasks(self):
        return self._tasks_dict

    def batched(
        self,
        iterable: list[Future],
        n: int,
    ) -> list[Future]:
        """
        Batch futures from the iterable into tuples of length n. The last batch may be shorter than n.

        Args:
            iterable (list): list of future objects to batch based on which future objects finish first
            n (int): badge size

        Returns:
            list[Future]: list of future objects one for each batch
        """
        raise NotImplementedError("The batched method is not implemented.")

    def map(
        self,
        fn: Callable,
        *iterables,
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a child process. This argument is only
                used by ProcessPoolExecutor; it is ignored by
                ThreadPoolExecutor.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        raise NotImplementedError("The map method is not implemented.")

    def submit(  # type: ignore
        self,
        fn: Callable,
        /,
        *args,
        resource_dict: Optional[dict] = None,
        **kwargs,
    ) -> Future:
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            fn (callable): function to submit for execution
            args: arguments for the submitted function
            kwargs: keyword arguments for the submitted function
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

        Returns:
            Future: A Future representing the given call.
        """
        if resource_dict is None:
            resource_dict = {}
        f: Future = Future()
        self._future_dict[f] = id(f)
        self._tasks_dict[id(f)] = {
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "resource_dict": resource_dict,
            "dependencies": [
                self._future_dict[future_dependency]
                for future_dependency in get_future_objects_from_input(
                    args=args, kwargs=kwargs
                )
            ],
        }
        return f
