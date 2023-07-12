from threading import Thread

from pympipool.share.executor import ExecutorBase
from pympipool.share.communication import SocketInterface
from pympipool.share.serial import (
    execute_serial_tasks_loop,
    get_parallel_subprocess_command,
    execute_parallel_tasks_loop,
)


def execute_parallel_tasks(
    future_queue,
    cores,
    oversubscribe=False,
    enable_flux_backend=False,
    cwd=None,
    queue_adapter=None,
    queue_adapter_kwargs=None,
):
    interface = SocketInterface()
    command_lst = get_parallel_subprocess_command(
        port_selected=interface.bind_to_random_port(),
        cores=cores,
        cores_per_task=1,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_mpi4py_backend=False,
    )
    if queue_adapter is not None:
        queue_adapter.submit(
            working_directory=cwd,
            cores=cores,
            command=' '.join(command_lst),
            **queue_adapter_kwargs
        )
    else:
        interface.bootup(command_lst=command_lst, cwd=cwd)
    execute_parallel_tasks_loop(interface=interface, future_queue=future_queue)


def execute_serial_tasks(
    future_queue,
    cores,
    oversubscribe=False,
    enable_flux_backend=False,
    cwd=None,
    sleep_interval=0.1,
    queue_adapter=None,
    queue_adapter_kwargs=None,
):
    future_dict = {}
    interface = SocketInterface()
    command_lst = get_parallel_subprocess_command(
        port_selected=interface.bind_to_random_port(),
        cores=cores,
        cores_per_task=1,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_mpi4py_backend=True,
    )
    if queue_adapter is not None:
        queue_adapter.submit(
            working_directory=cwd,
            cores=cores,
            command=' '.join(command_lst),
            **queue_adapter_kwargs
        )
    else:
        interface.bootup(command_lst=command_lst, cwd=cwd)
    execute_serial_tasks_loop(
        interface=interface,
        future_queue=future_queue,
        future_dict=future_dict,
        sleep_interval=sleep_interval,
    )


class QueueExecutor(ExecutorBase):
    """
    The pympipool.Executor behaves like the concurrent.futures.Executor but it uses mpi4py to execute parallel tasks.
    In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.Executor can be executed in a serial python process
    and does not require the python script to be executed with MPI. Still internally the pympipool.Executor uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        enable_flux_backend (bool): use the flux-framework as backend rather than just calling mpiexec
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed

    Simple example:
        ```
        import numpy as np
        from pympipool import Executor

        def calc(i, j, k):
            from mpi4py import MPI
            size = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()
            return np.array([i, j, k]), size, rank

        def init_k():
            return {"k": 3}

        with Executor(cores=2, init_function=init_k) as p:
            fs = p.submit(calc, 2, j=4)
            print(fs.result())

        >>> [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]
        ```
    """

    def __init__(
        self,
        cores,
        oversubscribe=False,
        enable_flux_backend=False,
        init_function=None,
        cwd=None,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__()
        self._process = Thread(
            target=execute_parallel_tasks,
            args=(
                self._future_queue,
                cores,
                oversubscribe,
                enable_flux_backend,
                cwd,
                queue_adapter,
                queue_adapter_kwargs
            ),
        )
        self._process.start()
        if init_function is not None:
            self._future_queue.put(
                {"init": True, "fn": init_function, "args": (), "kwargs": {}}
            )


class PoolExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers=1,
        oversubscribe=False,
        enable_flux_backend=False,
        cwd=None,
        sleep_interval=0.1,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__()
        self._process = Thread(
            target=execute_serial_tasks,
            args=(
                self._future_queue,
                max_workers,
                oversubscribe,
                enable_flux_backend,
                cwd,
                sleep_interval,
                queue_adapter,
                queue_adapter_kwargs,
            ),
        )
        self._process.start()
