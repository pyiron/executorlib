import os
import queue
import socket
import sys
from time import sleep

from pympipool.shared.broker import (
    MetaExecutorFuture,
    _get_future_done,
    _execute_task_dict,
)
from pympipool.interfaces.base import ExecutorBase
from pympipool.shared.thread import RaisingThread
from pympipool.shared.taskexecutor import (
    cloudpickle_register,
    _execute_parallel_tasks_loop,
)
from pympipool.shared.connections import FluxPythonInterface
from pympipool.shared.communication import SocketInterface


class SingleTaskExecutor(ExecutorBase):
    """
    The pympipool.Executor behaves like the concurrent.futures.Executor but it uses mpi4py to execute parallel tasks.
    In contrast to the mpi4py.futures.MPIPoolExecutor the pympipool.Executor can be executed in a serial python process
    and does not require the python script to be executed with MPI. Still internally the pympipool.Executor uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        enable_flux_backend (bool): use the flux-framework as backend rather than just calling mpiexec
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed
        queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
        queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter

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
        gpus_per_task=0,
        init_function=None,
        cwd=None,
        executor=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=execute_parallel_tasks,
            kwargs={
                "future_queue": self._future_queue,
                "cores": cores,
                "gpus_per_task": gpus_per_task,
                "cwd": cwd,
                "executor": executor,
            },
        )
        self._process.start()
        if init_function is not None:
            self._future_queue.put(
                {"init": True, "fn": init_function, "args": (), "kwargs": {}}
            )
        cloudpickle_register(ind=3)


class PyFluxExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers,
        cores_per_worker=1,
        gpus_per_worker=0,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
        executor=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "gpus_per_worker": gpus_per_worker,
                "init_function": init_function,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
                "executor": executor,
            },
        )
        self._process.start()


def execute_parallel_tasks(
    future_queue,
    cores,
    gpus_per_task=0,
    cwd=None,
    executor=None,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
       cwd (str/None): current working directory where the parallel python task is executed
       executor (flux.job.FluxExecutor/None): flux executor to submit tasks to - optional
    """
    command_lst = [
        sys.executable,
        os.path.abspath(os.path.join(__file__, "..", "..", "backend", "mpiexec.py")),
    ]
    interface = interface_bootup(
        command_lst=command_lst,
        cwd=cwd,
        cores=cores,
        gpus_per_core=gpus_per_task,
        executor=executor,
    )
    _execute_parallel_tasks_loop(interface=interface, future_queue=future_queue)


def interface_bootup(
    command_lst,
    cwd=None,
    cores=1,
    gpus_per_core=0,
    executor=None,
):
    command_lst += [
        "--host",
        socket.gethostname(),
    ]
    connections = FluxPythonInterface(
        cwd=cwd,
        cores=cores,
        gpus_per_core=gpus_per_core,
        oversubscribe=False,
        executor=executor,
    )
    interface = SocketInterface(interface=connections)
    command_lst += [
        "--zmqport",
        str(interface.bind_to_random_port()),
    ]
    interface.bootup(command_lst=command_lst)
    return interface


def executor_broker(
    future_queue,
    max_workers,
    cores_per_worker=1,
    gpus_per_worker=0,
    init_function=None,
    cwd=None,
    sleep_interval=0.1,
    executor=None,
):
    meta_future_lst = _get_executor_list(
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        gpus_per_worker=gpus_per_worker,
        init_function=init_function,
        cwd=cwd,
        executor=executor,
    )
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            sleep(sleep_interval)
        else:
            if _execute_task_dict(task_dict=task_dict, meta_future_lst=meta_future_lst):
                future_queue.task_done()
            else:
                future_queue.task_done()
                break


def _get_executor_list(
    max_workers,
    cores_per_worker=1,
    gpus_per_worker=0,
    init_function=None,
    cwd=None,
    executor=None,
):
    return [
        MetaExecutorFuture(
            future=_get_future_done(),
            executor=SingleTaskExecutor(
                cores=cores_per_worker,
                gpus_per_task=int(gpus_per_worker / cores_per_worker),
                init_function=init_function,
                cwd=cwd,
                executor=executor,
            ),
        )
        for _ in range(max_workers)
    ]
