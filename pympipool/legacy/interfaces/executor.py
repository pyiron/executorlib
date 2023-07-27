from pympipool.interfaces.taskexecutor import ExecutorBase
from pympipool.shared.thread import RaisingThread
from pympipool.legacy.shared.interface import execute_serial_tasks


class PoolExecutor(ExecutorBase):
    """
    To combine the functionality of the pympipool.Pool and the pympipool.Executor the pympipool.PoolExecutor again
    connects to the mpi4py.futures.MPIPoolExecutor. Still in contrast to the pympipool.Pool it does not implement the
    map() and starmap() functions but rather the submit() function based on the concurrent.futures.Executor interface.
    In this case the load balancing happens internally and the maximum number of workers max_workers defines the maximum
    number of parallel tasks. But only serial python tasks can be executed in contrast to the pympipool.Executor which
    can also execute MPI parallel python tasks.

    Args:
        max_workers (int): defines the total number of MPI ranks to use
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        enable_flux_backend (bool): use the flux-framework as backend rather than just calling mpiexec
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        cwd (str/None): current working directory where the parallel python task is executed
        sleep_interval (float):
        queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
        queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter

    Simple example:
        ```
        from pympipool import PoolExecutor

        def calc(i, j):
            return i + j

        with PoolExecutor(max_workers=2) as p:
            fs1 = p.submit(calc, 1, 2)
            fs2 = p.submit(calc, 3, 4)
            fs3 = p.submit(calc, 5, 6)
            fs4 = p.submit(calc, 7, 8)
            print(fs1.result(), fs2.result(), fs3.result(), fs4.result()
        ```
    """

    def __init__(
        self,
        max_workers=1,
        gpus_per_task=0,
        oversubscribe=False,
        enable_flux_backend=False,
        enable_slurm_backend=False,
        cwd=None,
        sleep_interval=0.1,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=execute_serial_tasks,
            kwargs={
                "future_queue": self._future_queue,
                "cores": max_workers,
                "gpus_per_task": gpus_per_task,
                "oversubscribe": oversubscribe,
                "enable_flux_backend": enable_flux_backend,
                "enable_slurm_backend": enable_slurm_backend,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
                "queue_adapter": queue_adapter,
                "queue_adapter_kwargs": queue_adapter_kwargs,
            },
        )
        self._process.start()
