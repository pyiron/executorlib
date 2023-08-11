from pympipool.shared.executorbase import (
    ExecutorBase,
    executor_broker,
    get_executor_dict,
)
from pympipool.shared.thread import RaisingThread
from pympipool.legacy.legacytask import LegacySingleTaskExecutor


class HPCExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers,
        cores_per_worker=1,
        gpus_per_worker=0,
        oversubscribe=False,
        enable_flux_backend=False,
        enable_slurm_backend=False,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=_mpi_executor_broker,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "gpus_per_worker": gpus_per_worker,
                "oversubscribe": oversubscribe,
                "init_function": init_function,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
                "enable_flux_backend": enable_flux_backend,
                "enable_slurm_backend": enable_slurm_backend,
                "queue_adapter": queue_adapter,
                "queue_adapter_kwargs": queue_adapter_kwargs,
            },
        )
        self._process.start()


def _mpi_executor_broker(
    future_queue,
    max_workers,
    cores_per_worker=1,
    gpus_per_worker=0,
    oversubscribe=False,
    init_function=None,
    cwd=None,
    sleep_interval=0.1,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    queue_adapter=None,
    queue_adapter_kwargs=None,
):
    executor_broker(
        future_queue=future_queue,
        meta_future_lst=get_executor_dict(
            max_workers=max_workers,
            executor_class=LegacySingleTaskExecutor,
            cores=cores_per_worker,
            gpus_per_task=int(gpus_per_worker / cores_per_worker),
            oversubscribe=oversubscribe,
            init_function=init_function,
            cwd=cwd,
            enable_flux_backend=enable_flux_backend,
            enable_slurm_backend=enable_slurm_backend,
            queue_adapter=queue_adapter,
            queue_adapter_kwargs=queue_adapter_kwargs,
        ),
        sleep_interval=sleep_interval,
    )
