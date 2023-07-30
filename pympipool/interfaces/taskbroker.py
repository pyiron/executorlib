from pympipool.interfaces.base import ExecutorBase
from pympipool.shared.thread import RaisingThread
from pympipool.shared.broker import executor_broker


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
            target=executor_broker,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "gpus_per_worker": gpus_per_worker,
                "oversubscribe": oversubscribe,
                "enable_flux_backend": enable_flux_backend,
                "enable_slurm_backend": enable_slurm_backend,
                "init_function": init_function,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
                "queue_adapter": queue_adapter,
                "queue_adapter_kwargs": queue_adapter_kwargs,
            },
        )
        self._process.start()
