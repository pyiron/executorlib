from pympipool.shared.executorbase import (
    ExecutorBase,
    executor_broker,
    get_executor_dict,
)
from pympipool.shared.thread import RaisingThread
from pympipool.flux.fluxtask import FluxSingleTaskExecutor


class FluxExecutor(ExecutorBase):
    def __init__(
        self,
        max_workers,
        cores_per_worker=1,
        threads_per_core=1,
        gpus_per_worker=0,
        init_function=None,
        cwd=None,
        sleep_interval=0.1,
        executor=None,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=_flux_executor_broker,
            kwargs={
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "cores_per_worker": cores_per_worker,
                "threads_per_core": threads_per_core,
                "gpus_per_worker": gpus_per_worker,
                "init_function": init_function,
                "cwd": cwd,
                "sleep_interval": sleep_interval,
                "executor": executor,
            },
        )
        self._process.start()


def _flux_executor_broker(
    future_queue,
    max_workers,
    cores_per_worker=1,
    threads_per_core=1,
    gpus_per_worker=0,
    init_function=None,
    cwd=None,
    sleep_interval=0.1,
    executor=None,
):
    executor_broker(
        future_queue=future_queue,
        meta_future_lst=get_executor_dict(
            max_workers=max_workers,
            executor_class=FluxSingleTaskExecutor,
            cores=cores_per_worker,
            threads_per_core=threads_per_core,
            gpus_per_task=int(gpus_per_worker / cores_per_worker),
            init_function=init_function,
            cwd=cwd,
            executor=executor,
        ),
        sleep_interval=sleep_interval,
    )
