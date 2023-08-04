import queue
from time import sleep

from concurrent.futures import as_completed, Future

from pympipool.shared.base import SingleTaskExecutor


def executor_broker(
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
    meta_future_lst = _get_executor_list(
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        threads_per_core=threads_per_core,
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
            if execute_task_dict(task_dict=task_dict, meta_future_lst=meta_future_lst):
                future_queue.task_done()
            else:
                future_queue.task_done()
                break


def execute_task_dict(task_dict, meta_future_lst):
    if "fn" in task_dict.keys():
        meta_future = next(as_completed(meta_future_lst.keys()))
        executor = meta_future_lst.pop(meta_future)
        executor.future_queue.put(task_dict)
        meta_future_lst[task_dict["future"]] = executor
        return True
    elif "shutdown" in task_dict.keys() and task_dict["shutdown"]:
        for executor in meta_future_lst.values():
            executor.shutdown(wait=task_dict["wait"])
        return False
    else:
        raise ValueError("Unrecognized Task in task_dict: ", task_dict)


def get_future_done():
    f = Future()
    f.set_result(True)
    return f


def _get_executor_list(
    max_workers,
    cores_per_worker=1,
    threads_per_core=1,
    gpus_per_worker=0,
    init_function=None,
    cwd=None,
    executor=None,
):
    return {
        get_future_done(): SingleTaskExecutor(
            cores=cores_per_worker,
            threads_per_core=threads_per_core,
            gpus_per_task=int(gpus_per_worker / cores_per_worker),
            init_function=init_function,
            cwd=cwd,
            executor=executor,
        )
        for _ in range(max_workers)
    }
