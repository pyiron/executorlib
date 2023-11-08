from concurrent.futures import (
    as_completed,
    Executor as FutureExecutor,
    Future,
)
import inspect
import os
import queue
import sys
from time import sleep

import cloudpickle

from pympipool.shared.communication import interface_bootup


class ExecutorBase(FutureExecutor):
    def __init__(self):
        self._future_queue = queue.Queue()
        self._process = None

    @property
    def future_queue(self):
        return self._future_queue

    def submit(self, fn, *args, **kwargs):
        """Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Returns:
            A Future representing the given call.
        """
        f = Future()
        self._future_queue.put({"fn": fn, "args": args, "kwargs": kwargs, "future": f})
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                parallel_executors have been reclaimed.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        if cancel_futures:
            cancel_items_in_queue(que=self._future_queue)
        self._future_queue.put({"shutdown": True, "wait": wait})
        if wait:
            self._process.join()
            self._future_queue.join()
        self._process = None
        self._future_queue = None

    def __len__(self):
        return self._future_queue.qsize()

    def __del__(self):
        try:
            self.shutdown(wait=False)
        except (AttributeError, RuntimeError):
            pass

    def _set_init_function(self, init_function):
        if init_function is not None:
            self._future_queue.put(
                {"init": True, "fn": init_function, "args": (), "kwargs": {}}
            )


def cancel_items_in_queue(que):
    """
    Cancel items which are still waiting in the queue. If the executor is busy tasks remain in the queue, so the future
    objects have to be cancelled when the executor shuts down.

    Args:
        que (queue.Queue): Queue with task objects which should be executed
    """
    while True:
        try:
            item = que.get_nowait()
            if isinstance(item, dict) and "future" in item.keys():
                item["future"].cancel()
                que.task_done()
        except queue.Empty:
            break


def cloudpickle_register(ind=2):
    """
    Cloudpickle can either pickle by value or pickle by reference. The functions which are communicated have to
    be pickled by value rather than by reference, so the module which calls the map function is pickled by value.
    https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
    inspect can help to find the module which is calling pympipool
    https://docs.python.org/3/library/inspect.html
    to learn more about inspect another good read is:
    http://pymotw.com/2/inspect/index.html#module-inspect
    1 refers to 1 level higher than the map function

    Args:
        ind (int): index of the level at which pickle by value starts while for the rest pickle by reference is used
    """
    try:  # When executed in a jupyter notebook this can cause a ValueError - in this case we just ignore it.
        cloudpickle.register_pickle_by_value(inspect.getmodule(inspect.stack()[ind][0]))
    except IndexError:
        cloudpickle_register(ind=ind - 1)
    except ValueError:
        pass


def execute_parallel_tasks(
    future_queue,
    cores,
    interface_class,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       interface_class:
    """
    execute_parallel_tasks_loop(
        interface=interface_bootup(
            command_lst=_get_backend_path(cores=cores),
            connections=interface_class(cores=cores, **kwargs),
        ),
        future_queue=future_queue,
    )


def execute_parallel_tasks_loop(interface, future_queue):
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            interface.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
            future_queue.join()
            break
        elif "fn" in task_dict.keys() and "future" in task_dict.keys():
            f = task_dict.pop("future")
            if f.set_running_or_notify_cancel():
                try:
                    f.set_result(interface.send_and_receive_dict(input_dict=task_dict))
                except Exception as thread_exception:
                    interface.shutdown(wait=True)
                    future_queue.task_done()
                    f.set_exception(exception=thread_exception)
                    raise thread_exception
                else:
                    future_queue.task_done()
        elif "fn" in task_dict.keys() and "init" in task_dict.keys():
            interface.send_dict(input_dict=task_dict)
            future_queue.task_done()


def executor_broker(
    future_queue,
    max_workers,
    executor_class,
    sleep_interval=0.1,
    **kwargs,
):
    meta_future_lst = _get_executor_dict(
        max_workers=max_workers,
        executor_class=executor_class,
        **kwargs,
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
                future_queue.join()
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


def _get_backend_path(cores):
    command_lst = [sys.executable]
    if cores > 1:
        command_lst += [_get_command_path(executable="mpiexec.py")]
    else:
        command_lst += [_get_command_path(executable="serial.py")]
    return command_lst


def _get_command_path(executable):
    return os.path.abspath(os.path.join(__file__, "..", "..", "backend", executable))


def _get_executor_dict(max_workers, executor_class, **kwargs):
    return {_get_future_done(): executor_class(**kwargs) for _ in range(max_workers)}


def _get_future_done():
    f = Future()
    f.set_result(True)
    return f
