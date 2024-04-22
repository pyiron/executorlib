from typing import Optional, List
from concurrent.futures import (
    Executor as FutureExecutor,
    Future,
)
import inspect
import os
import queue
import sys

import cloudpickle

from pympipool.shared.communication import interface_bootup
from pympipool.shared.thread import RaisingThread
from pympipool.shared.interface import BaseInterface


class ExecutorBase(FutureExecutor):
    def __init__(self):
        cloudpickle_register(ind=3)
        self._future_queue = queue.Queue()
        self._process = None

    @property
    def info(self):
        if self._process is not None and isinstance(self._process, list):
            meta_data_dict = self._process[0]._kwargs.copy()
            if "future_queue" in meta_data_dict.keys():
                del meta_data_dict["future_queue"]
            meta_data_dict["max_workers"] = len(self._process)
            return meta_data_dict
        elif self._process is not None:
            meta_data_dict = self._process._kwargs.copy()
            if "future_queue" in meta_data_dict.keys():
                del meta_data_dict["future_queue"]
            return meta_data_dict
        else:
            return None

    @property
    def future_queue(self):
        return self._future_queue

    def submit(self, fn: callable, *args, resource_dict: dict = {}, **kwargs):
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            fn (callable): function to submit for execution
            args: arguments for the submitted function
            kwargs: keyword arguments for the submitted function
            resource_dict (dict): resource dictionary, which defines the resources used for the execution of the
                                  function. Example resource dictionary: {
                                      cores: 1,
                                      threads_per_core: 1,
                                      gpus_per_worker: 0,
                                      oversubscribe: False,
                                      cwd: None,
                                      executor: None,
                                      hostname_localhost: False,
                                  }

        Returns:
            A Future representing the given call.
        """
        if len(resource_dict) > 0:
            raise ValueError(
                "When block_allocation is enabled, the resource requirements have to be defined on the executor level."
            )
        f = Future()
        self._future_queue.put({"fn": fn, "args": args, "kwargs": kwargs, "future": f})
        return f

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """
        Clean-up the resources associated with the Executor.

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
        if wait and self._process is not None:
            self._process.join()
            self._future_queue.join()
        self._process = None
        self._future_queue = None

    def _set_process(self, process: RaisingThread):
        self._process = process
        self._process.start()

    def __len__(self):
        return self._future_queue.qsize()

    def __del__(self):
        try:
            self.shutdown(wait=False)
        except (AttributeError, RuntimeError):
            pass


class ExecutorBroker(ExecutorBase):
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
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
        if self._process is not None:
            for _ in range(len(self._process)):
                self._future_queue.put({"shutdown": True, "wait": wait})
            if wait:
                for process in self._process:
                    process.join()
                self._future_queue.join()
        self._process = None
        self._future_queue = None

    def _set_process(self, process: List[RaisingThread]):
        self._process = process
        for process in self._process:
            process.start()


class ExecutorSteps(ExecutorBase):
    def submit(self, fn: callable, *args, resource_dict: dict = {}, **kwargs):
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            fn (callable): function to submit for execution
            args: arguments for the submitted function
            kwargs: keyword arguments for the submitted function
            resource_dict (dict): resource dictionary, which defines the resources used for the execution of the
                                  function. Example resource dictionary: {
                                      cores: 1,
                                      threads_per_core: 1,
                                      gpus_per_worker: 0,
                                      oversubscribe: False,
                                      cwd: None,
                                      executor: None,
                                      hostname_localhost: False,
                                  }

        Returns:
            A Future representing the given call.
        """
        f = Future()
        self._future_queue.put(
            {
                "fn": fn,
                "args": args,
                "kwargs": kwargs,
                "future": f,
                "resource_dict": resource_dict,
            }
        )
        return f

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
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
        if self._process is not None:
            self._future_queue.put({"shutdown": True, "wait": wait})
            if wait:
                self._process.join()
                self._future_queue.join()
        self._process = None
        self._future_queue = None


def cancel_items_in_queue(que: queue.Queue):
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


def cloudpickle_register(ind: int = 2):
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
    future_queue: queue.Queue,
    cores: int,
    interface_class: BaseInterface,
    hostname_localhost: bool = False,
    init_function: Optional[callable] = None,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       interface_class (BaseInterface): Interface to start process on selected compute resources
       hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
       init_function (callable): optional function to preset arguments for functions which are submitted later
    """
    interface = interface_bootup(
        command_lst=_get_backend_path(cores=cores),
        connections=interface_class(cores=cores, **kwargs),
        hostname_localhost=hostname_localhost,
    )
    if init_function is not None:
        interface.send_dict(
            input_dict={"init": True, "fn": init_function, "args": (), "kwargs": {}}
        )
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


def execute_separate_tasks(
    future_queue: queue.Queue,
    interface_class: BaseInterface,
    max_cores: int,
    hostname_localhost: bool = False,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       interface_class (BaseInterface): Interface to start process on selected compute resources
       max_cores (int): defines the number cores which can be used in parallel
       hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
    """
    active_task_dict = {}
    process_lst = []
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            if task_dict["wait"]:
                _ = [process.join() for process in process_lst]
            future_queue.task_done()
            future_queue.join()
            break
        elif "fn" in task_dict.keys() and "future" in task_dict.keys():
            process = _submit_function_to_separate_process(
                task_dict=task_dict,
                active_task_dict=active_task_dict,
                interface_class=interface_class,
                executor_kwargs=kwargs,
                max_cores=max_cores,
                hostname_localhost=hostname_localhost,
            )
            process_lst.append(process)
            future_queue.task_done()


def _get_backend_path(cores: int):
    """
    Get command to call backend as a list of two strings

    Args:
        cores (int): Number of cores used to execute the task, if it is greater than one use mpiexec.py else serial.py

    Returns:
        list[str]: List of strings containing the python executable path and the backend script to execute
    """
    command_lst = [sys.executable]
    if cores > 1:
        command_lst += [_get_command_path(executable="mpiexec.py")]
    else:
        command_lst += [_get_command_path(executable="serial.py")]
    return command_lst


def _get_command_path(executable: str):
    """
    Get path of the backend executable script

    Args:
        executable (str): Name of the backend executable script, either mpiexec.py or serial.py

    Returns:
        str: absolute path to the executable script
    """
    return os.path.abspath(os.path.join(__file__, "..", "..", "backend", executable))


def _wait_for_free_slots(active_task_dict: dict, cores_requested: int, max_cores: int):
    """
    Wait for available computing resources to become available.

    Args:
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        cores_requested (int): Number of cores required for executing the next task
        max_cores (int): defines the number cores which can be used in parallel

    Returns:
        dict: Dictionary containing the future objects and the number of cores they require
    """
    while sum(active_task_dict.values()) + cores_requested > max_cores:
        active_task_dict = {k: v for k, v in active_task_dict.items() if not k.done()}
    return active_task_dict


def _submit_function_to_separate_process(
    task_dict: dict,
    active_task_dict: dict,
    interface_class: BaseInterface,
    executor_kwargs: dict,
    max_cores: int,
    hostname_localhost: bool = False,
):
    """
    Submit function to be executed in separate Python process

    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": callable, "args": (), "kwargs": {}, "resource_dict": {}}
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        interface_class (BaseInterface): Interface to start process on selected compute resources
        executor_kwargs (dict): keyword parameters used to initialize the Executor
        max_cores (int): defines the number cores which can be used in parallel
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true

    Returns:
        RaisingThread: thread for communicating with the python process which is executing the function
    """
    resource_dict = task_dict.pop("resource_dict")
    qtask = queue.Queue()
    qtask.put(task_dict)
    qtask.put({"shutdown": True, "wait": True})
    if "cores" not in resource_dict.keys() or (
        resource_dict["cores"] == 1 and executor_kwargs["cores"] >= 1
    ):
        resource_dict["cores"] = executor_kwargs["cores"]
    active_task_dict = _wait_for_free_slots(
        active_task_dict=active_task_dict,
        cores_requested=resource_dict["cores"],
        max_cores=max_cores,
    )
    active_task_dict[task_dict["future"]] = resource_dict["cores"]
    task_kwargs = executor_kwargs.copy()
    task_kwargs.update(resource_dict)
    task_kwargs.update(
        {
            "future_queue": qtask,
            "interface_class": interface_class,
            "hostname_localhost": hostname_localhost,
            "init_function": None,
        }
    )
    process = RaisingThread(
        target=execute_parallel_tasks,
        kwargs=task_kwargs,
    )
    process.start()
    return process
