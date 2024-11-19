import importlib.util
import os
import queue
import sys
from concurrent.futures import Future
from time import sleep
from typing import Callable, List, Optional

from executorlib.base.executor import ExecutorBase, cancel_items_in_queue
from executorlib.standalone.command import get_command_path
from executorlib.standalone.inputcheck import (
    check_resource_dict,
    check_resource_dict_is_empty,
)
from executorlib.standalone.interactive.communication import (
    SocketInterface,
    interface_bootup,
)
from executorlib.standalone.interactive.spawner import BaseSpawner, MpiExecSpawner
from executorlib.standalone.serialize import serialize_funct_h5
from executorlib.standalone.thread import RaisingThread


class ExecutorBroker(ExecutorBase):
    def submit(self, fn: callable, *args, resource_dict: dict = {}, **kwargs) -> Future:
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
            Future: A Future representing the given call.
        """
        check_resource_dict_is_empty(resource_dict=resource_dict)
        check_resource_dict(function=fn)
        f = Future()
        self._future_queue.put({"fn": fn, "args": args, "kwargs": kwargs, "future": f})
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
            for _ in range(len(self._process)):
                self._future_queue.put({"shutdown": True, "wait": wait})
            if wait:
                for process in self._process:
                    process.join()
                self._future_queue.join()
        self._process = None
        self._future_queue = None

    def _set_process(self, process: List[RaisingThread]):
        """
        Set the process for the executor.

        Args:
            process (List[RaisingThread]): The process for the executor.
        """
        self._process = process
        for process in self._process:
            process.start()


class InteractiveExecutor(ExecutorBroker):
    """
    The executorlib.interactive.executor.InteractiveExecutor leverages the exeutorlib interfaces to distribute python
    tasks on a workstation or inside a queuing system allocation. In contrast to the mpi4py.futures.MPIPoolExecutor the
    executorlib.interactive.executor.InteractiveExecutor can be executed in a serial python process and does not require
    the python script to be executed with MPI. Consequently, it is primarily an abstraction of its functionality to
    improves the usability in particular when used in combination with Jupyter notebooks.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel
        executor_kwargs (dict): keyword arguments for the executor
        spawner (BaseSpawner): interface class to initiate python processes

    Examples:

        >>> import numpy as np
        >>> from executorlib.interactive.executor import InteractiveExecutor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> def init_k():
        >>>     return {"k": 3}
        >>>
        >>> with InteractiveExecutor(max_workers=2, executor_kwargs={"init_function": init_k}) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]

    """

    def __init__(
        self,
        max_workers: int = 1,
        executor_kwargs: dict = {},
        spawner: BaseSpawner = MpiExecSpawner,
    ):
        super().__init__(max_cores=executor_kwargs.get("max_cores", None))
        executor_kwargs["future_queue"] = self._future_queue
        executor_kwargs["spawner"] = spawner
        self._set_process(
            process=[
                RaisingThread(
                    target=execute_parallel_tasks,
                    kwargs=executor_kwargs,
                )
                for _ in range(max_workers)
            ],
        )


class InteractiveStepExecutor(ExecutorBase):
    """
    The executorlib.interactive.executor.InteractiveStepExecutor leverages the executorlib interfaces to distribute python
    tasks. In contrast to the mpi4py.futures.MPIPoolExecutor the executorlib.interactive.executor.InteractiveStepExecutor
    can be executed in a serial python process and does not require the python script to be executed with MPI.
    Consequently, it is primarily an abstraction of its functionality to improve the usability in particular when used
    in combination with Jupyter notebooks.

    Args:
        max_cores (int): defines the number workers which can execute functions in parallel
        executor_kwargs (dict): keyword arguments for the executor
        spawner (BaseSpawner): interface class to initiate python processes

    Examples:

        >>> import numpy as np
        >>> from executorlib.interactive.executor import InteractiveStepExecutor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> with PyFluxStepExecutor(max_cores=2) as p:
        >>>     fs = p.submit(calc, 2, j=4, k=3, resource_dict={"cores": 2})
        >>>     print(fs.result())

        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]

    """

    def __init__(
        self,
        max_cores: Optional[int] = None,
        max_workers: Optional[int] = None,
        executor_kwargs: dict = {},
        spawner: BaseSpawner = MpiExecSpawner,
    ):
        super().__init__(max_cores=executor_kwargs.get("max_cores", None))
        executor_kwargs["future_queue"] = self._future_queue
        executor_kwargs["spawner"] = spawner
        executor_kwargs["max_cores"] = max_cores
        executor_kwargs["max_workers"] = max_workers
        self._set_process(
            RaisingThread(
                target=execute_separate_tasks,
                kwargs=executor_kwargs,
            )
        )


def execute_parallel_tasks(
    future_queue: queue.Queue,
    cores: int = 1,
    spawner: BaseSpawner = MpiExecSpawner,
    hostname_localhost: Optional[bool] = None,
    init_function: Optional[Callable] = None,
    cache_directory: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       spawner (BaseSpawner): Spawner to start process on selected compute resources
       hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
       init_function (callable): optional function to preset arguments for functions which are submitted later
       cache_directory (str, optional): The directory to store cache files. Defaults to "cache".
    """
    interface = interface_bootup(
        command_lst=_get_backend_path(
            cores=cores,
        ),
        connections=spawner(cores=cores, **kwargs),
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
            if cache_directory is None:
                _execute_task(
                    interface=interface, task_dict=task_dict, future_queue=future_queue
                )
            else:
                _execute_task_with_cache(
                    interface=interface,
                    task_dict=task_dict,
                    future_queue=future_queue,
                    cache_directory=cache_directory,
                )


def execute_separate_tasks(
    future_queue: queue.Queue,
    spawner: BaseSpawner = MpiExecSpawner,
    max_cores: Optional[int] = None,
    max_workers: Optional[int] = None,
    hostname_localhost: Optional[bool] = None,
    **kwargs,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       spawner (BaseSpawner): Interface to start process on selected compute resources
       max_cores (int): defines the number cores which can be used in parallel
       max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                          cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                          recommended, as computers have a limited number of compute cores.
       hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
    """
    active_task_dict = {}
    process_lst, qtask_lst = [], []
    if "cores" not in kwargs.keys():
        kwargs["cores"] = 1
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            if task_dict["wait"]:
                _ = [process.join() for process in process_lst]
            future_queue.task_done()
            future_queue.join()
            break
        elif "fn" in task_dict.keys() and "future" in task_dict.keys():
            qtask = queue.Queue()
            process, active_task_dict = _submit_function_to_separate_process(
                task_dict=task_dict,
                qtask=qtask,
                active_task_dict=active_task_dict,
                spawner=spawner,
                executor_kwargs=kwargs,
                max_cores=max_cores,
                max_workers=max_workers,
                hostname_localhost=hostname_localhost,
            )
            qtask_lst.append(qtask)
            process_lst.append(process)
            future_queue.task_done()


def execute_tasks_with_dependencies(
    future_queue: queue.Queue,
    executor_queue: queue.Queue,
    executor: ExecutorBase,
    refresh_rate: float = 0.01,
):
    """
    Resolve the dependencies of multiple tasks, by analysing which task requires concurrent.future.Futures objects from
    other tasks.

    Args:
        future_queue (Queue): Queue for receiving new tasks.
        executor_queue (Queue): Queue for the internal executor.
        executor (ExecutorBase): Executor to execute the tasks with after the dependencies are resolved.
        refresh_rate (float): Set the refresh rate in seconds, how frequently the input queue is checked.
    """
    wait_lst = []
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            task_dict = None
        if (  # shutdown the executor
            task_dict is not None
            and "shutdown" in task_dict.keys()
            and task_dict["shutdown"]
        ):
            executor.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
            future_queue.join()
            break
        elif (  # handle function submitted to the executor
            task_dict is not None
            and "fn" in task_dict.keys()
            and "future" in task_dict.keys()
        ):
            future_lst, ready_flag = _get_future_objects_from_input(task_dict=task_dict)
            if len(future_lst) == 0 or ready_flag:
                # No future objects are used in the input or all future objects are already done
                task_dict["args"], task_dict["kwargs"] = _update_futures_in_input(
                    args=task_dict["args"], kwargs=task_dict["kwargs"]
                )
                executor_queue.put(task_dict)
            else:  # Otherwise add the function to the wait list
                task_dict["future_lst"] = future_lst
                wait_lst.append(task_dict)
            future_queue.task_done()
        elif len(wait_lst) > 0:
            number_waiting = len(wait_lst)
            # Check functions in the wait list and execute them if all future objects are now ready
            wait_lst = _submit_waiting_task(
                wait_lst=wait_lst, executor_queue=executor_queue
            )
            # if no job is ready, sleep for a moment
            if len(wait_lst) == number_waiting:
                sleep(refresh_rate)
        else:
            # If there is nothing else to do, sleep for a moment
            sleep(refresh_rate)


def _get_backend_path(
    cores: int,
) -> list:
    """
    Get command to call backend as a list of two strings

    Args:
        cores (int): Number of cores used to execute the task, if it is greater than one use interactive_parallel.py else interactive_serial.py

    Returns:
        list[str]: List of strings containing the python executable path and the backend script to execute
    """
    command_lst = [sys.executable]
    if cores > 1 and importlib.util.find_spec("mpi4py") is not None:
        command_lst += [get_command_path(executable="interactive_parallel.py")]
    elif cores > 1:
        raise ImportError(
            "mpi4py is required for parallel calculations. Please install mpi4py."
        )
    else:
        command_lst += [get_command_path(executable="interactive_serial.py")]
    return command_lst


def _wait_for_free_slots(
    active_task_dict: dict,
    cores_requested: int,
    max_cores: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> dict:
    """
    Wait for available computing resources to become available.

    Args:
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        cores_requested (int): Number of cores required for executing the next task
        max_cores (int): Maximum number cores which can be used
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.

    Returns:
        dict: Dictionary containing the future objects and the number of cores they require
    """
    if max_cores is not None:
        while sum(active_task_dict.values()) + cores_requested > max_cores:
            active_task_dict = {
                k: v for k, v in active_task_dict.items() if not k.done()
            }
    elif max_workers is not None and max_cores is None:
        while len(active_task_dict.values()) + 1 > max_workers:
            active_task_dict = {
                k: v for k, v in active_task_dict.items() if not k.done()
            }
    return active_task_dict


def _submit_waiting_task(wait_lst: List[dict], executor_queue: queue.Queue) -> list:
    """
    Submit the waiting tasks, which future inputs have been completed, to the executor

    Args:
        wait_lst (list): List of waiting tasks
        executor_queue (Queue): Queue of the internal executor

    Returns:
        list: list tasks which future inputs have not been completed
    """
    wait_tmp_lst = []
    for task_wait_dict in wait_lst:
        if all([future.done() for future in task_wait_dict["future_lst"]]):
            del task_wait_dict["future_lst"]
            task_wait_dict["args"], task_wait_dict["kwargs"] = _update_futures_in_input(
                args=task_wait_dict["args"], kwargs=task_wait_dict["kwargs"]
            )
            executor_queue.put(task_wait_dict)
        else:
            wait_tmp_lst.append(task_wait_dict)
    return wait_tmp_lst


def _update_futures_in_input(args: tuple, kwargs: dict):
    """
    Evaluate future objects in the arguments and keyword arguments by calling future.result()

    Args:
        args (tuple): function arguments
        kwargs (dict): function keyword arguments

    Returns:
        tuple, dict: arguments and keyword arguments with each future object in them being evaluated
    """

    def get_result(arg):
        if isinstance(arg, Future):
            return arg.result()
        elif isinstance(arg, list):
            return [get_result(arg=el) for el in arg]
        else:
            return arg

    args = [get_result(arg=arg) for arg in args]
    kwargs = {key: get_result(arg=value) for key, value in kwargs.items()}
    return args, kwargs


def _get_future_objects_from_input(task_dict: dict):
    """
    Check the input parameters if they contain future objects and which of these future objects are executed

    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": callable, "args": (), "kwargs": {}, "resource_dict": {}}

    Returns:
        list, boolean: list of future objects and boolean flag if all future objects are already done
    """
    future_lst = []

    def find_future_in_list(lst):
        for el in lst:
            if isinstance(el, Future):
                future_lst.append(el)
            elif isinstance(el, list):
                find_future_in_list(lst=el)

    find_future_in_list(lst=task_dict["args"])
    find_future_in_list(lst=task_dict["kwargs"].values())
    boolean_flag = len([future for future in future_lst if future.done()]) == len(
        future_lst
    )
    return future_lst, boolean_flag


def _submit_function_to_separate_process(
    task_dict: dict,
    active_task_dict: dict,
    qtask: queue.Queue,
    spawner: BaseSpawner,
    executor_kwargs: dict,
    max_cores: Optional[int] = None,
    max_workers: Optional[int] = None,
    hostname_localhost: Optional[bool] = None,
):
    """
    Submit function to be executed in separate Python process
    Args:
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": callable, "args": (), "kwargs": {}, "resource_dict": {}}
        active_task_dict (dict): Dictionary containing the future objects and the number of cores they require
        qtask (queue.Queue): Queue to communicate with the thread linked to the process executing the python function
        spawner (BaseSpawner): Interface to start process on selected compute resources
        executor_kwargs (dict): keyword parameters used to initialize the Executor
        max_cores (int): defines the number cores which can be used in parallel
        max_workers (int): for backwards compatibility with the standard library, max_workers also defines the number of
                           cores which can be used in parallel - just like the max_cores parameter. Using max_cores is
                           recommended, as computers have a limited number of compute cores.
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                     context of an HPC cluster this essential to be able to communicate to an
                                     Executor running on a different compute node within the same allocation. And
                                     in principle any computer should be able to resolve that their own hostname
                                     points to the same address as localhost. Still MacOS >= 12 seems to disable
                                     this look up for security reasons. So on MacOS it is required to set this
                                     option to true
    Returns:
        RaisingThread, dict: thread for communicating with the python process which is executing the function and
                             dictionary containing the future objects and the number of cores they require
    """
    resource_dict = task_dict.pop("resource_dict").copy()
    qtask.put(task_dict)
    qtask.put({"shutdown": True, "wait": True})
    if "cores" not in resource_dict.keys() or (
        resource_dict["cores"] == 1 and executor_kwargs["cores"] >= 1
    ):
        resource_dict["cores"] = executor_kwargs["cores"]
    slots_required = resource_dict["cores"] * resource_dict.get("threads_per_core", 1)
    active_task_dict = _wait_for_free_slots(
        active_task_dict=active_task_dict,
        cores_requested=slots_required,
        max_cores=max_cores,
        max_workers=max_workers,
    )
    active_task_dict[task_dict["future"]] = slots_required
    task_kwargs = executor_kwargs.copy()
    task_kwargs.update(resource_dict)
    task_kwargs.update(
        {
            "future_queue": qtask,
            "spawner": spawner,
            "hostname_localhost": hostname_localhost,
            "init_function": None,
        }
    )
    process = RaisingThread(
        target=execute_parallel_tasks,
        kwargs=task_kwargs,
    )
    process.start()
    return process, active_task_dict


def _execute_task(
    interface: SocketInterface, task_dict: dict, future_queue: queue.Queue
):
    """
    Execute the task in the task_dict by communicating it via the interface.

    Args:
        interface (SocketInterface): socket interface for zmq communication
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_queue (Queue): Queue for receiving new tasks.
    """
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


def _execute_task_with_cache(
    interface: SocketInterface,
    task_dict: dict,
    future_queue: queue.Queue,
    cache_directory: str,
):
    """
    Execute the task in the task_dict by communicating it via the interface using the cache in the cache directory.

    Args:
        interface (SocketInterface): socket interface for zmq communication
        task_dict (dict): task submitted to the executor as dictionary. This dictionary has the following keys
                          {"fn": callable, "args": (), "kwargs": {}, "resource_dict": {}}
        future_queue (Queue): Queue for receiving new tasks.
        cache_directory (str): The directory to store cache files.
    """
    from executorlib.standalone.hdf import dump, get_output

    task_key, data_dict = serialize_funct_h5(
        fn=task_dict["fn"],
        fn_args=task_dict["args"],
        fn_kwargs=task_dict["kwargs"],
        resource_dict=task_dict.get("resource_dict", {}),
    )
    os.makedirs(cache_directory, exist_ok=True)
    file_name = os.path.join(cache_directory, task_key + ".h5out")
    if task_key + ".h5out" not in os.listdir(cache_directory):
        f = task_dict.pop("future")
        if f.set_running_or_notify_cancel():
            try:
                result = interface.send_and_receive_dict(input_dict=task_dict)
                data_dict["output"] = result
                dump(file_name=file_name, data_dict=data_dict)
                f.set_result(result)
            except Exception as thread_exception:
                interface.shutdown(wait=True)
                future_queue.task_done()
                f.set_exception(exception=thread_exception)
                raise thread_exception
            else:
                future_queue.task_done()
    else:
        _, result = get_output(file_name=file_name)
        future = task_dict["future"]
        future.set_result(result)
        future_queue.task_done()
