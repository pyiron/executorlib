import inspect
import os
import socket
import time
import queue

import cloudpickle

from pympipool.external_interfaces.communication import SocketInterface


def cancel_items_in_queue(que):
    while True:
        try:
            item = que.get_nowait()
            if isinstance(item, dict) and "future" in item.keys():
                item["future"].cancel()
        except queue.Empty:
            break


def cloudpickle_register(ind=2):
    # Cloudpickle can either pickle by value or pickle by reference. The functions which are communicated have to
    # be pickled by value rather than by reference, so the module which calls the map function is pickled by value.
    # https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
    # inspect can help to find the module which is calling pympipool
    # https://docs.python.org/3/library/inspect.html
    # to learn more about inspect another good read is:
    # http://pymotw.com/2/inspect/index.html#module-inspect
    # 1 refers to 1 level higher than the map function
    try:  # When executed in a jupyter notebook this can cause a ValueError - in this case we just ignore it.
        cloudpickle.register_pickle_by_value(inspect.getmodule(inspect.stack()[ind][0]))
    except ValueError:
        pass


def command_line_options(
    hostname,
    port_selected,
    path,
    cores,
    cores_per_task=1,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    enable_mpi4py_backend=True,
    enable_multi_host=False,
):
    """
    Translate the individual parameters to command line options.

    Args:
        hostname (str): name of the host where the SocketInterface instance runs the client process should conenct to.
        port_selected (int): port the SocketInterface instance runs on.
        path (str): path to the python script which should be executed as client process.
        cores (int): defines the total number of MPI ranks to use
        cores_per_task (int): number of MPI ranks per task - defaults to 1
        oversubscribe (bool): enable of disable the oversubscribe feature of OpenMPI - defaults to False
        enable_flux_backend (bool): enable the flux-framework as backend - defaults to False
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        enable_mpi4py_backend (bool): enable the mpi4py.futures module - defaults to True
        enable_multi_host (bool): communicate the host to connect to - defaults to False

    Returns:
        list: list of strings to be executed on the command line
    """
    if enable_flux_backend:
        command_lst = ["flux", "run"]
    elif enable_slurm_backend:
        command_lst = ["srun"]
    else:
        command_lst = ["mpiexec"]
    if oversubscribe:
        command_lst += ["--oversubscribe"]
    if cores_per_task == 1 and enable_mpi4py_backend:
        command_lst += ["-n", str(cores), "python", "-m", "mpi4py.futures"]
    elif cores_per_task > 1 and enable_mpi4py_backend:
        # Running MPI parallel tasks within the map() requires mpi4py to use mpi spawn:
        # https://github.com/mpi4py/mpi4py/issues/324
        command_lst += ["-n", "1", "python"]
    else:
        command_lst += ["-n", str(cores), "python"]
    command_lst += [path]
    if enable_flux_backend or enable_slurm_backend or enable_multi_host:
        command_lst += [
            "--host",
            hostname,
        ]
    command_lst += [
        "--zmqport",
        str(port_selected),
    ]
    if enable_mpi4py_backend:
        command_lst += [
            "--cores-per-task",
            str(cores_per_task),
            "--cores-total",
            str(cores),
        ]
    return command_lst


def execute_parallel_tasks(
    future_queue,
    cores,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    cwd=None,
    queue_adapter=None,
    queue_adapter_kwargs=None,
):
    """
    Execute a single tasks in parallel using the message passing interface (MPI).

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       oversubscribe (bool): enable of disable the oversubscribe feature of OpenMPI - defaults to False
       enable_flux_backend (bool): enable the flux-framework as backend - defaults to False
       enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
       cwd (str/None): current working directory where the parallel python task is executed
       queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
       queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter
    """
    interface = SocketInterface(
        queue_adapter=queue_adapter, queue_adapter_kwargs=queue_adapter_kwargs
    )
    interface.bootup(
        command_lst=get_parallel_subprocess_command(
            port_selected=interface.bind_to_random_port(),
            cores=cores,
            cores_per_task=1,
            oversubscribe=oversubscribe,
            enable_flux_backend=enable_flux_backend,
            enable_slurm_backend=enable_slurm_backend,
            enable_mpi4py_backend=False,
            enable_multi_host=queue_adapter is not None,
        ),
        cwd=cwd,
        cores=cores,
    )
    _execute_parallel_tasks_loop(interface=interface, future_queue=future_queue)


def execute_serial_tasks(
    future_queue,
    cores,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    cwd=None,
    sleep_interval=0.1,
    queue_adapter=None,
    queue_adapter_kwargs=None,
):
    """
    Execute a single tasks in serial.

    Args:
       future_queue (queue.Queue): task queue of dictionary objects which are submitted to the parallel process
       cores (int): defines the total number of MPI ranks to use
       oversubscribe (bool): enable of disable the oversubscribe feature of OpenMPI - defaults to False
       enable_flux_backend (bool): enable the flux-framework as backend - defaults to False
       enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
       cwd (str/None): current working directory where the parallel python task is executed
       sleep_interval (float):
       queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
       queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter
    """
    future_dict = {}
    interface = SocketInterface(
        queue_adapter=queue_adapter, queue_adapter_kwargs=queue_adapter_kwargs
    )
    interface.bootup(
        command_lst=get_parallel_subprocess_command(
            port_selected=interface.bind_to_random_port(),
            cores=cores,
            cores_per_task=1,
            oversubscribe=oversubscribe,
            enable_flux_backend=enable_flux_backend,
            enable_slurm_backend=enable_slurm_backend,
            enable_mpi4py_backend=True,
            enable_multi_host=queue_adapter is not None,
        ),
        cwd=cwd,
        cores=cores,
    )
    _execute_serial_tasks_loop(
        interface=interface,
        future_queue=future_queue,
        future_dict=future_dict,
        sleep_interval=sleep_interval,
    )


def get_parallel_subprocess_command(
    port_selected,
    cores,
    cores_per_task=1,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    enable_mpi4py_backend=True,
    enable_multi_host=False,
):
    """
    Translate the individual parameters to command line options.

    Args:
        port_selected (int): port the SocketInterface instance runs on.
        cores (int): defines the total number of MPI ranks to use
        cores_per_task (int): number of MPI ranks per task - defaults to 1
        oversubscribe (bool): enable of disable the oversubscribe feature of OpenMPI - defaults to False
        enable_flux_backend (bool): enable the flux-framework as backend - defaults to False
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        enable_mpi4py_backend (bool): enable the mpi4py.futures module - defaults to True
        enable_multi_host (bool): communicate the host to connect to - defaults to False

    Returns:
        list: list of strings to be executed on the command line
    """
    if enable_mpi4py_backend:
        executable = "mpipool.py"
    else:
        executable = "mpiexec.py"
    return command_line_options(
        hostname=socket.gethostname(),
        port_selected=port_selected,
        path=os.path.abspath(
            os.path.join(__file__, "../../parallel_executors", executable)
        ),
        cores=cores,
        cores_per_task=cores_per_task,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_slurm_backend=enable_slurm_backend,
        enable_mpi4py_backend=enable_mpi4py_backend,
        enable_multi_host=enable_multi_host,
    )


def _execute_parallel_tasks_loop(interface, future_queue):
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            interface.shutdown(wait=task_dict["wait"])
            break
        elif "fn" in task_dict.keys() and "future" in task_dict.keys():
            f = task_dict.pop("future")
            if f.set_running_or_notify_cancel():
                try:
                    f.set_result(interface.send_and_receive_dict(input_dict=task_dict))
                except Exception as thread_exeception:
                    f.set_exception(exception=thread_exeception)
                    raise thread_exeception
        elif "fn" in task_dict.keys() and "init" in task_dict.keys():
            interface.send_dict(input_dict=task_dict)


def _execute_serial_tasks_loop(
    interface, future_queue, future_dict, sleep_interval=0.1
):
    while True:
        try:
            task_dict = future_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
                done_dict = interface.shutdown(wait=task_dict["wait"])
                if isinstance(done_dict, dict):
                    for k, v in done_dict.items():
                        if k in future_dict.keys() and not future_dict[k].cancelled():
                            future_dict.pop(k).set_result(v)
                break
            elif "fn" in task_dict.keys() and "future" in task_dict.keys():
                f = task_dict.pop("future")
                future_hash = interface.send_and_receive_dict(input_dict=task_dict)
                future_dict[future_hash] = f
        _update_future_dict(
            interface=interface, future_dict=future_dict, sleep_interval=sleep_interval
        )


def _update_future_dict(interface, future_dict, sleep_interval=0.1):
    time.sleep(sleep_interval)
    hash_to_update = [h for h, f in future_dict.items() if not f.done()]
    hash_to_cancel = [h for h, f in future_dict.items() if f.cancelled()]
    if len(hash_to_update) > 0:
        for k, v in interface.send_and_receive_dict(
            input_dict={"update": hash_to_update}
        ).items():
            future_dict.pop(k).set_result(v)
    if len(hash_to_cancel) > 0 and interface.send_and_receive_dict(
        input_dict={"cancel": hash_to_cancel}
    ):
        for h in hash_to_cancel:
            del future_dict[h]
