import inspect
import os
import socket
import queue

import cloudpickle

from pympipool.shared.communication import SocketInterface


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
    except ValueError:
        pass


def command_line_options(
    hostname,
    port_selected,
    path,
    cores,
    gpus_per_task=0,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
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
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
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
    if gpus_per_task > 0 and (enable_flux_backend or enable_slurm_backend):
        command_lst += ["--gpus-per-task=" + str(gpus_per_task)]
    elif gpus_per_task > 0:
        raise ValueError("GPU binding is only supported for flux and SLURM backend.")
    if oversubscribe:
        command_lst += ["--oversubscribe"]
    command_lst += ["-n", str(cores), "python", path]
    if enable_flux_backend or enable_slurm_backend or enable_multi_host:
        command_lst += [
            "--host",
            hostname,
        ]
    command_lst += [
        "--zmqport",
        str(port_selected),
    ]
    return command_lst


def execute_parallel_tasks(
    future_queue,
    cores,
    gpus_per_task=0,
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
       gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
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
        command_lst=command_line_options(
            hostname=socket.gethostname(),
            port_selected=interface.bind_to_random_port(),
            path=os.path.abspath(
                os.path.join(__file__, "..", "..", "backend", "mpiexec.py")
            ),
            cores=cores,
            gpus_per_task=gpus_per_task,
            oversubscribe=oversubscribe,
            enable_flux_backend=enable_flux_backend,
            enable_slurm_backend=enable_slurm_backend,
            enable_multi_host=queue_adapter is not None,
        ),
        cwd=cwd,
        cores=cores,
    )
    _execute_parallel_tasks_loop(interface=interface, future_queue=future_queue)


def _execute_parallel_tasks_loop(interface, future_queue):
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            interface.shutdown(wait=task_dict["wait"])
            future_queue.task_done()
            break
        elif "fn" in task_dict.keys() and "future" in task_dict.keys():
            f = task_dict.pop("future")
            if f.set_running_or_notify_cancel():
                try:
                    f.set_result(interface.send_and_receive_dict(input_dict=task_dict))
                except Exception as thread_exeception:
                    interface.shutdown(wait=True)
                    future_queue.task_done()
                    f.set_exception(exception=thread_exeception)
                    raise thread_exeception
                else:
                    future_queue.task_done()
        elif "fn" in task_dict.keys() and "init" in task_dict.keys():
            interface.send_dict(input_dict=task_dict)
            future_queue.task_done()
