import os
import queue
import socket
import time


from pympipool.shared.communication import SocketInterface


def command_line_options(
    hostname,
    port_selected,
    path,
    cores,
    cores_per_task=1,
    gpus_per_task=0,
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


def get_parallel_subprocess_command(
    port_selected,
    cores,
    cores_per_task=1,
    gpus_per_task=0,
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
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): enable of disable the oversubscribe feature of OpenMPI - defaults to False
        enable_flux_backend (bool): enable the flux-framework as backend - defaults to False
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        enable_mpi4py_backend (bool): enable the mpi4py.futures module - defaults to True
        enable_multi_host (bool): communicate the host to connect to - defaults to False

    Returns:
        list: list of strings to be executed on the command line
    """
    if enable_mpi4py_backend:
        executable = os.path.abspath(
            os.path.join(__file__, "..", "..", "backend", "mpipool.py")
        )
    else:
        executable = os.path.abspath(
            os.path.join(__file__, "..", "..", "..", "backend", "mpiexec.py")
        )
    return command_line_options(
        hostname=socket.gethostname(),
        port_selected=port_selected,
        path=executable,
        cores=cores,
        cores_per_task=cores_per_task,
        gpus_per_task=gpus_per_task,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_slurm_backend=enable_slurm_backend,
        enable_mpi4py_backend=enable_mpi4py_backend,
        enable_multi_host=enable_multi_host,
    )


def execute_serial_tasks(
    future_queue,
    cores,
    gpus_per_task=0,
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
       gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
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
            gpus_per_task=gpus_per_task,
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
                future_queue.task_done()
                break
            elif "fn" in task_dict.keys() and "future" in task_dict.keys():
                f = task_dict.pop("future")
                future_hash = interface.send_and_receive_dict(input_dict=task_dict)
                future_dict[future_hash] = f
                future_queue.task_done()
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
