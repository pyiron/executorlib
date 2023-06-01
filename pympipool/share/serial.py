import os
import socket

from pympipool.share.communication import SocketInterface


def command_line_options(
    hostname,
    port_selected,
    path,
    cores,
    cores_per_task=1,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_mpi4py_backend=True,
):
    if enable_flux_backend:
        command_lst = ["flux", "run"]
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
    if enable_flux_backend:
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
    cores_per_task,
    oversubscribe,
    enable_flux_backend,
    enable_mpi4py_backend,
):
    if enable_mpi4py_backend:
        executable = "mpipool.py"
    else:
        executable = "mpiexec.py"
    command_lst = command_line_options(
        hostname=socket.gethostname(),
        port_selected=port_selected,
        path=os.path.abspath(os.path.join(__file__, "../../executor", executable)),
        cores=cores,
        cores_per_task=cores_per_task,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_mpi4py_backend=enable_mpi4py_backend,
    )
    return command_lst


def execute_tasks(future_queue, cores, oversubscribe, enable_flux_backend):
    interface = SocketInterface()
    interface.bootup(
        command_lst=get_parallel_subprocess_command(
            port_selected=interface.bind_to_random_port(),
            cores=cores,
            cores_per_task=1,
            oversubscribe=oversubscribe,
            enable_flux_backend=enable_flux_backend,
            enable_mpi4py_backend=False,
        )
    )
    while True:
        task_dict = future_queue.get()
        if "c" in task_dict.keys() and task_dict["c"] == "close":
            interface.shutdown(wait=True)
            break
        elif "f" in task_dict.keys() and "l" in task_dict.keys():
            f = task_dict.pop("l")
            if f.set_running_or_notify_cancel():
                f.set_result(interface.send_and_receive_dict(input_dict=task_dict))
