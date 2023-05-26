import os
import socket
import subprocess


def command_line_options(
    hostname,
    port_selected,
    path,
    cores,
    cores_per_task=1,
    oversubscribe=False,
    enable_flux_backend=False,
):
    if enable_flux_backend:
        command_lst = ["flux", "run"]
    else:
        command_lst = ["mpiexec"]
    if oversubscribe:
        command_lst += ["--oversubscribe"]
    if cores_per_task == 1:
        command_lst += ["-n", str(cores), "python", "-m", "mpi4py.futures"]
    else:
        command_lst += [
            "-n",
            "1",
            "python",
        ]
    command_lst += [path]
    if enable_flux_backend:
        command_lst += [
            "--host",
            hostname,
        ]
    command_lst += [
        "--zmqport",
        str(port_selected),
        "--cores-per-task",
        str(cores_per_task),
        "--cores-total",
        str(cores),
    ]
    return command_lst


def start_parallel_subprocess(
    port_selected, cores, cores_per_task, oversubscribe, enable_flux_backend
):
    command_lst = command_line_options(
        hostname=socket.gethostname(),
        port_selected=port_selected,
        path=os.path.abspath(os.path.join(__file__, "..", "__main__.py")),
        cores=cores,
        cores_per_task=cores_per_task,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
    )
    process = subprocess.Popen(
        args=command_lst,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )
    return process


def parse_arguments(argument_lst):
    argument_dict = {
        "total_cores": "--cores-total",
        "zmqport": "--zmqport",
        "cores_per_task": "--cores-per-task",
        "host": "--host",
    }
    parse_dict = {"host": "localhost"}
    parse_dict.update(
        {
            k: argument_lst[argument_lst.index(v) + 1]
            for k, v in argument_dict.items()
            if v in argument_lst
        }
    )
    return parse_dict
