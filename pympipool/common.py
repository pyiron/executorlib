import os
import socket
import subprocess
import zmq
from tqdm import tqdm


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


def initialize_zmq(host, port):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://" + host + ":" + port)
    return context, socket


def wrap(funct, number_of_cores_per_communicator):
    def functwrapped(*args, **kwargs):
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
        rank = MPI.COMM_WORLD.Get_rank()
        comm_new = MPI.COMM_WORLD.Split(
            rank // number_of_cores_per_communicator,
            rank % number_of_cores_per_communicator,
        )
        comm_new.Barrier()
        return funct(*args, comm=comm_new, **kwargs)

    return functwrapped


def exec_funct(executor, funct, lst, cores_per_task):
    if cores_per_task == 1:
        results = executor.map(funct, lst)
        return list(tqdm(results, desc="Tasks", total=len(lst)))
    else:
        lst_parallel = []
        for input_parameter in lst:
            for _ in range(cores_per_task):
                lst_parallel.append(input_parameter)
        results = executor.map(
            wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
            lst_parallel,
        )
        return list(tqdm(results, desc="Tasks", total=len(lst_parallel)))[
            ::cores_per_task
        ]


def parse_socket_communication(executor, input_dict, future_dict, cores_per_task):
    if "c" in input_dict.keys() and input_dict["c"] == "close":
        # If close "c" is communicated the process is shutdown.
        return "exit"
    elif "f" in input_dict.keys() and "l" in input_dict.keys():
        # If a function "f" and a list or arguments "l" are communicated,
        # pympipool uses the map() function to apply the function on the list.
        try:
            output = exec_funct(
                executor=executor,
                funct=input_dict["f"],
                lst=input_dict["l"],
                cores_per_task=cores_per_task,
            )
        except Exception as error:
            return {"e": error, "et": str(type(error))}
        else:
            return {"r": output}
    elif "f" in input_dict.keys() and (
        "a" in input_dict.keys() or "k" in input_dict.keys()
    ):
        # If a function "f" and either arguments "a" or keyword arguments "k" are
        # communicated pympipool uses submit() to asynchronously apply the function
        # on the arguments and or keyword arguments.
        if "a" in input_dict.keys() and "k" in input_dict.keys():
            future = executor.submit(
                input_dict["f"], *input_dict["a"], **input_dict["k"]
            )
        elif "a" in input_dict.keys():
            future = executor.submit(input_dict["f"], *input_dict["a"])
        elif "k" in input_dict.keys():
            future = executor.submit(input_dict["f"], **input_dict["k"])
        else:
            raise ValueError("Neither *args nor *kwargs are defined.")
        future_hash = hash(future)
        future_dict[future_hash] = future
        return {"r": future_hash}
    elif "u" in input_dict.keys():
        # If update "u" is communicated pympipool checks for asynchronously submitted
        # functions which have completed in the meantime and communicates their results.
        done_dict = {
            k: f.result()
            for k, f in {k: future_dict[k] for k in input_dict["u"]}.items()
            if f.done()
        }
        for k in done_dict.keys():
            del future_dict[k]
        return {"r": done_dict}


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


def start_parallel_subprocess(
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
        path=os.path.abspath(os.path.join(__file__, "..", executable)),
        cores=cores,
        cores_per_task=cores_per_task,
        oversubscribe=oversubscribe,
        enable_flux_backend=enable_flux_backend,
        enable_mpi4py_backend=enable_mpi4py_backend,
    )
    process = subprocess.Popen(
        args=command_lst,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )
    return process
