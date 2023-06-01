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


def map_funct(executor, funct, lst, cores_per_task):
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


def call_funct(input_dict, funct=None):
    if funct is None:

        def funct(*args, **kwargs):
            return args[0].__call__(*args[1:], **kwargs)

    if "a" in input_dict.keys() and "k" in input_dict.keys():
        return funct(input_dict["f"], *input_dict["a"], **input_dict["k"])
    elif "a" in input_dict.keys():
        return funct(input_dict["f"], *input_dict["a"])
    elif "k" in input_dict.keys():
        return funct(input_dict["f"], **input_dict["k"])
    else:
        raise ValueError("Neither *args nor *kwargs are defined.")


def parse_socket_communication(executor, input_dict, future_dict, cores_per_task):
    if "c" in input_dict.keys() and input_dict["c"] == "close":
        # If close "c" is communicated the process is shutdown.
        return "exit"
    elif "f" in input_dict.keys() and "l" in input_dict.keys():
        # If a function "f" and a list or arguments "l" are communicated,
        # pympipool uses the map() function to apply the function on the list.
        try:
            output = map_funct(
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
        future = call_funct(input_dict=input_dict, funct=executor.submit)
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
