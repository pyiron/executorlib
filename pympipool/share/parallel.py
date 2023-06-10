import inspect
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


def wrap(funct, number_of_cores_per_communicator=1):
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


def map_funct(executor, funct, lst, chunksize=1, cores_per_task=1, map_flag=True):
    if cores_per_task == 1:
        if map_flag:
            results = executor.map(funct, lst, chunksize=chunksize)
        else:
            results = executor.starmap(funct, lst, chunksize=chunksize)
        return list(tqdm(results, desc="Tasks", total=len(lst)))
    else:
        lst_parallel = []
        for input_parameter in lst:
            for _ in range(cores_per_task):
                lst_parallel.append(input_parameter)
        if map_flag:
            results = executor.map(
                wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
                lst_parallel,
                chunksize=chunksize,
            )
        else:
            results = executor.starmap(
                wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
                lst_parallel,
                chunksize=chunksize,
            )
        return list(tqdm(results, desc="Tasks", total=len(lst_parallel)))[
            ::cores_per_task
        ]


def call_funct(input_dict, funct=None, memory=None):
    if funct is None:

        def funct(*args, **kwargs):
            return args[0].__call__(*args[1:], **kwargs)

    funct_args = inspect.getfullargspec(input_dict["fn"]).args
    if memory is not None:
        input_dict["kwargs"].update(
            update_dict_delta(
                dict_input=memory,
                dict_output=input_dict["kwargs"],
                keys_possible_lst=funct_args,
            )
        )
    return funct(input_dict["fn"], *input_dict["args"], **input_dict["kwargs"])


def parse_socket_communication(executor, input_dict, future_dict, cores_per_task=1):
    if "shutdown" in input_dict.keys() and input_dict["shutdown"]:
        # If close "c" is communicated the process is shutdown.
        return "exit"
    elif "fn" in input_dict.keys() and "iterable" in input_dict.keys():
        # If a function "f" and a list or arguments "l" are communicated,
        # pympipool uses the map() function to apply the function on the list.
        try:
            output = map_funct(
                executor=executor,
                funct=input_dict["fn"],
                lst=input_dict["iterable"],
                cores_per_task=cores_per_task,
                chunksize=input_dict["chunksize"],
                map_flag=input_dict["map"],
            )
        except Exception as error:
            return {"error": error, "error_type": str(type(error))}
        else:
            return {"result": output}
    elif (
        "fn" in input_dict.keys()
        and "args" in input_dict.keys()
        and "kwargs" in input_dict.keys()
    ):
        # If a function "f" and either arguments "a" or keyword arguments "k" are
        # communicated pympipool uses submit() to asynchronously apply the function
        # on the arguments and or keyword arguments.
        future = call_funct(input_dict=input_dict, funct=executor.submit)
        future_hash = hash(future)
        future_dict[future_hash] = future
        return {"result": future_hash}
    elif "update" in input_dict.keys():
        # If update "u" is communicated pympipool checks for asynchronously submitted
        # functions which have completed in the meantime and communicates their results.
        done_dict = {
            k: f.result()
            for k, f in {k: future_dict[k] for k in input_dict["update"]}.items()
            if f.done()
        }
        for k in done_dict.keys():
            del future_dict[k]
        return {"result": done_dict}
    elif "cancel" in input_dict.keys():
        for k in input_dict["cancel"]:
            future_dict[k].cancel()
        return {"result": True}


def update_dict_delta(dict_input, dict_output, keys_possible_lst):
    return {
        k: v
        for k, v in dict_input.items()
        if k in keys_possible_lst and k not in dict_output.keys()
    }
