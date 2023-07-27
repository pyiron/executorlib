from tqdm import tqdm

from pympipool.shared.backend import call_funct, _update_default_dict_from_arguments


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
                _wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
                lst_parallel,
                chunksize=chunksize,
            )
        else:
            results = executor.starmap(
                _wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
                lst_parallel,
                chunksize=chunksize,
            )
        return list(tqdm(results, desc="Tasks", total=len(lst_parallel)))[
            ::cores_per_task
        ]


def parse_arguments(argument_lst):
    """
    Simple function to parse command line arguments

    Args:
        argument_lst (list): list of arguments as strings

    Returns:
        dict: dictionary with the parsed arguments and their corresponding values
    """
    return _update_default_dict_from_arguments(
        argument_lst=argument_lst,
        argument_dict={
            "total_cores": "--cores-total",
            "zmqport": "--zmqport",
            "cores_per_task": "--cores-per-task",
            "host": "--host",
        },
        default_dict={"host": "localhost"},
    )


def parse_socket_communication(executor, input_dict, future_dict, cores_per_task=1):
    if "shutdown" in input_dict.keys() and input_dict["shutdown"]:
        executor.shutdown(wait=input_dict["wait"])
        done_dict = _update_futures(future_dict=future_dict)
        # If close "shutdown" is communicated the process is shutdown.
        if done_dict is not None and len(done_dict) > 0:
            return {"exit": True, "result": done_dict}
        else:
            return {"exit": True}
    elif "fn" in input_dict.keys() and "iterable" in input_dict.keys():
        # If a function "fn" and a list or arguments "iterable" are communicated,
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
        # If a function "fn", arguments "args" and keyword arguments "kwargs" are
        # communicated pympipool uses submit() to asynchronously apply the function
        # on the arguments and or keyword arguments.
        future = call_funct(input_dict=input_dict, funct=executor.submit)
        future_hash = hash(future)
        future_dict[future_hash] = future
        return {"result": future_hash}
    elif "update" in input_dict.keys():
        # If update "update" is communicated pympipool checks for asynchronously submitted
        # functions which have completed in the meantime and communicates their results.
        done_dict = _update_futures(
            future_dict=future_dict, hash_lst=input_dict["update"]
        )
        return {"result": done_dict}
    elif "cancel" in input_dict.keys():
        for k in input_dict["cancel"]:
            future_dict[k].cancel()
        return {"result": True}


def _update_futures(future_dict, hash_lst=None):
    if hash_lst is None:
        hash_lst = list(future_dict.keys())
    done_dict = {
        k: f.result()
        for k, f in {k: future_dict[k] for k in hash_lst}.items()
        if f.done()
    }
    for k in done_dict.keys():
        del future_dict[k]
    return done_dict


def _wrap(funct, number_of_cores_per_communicator=1):
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
