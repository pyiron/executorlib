import inspect


def call_funct(input_dict, funct=None, memory=None):
    """
    Call function from dictionary

    Args:
        input_dict (dict): dictionary containing the function 'fn', its arguments 'args' and keyword arguments 'kwargs'
        funct (None): function to be evaluated if it is not included in the input dictionary
        memory (dict/ None): variables stored in memory which can be used as keyword arguments

    Returns:
        Result of the function
    """
    if funct is None:

        def funct(*args, **kwargs):
            return args[0].__call__(*args[1:], **kwargs)

    funct_args = inspect.getfullargspec(input_dict["fn"]).args
    if memory is not None:
        input_dict["kwargs"].update(
            _update_dict_delta(
                dict_input=memory,
                dict_output=input_dict["kwargs"],
                keys_possible_lst=funct_args,
            )
        )
    return funct(input_dict["fn"], *input_dict["args"], **input_dict["kwargs"])


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
            "zmqport": "--zmqport",
            "host": "--host",
        },
        default_dict={"host": "localhost"},
    )


def _update_default_dict_from_arguments(argument_lst, argument_dict, default_dict):
    default_dict.update(
        {
            k: argument_lst[argument_lst.index(v) + 1]
            for k, v in argument_dict.items()
            if v in argument_lst
        }
    )
    return default_dict


def _update_dict_delta(dict_input, dict_output, keys_possible_lst):
    return {
        k: v
        for k, v in dict_input.items()
        if k in keys_possible_lst and k not in dict_output.keys()
    }
