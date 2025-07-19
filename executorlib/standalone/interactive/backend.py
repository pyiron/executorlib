import inspect
from typing import Any, Callable, Optional


def call_funct(
    input_dict: dict, funct: Optional[Callable] = None, memory: Optional[dict] = None
) -> Any:
    """
    Call function from dictionary

    Args:
        input_dict (dict): dictionary containing the function 'fn', its arguments 'args' and keyword arguments 'kwargs'
        funct (Callable, optional): function to be evaluated if it is not included in the input dictionary
        memory (dict, optional): variables stored in memory which can be used as keyword arguments

    Returns:
        Any: Result of the function
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


def parse_arguments(argument_lst: list[str]) -> dict:
    """
    Simple function to parse command line arguments

    Args:
        argument_lst (list): list of arguments as strings

    Returns:
        dict: dictionary with the parsed arguments and their corresponding values
    """
    return update_default_dict_from_arguments(
        argument_lst=argument_lst,
        argument_dict={
            "zmqport": "--zmqport",
            "host": "--host",
            "worker_id": "--worker-id",
        },
        default_dict={"host": "localhost", "worker_id": 0},
    )


def update_default_dict_from_arguments(
    argument_lst: list[str], argument_dict: dict, default_dict: dict
) -> dict:
    """
    Update default dictionary with values from command line arguments

    Args:
        argument_lst (list[str]): List of arguments as strings
        argument_dict (dict): Dictionary mapping argument names to their corresponding command line flags
        default_dict (dict): Default dictionary to be updated

    Returns:
        dict: Updated default dictionary
    """
    default_dict.update(
        {
            k: argument_lst[argument_lst.index(v) + 1]
            for k, v in argument_dict.items()
            if v in argument_lst
        }
    )
    return default_dict


def _update_dict_delta(
    dict_input: dict, dict_output: dict, keys_possible_lst: list[str]
) -> dict:
    """
    Update dictionary with values from another dictionary, only if the keys are present in a given list

    Args:
        dict_input (dict): Input dictionary
        dict_output (dict): Output dictionary to be updated
        keys_possible_lst (list[str]): List of possible keys to be updated

    Returns:
        dict: Updated dictionary
    """
    return {
        k: v
        for k, v in dict_input.items()
        if k in keys_possible_lst and k not in dict_output
    }
