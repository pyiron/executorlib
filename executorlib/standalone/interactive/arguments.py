from asyncio.exceptions import CancelledError
from concurrent.futures import Future, TimeoutError
from typing import Any, Union


def get_future_objects_from_input(args: tuple, kwargs: dict):
    """
    Check the input parameters if they contain future objects and which of these future objects are executed

    Args:
        args (tuple): function arguments
        kwargs (dict): function keyword arguments

    Returns:
        list, boolean: list of future objects and boolean flag if all future objects are already done
    """
    future_lst = []

    def find_future_in_list(lst):
        for el in lst:
            if isinstance(el, Future):
                future_lst.append(el)
            elif isinstance(el, list):
                find_future_in_list(lst=el)
            elif isinstance(el, dict):
                find_future_in_list(lst=el.values())

    find_future_in_list(lst=args)
    find_future_in_list(lst=kwargs.values())
    boolean_flag = len([future for future in future_lst if future.done()]) == len(
        future_lst
    )
    return future_lst, boolean_flag


def get_exception_lst(future_lst: list[Future]) -> list:
    """
    Get list of exceptions raised by the future objects in the list of future objects

    Args:
        future_lst (list): list of future objects

    Returns:
        list: list of exceptions raised by the future objects in the list of future objects. Returns empty list if no
              exception was raised.
    """
    return [
        f.exception() for f in future_lst if check_exception_was_raised(future_obj=f)
    ]


def check_exception_was_raised(future_obj: Future) -> bool:
    """
    Check if exception was raised by future object

    Args:
        future_obj (Future): future object

    Returns:
        bool: True if exception was raised, False otherwise
    """
    try:
        excp = future_obj.exception(timeout=10**-10)
        return excp is not None and not isinstance(excp, CancelledError)
    except TimeoutError:
        return False


def update_futures_in_input(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """
    Evaluate future objects in the arguments and keyword arguments by calling future.result()

    Args:
        args (tuple): function arguments
        kwargs (dict): function keyword arguments

    Returns:
        tuple, dict: arguments and keyword arguments with each future object in them being evaluated
    """

    def get_result(arg: Union[list[Future], Future]) -> Any:
        if isinstance(arg, Future):
            return arg.result()
        elif isinstance(arg, list):
            return [get_result(arg=el) for el in arg]
        elif isinstance(arg, dict):
            return {k: get_result(arg=v) for k, v in arg.items()}
        else:
            return arg

    args = tuple([get_result(arg=arg) for arg in args])
    kwargs = {key: get_result(arg=value) for key, value in kwargs.items()}
    return args, kwargs
