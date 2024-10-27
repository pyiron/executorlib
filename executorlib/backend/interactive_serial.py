import sys
from os.path import abspath
from typing import List, Optional

from executorlib.standalone.interactive.backend import call_funct, parse_arguments
from executorlib.standalone.interactive.communication import (
    interface_connect,
    interface_receive,
    interface_send,
    interface_shutdown,
)


def main(argument_lst: Optional[List[str]] = None):
    """
    The main function of the program.

    Args:
        argument_lst (Optional[List[str]]): List of command line arguments. If None, sys.argv will be used.

    Returns:
        None
    """
    if argument_lst is None:
        argument_lst = sys.argv
    argument_dict = parse_arguments(argument_lst=argument_lst)
    context, socket = interface_connect(
        host=argument_dict["host"], port=argument_dict["zmqport"]
    )

    memory = None

    # required for flux interface - otherwise the current path is not included in the python path
    cwd = abspath(".")
    if cwd not in sys.path:
        sys.path.insert(1, cwd)

    while True:
        # Read from socket
        input_dict = interface_receive(socket=socket)

        # Parse input
        if "shutdown" in input_dict.keys() and input_dict["shutdown"]:
            interface_send(socket=socket, result_dict={"result": True})
            interface_shutdown(socket=socket, context=context)
            break
        elif (
            "fn" in input_dict.keys()
            and "init" not in input_dict.keys()
            and "args" in input_dict.keys()
            and "kwargs" in input_dict.keys()
        ):
            # Execute function
            try:
                output = call_funct(input_dict=input_dict, funct=None, memory=memory)
            except Exception as error:
                interface_send(
                    socket=socket,
                    result_dict={"error": error, "error_type": str(type(error))},
                )
            else:
                # Send output
                interface_send(socket=socket, result_dict={"result": output})
        elif (
            "init" in input_dict.keys()
            and input_dict["init"]
            and "args" in input_dict.keys()
            and "kwargs" in input_dict.keys()
        ):
            memory = call_funct(input_dict=input_dict, funct=None)


if __name__ == "__main__":
    main(argument_lst=sys.argv)
