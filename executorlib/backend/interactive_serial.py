import sys
from os.path import abspath
from typing import Optional

from executorlib.standalone.error import backend_write_error_file
from executorlib.standalone.interactive.backend import call_funct, parse_arguments
from executorlib.standalone.interactive.communication import (
    interface_connect,
    interface_receive,
    interface_send,
    interface_shutdown,
)


def main(argument_lst: Optional[list[str]] = None):
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

    memory = {"executorlib_worker_id": int(argument_dict["worker_id"])}

    # required for flux interface - otherwise the current path is not included in the python path
    cwd = abspath(".")
    if cwd not in sys.path:
        sys.path.insert(1, cwd)

    while True:
        # Read from socket
        input_dict = interface_receive(socket=socket)

        # Parse input
        if "shutdown" in input_dict and input_dict["shutdown"]:
            interface_send(socket=socket, result_dict={"result": True})
            interface_shutdown(socket=socket, context=context)
            break
        elif (
            "fn" in input_dict
            and "init" not in input_dict
            and "args" in input_dict
            and "kwargs" in input_dict
        ):
            # Execute function
            try:
                output = call_funct(input_dict=input_dict, funct=None, memory=memory)
            except Exception as error:
                interface_send(
                    socket=socket,
                    result_dict={"error": error},
                )
                backend_write_error_file(
                    error=error,
                    apply_dict=input_dict,
                )
            else:
                # Send output
                interface_send(socket=socket, result_dict={"result": output})
        elif (
            "init" in input_dict
            and input_dict["init"]
            and "args" in input_dict
            and "kwargs" in input_dict
        ):
            memory.update(call_funct(input_dict=input_dict, funct=None, memory=memory))


if __name__ == "__main__":
    main(argument_lst=sys.argv)
