"""
Submodules in the executorlib.standalone module do not depend on other modules of the executorlib package. This strict
separation simplifies the development, testing and debugging. The functionality in executorlib.standalone is designed
to be used independently in other libraries.
"""

from executorlib.standalone.command import get_command_path
from executorlib.standalone.interactive.communication import (
    SocketInterface,
    interface_bootup,
    interface_connect,
    interface_receive,
    interface_send,
    interface_shutdown,
)
from executorlib.standalone.interactive.spawner import MpiExecSpawner, SubprocessSpawner
from executorlib.standalone.queue import cancel_items_in_queue
from executorlib.standalone.serialize import cloudpickle_register

__all__: list[str] = [
    "cancel_items_in_queue",
    "cloudpickle_register",
    "get_command_path",
    "interface_bootup",
    "interface_connect",
    "interface_receive",
    "interface_send",
    "interface_shutdown",
    "MpiExecSpawner",
    "SocketInterface",
    "SubprocessSpawner",
]
