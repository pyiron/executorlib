"""
External application programming interface (API) following the semantic versioning this interface is promised to remain
stable during minor releases and any change in the interface leads to a major version bump. External libraries should
only use the functionality in this API in combination with the user interface defined in the root __init__.py, all other
functionality is considered internal and might change during minor releases.
"""

from executorlib.executor.single import TestClusterExecutor
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
    "TestClusterExecutor",
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
