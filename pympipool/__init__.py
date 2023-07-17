from pympipool.external_interfaces.communication import (
    SocketInterface,
    connect_to_socket_interface,
    send_result,
    close_connection,
    receive_instruction,
)
from pympipool.external_interfaces.executor import Executor, PoolExecutor
from pympipool.external_interfaces.pool import Pool, MPISpawnPool
from pympipool.external_interfaces.thread import RaisingThread
from pympipool.shared_functions.external_interfaces import cancel_items_in_queue

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
