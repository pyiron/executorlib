from pympipool.share.pool import Pool, MPISpawnPool
from pympipool.share.executor import Executor, PoolExecutor
from pympipool.share.communication import (
    SocketInterface,
    connect_to_socket_interface,
    send_result,
    close_connection,
    receive_instruction,
)
from pympipool.share.serial import cancel_items_in_queue

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
