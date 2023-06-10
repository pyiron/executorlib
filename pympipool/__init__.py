from pympipool.share.pool import Pool, PoolExtended
from pympipool.share.executor import SingleTaskExecutor
from pympipool.share.communication import SocketInterface

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
