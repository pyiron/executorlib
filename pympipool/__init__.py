from pympipool.share.pool import Pool
from pympipool.share.executor import Executor
from pympipool.share.communication import SocketInterface

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
