from executorlib._version import get_versions as _get_versions
from executorlib.interfaces.flux import (
    FluxClusterExecutor,
    FluxJobExecutor,
)
from executorlib.interfaces.single import SingleNodeExecutor
from executorlib.interfaces.slurm import (
    SlurmClusterExecutor,
    SlurmJobExecutor,
)

_hdf_lst: list = []

try:
    from executorlib.standalone.hdf import get_cache_data
except ImportError:
    pass
else:
    _hdf_lst += [get_cache_data]

__version__ = _get_versions()["version"]
__all__: list = [
    "FluxJobExecutor",
    "FluxClusterExecutor",
    "SingleNodeExecutor",
    "SlurmJobExecutor",
    "SlurmClusterExecutor",
]
__all__ += _hdf_lst
