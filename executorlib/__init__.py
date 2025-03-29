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

__all__: list[str] = [
    "FluxJobExecutor",
    "FluxClusterExecutor",
    "SingleNodeExecutor",
    "SlurmJobExecutor",
    "SlurmClusterExecutor",
]

try:
    from executorlib.standalone.hdf import get_cache_data
except ImportError:
    pass
else:
    __all__ += ["get_cache_data"]

__version__ = _get_versions()["version"]
