from executorlib._version import get_versions as _get_versions
from executorlib.interfaces.flux import (
    FluxJobExecutor,
    FluxClusterExecutor,
)
from executorlib.interfaces.single import SingleNodeExecutor
from executorlib.interfaces.slurm import (
    SlurmJobExecutor,
    SlurmClusterExecutor,
)

__version__ = _get_versions()["version"]
__all__: list = [
    "FluxJobExecutor",
    "FluxClusterExecutor",
    "SingleNodeExecutor",
    "SlurmJobExecutor",
    "SlurmClusterExecutor",
]
