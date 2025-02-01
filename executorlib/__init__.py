from executorlib._version import get_versions as _get_versions
from executorlib.interfaces.flux import (
    FluxAllocationExecutor,
    FluxSubmissionExecutor,
)
from executorlib.interfaces.single import SingleNodeExecutor
from executorlib.interfaces.slurm import (
    SlurmAllocationExecutor,
    SlurmSubmissionExecutor,
)

__version__ = _get_versions()["version"]
__all__: list = [
    "FluxAllocationExecutor",
    "FluxSubmissionExecutor",
    "SingleNodeExecutor",
    "SlurmAllocationExecutor",
    "SlurmSubmissionExecutor",
]
