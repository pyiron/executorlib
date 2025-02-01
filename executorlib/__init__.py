from executorlib._version import get_versions as _get_versions
from executorlib.interfaces import (
    FluxAllocationExecutor,
    FluxSubmissionExecutor,
    LocalExecutor,
    SlurmAllocationExecutor,
    SlurmSubmissionExecutor,
)

__version__ = _get_versions()["version"]
__all__: list = []
