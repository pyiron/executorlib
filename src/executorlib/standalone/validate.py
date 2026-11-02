import warnings

try:
    from pydantic import BaseModel, Extra

    HAS_PYDANTIC = True
except ImportError:
    from dataclasses import dataclass

    BaseModel = object
    Extra = None
    HAS_PYDANTIC = False


class ResourceDictValidation(BaseModel):
    """
    Pydantic (or dataclass fallback) model for validating resource dictionaries passed to task submissions.

    Attributes:
        cores (int, optional): Number of MPI cores to be used for each function call.
        threads_per_core (int, optional): Number of OpenMP threads to be used for each function call.
        gpus_per_core (int, optional): Number of GPUs per worker.
        cwd (str, optional): Current working directory where the parallel python task is executed.
        cache_key (str, optional): External cache key to identify tasks on the file system.
        cache_directory (str, optional): The directory to store cache files.
        num_nodes (int, optional): Number of compute nodes used for the evaluation.
        exclusive (bool, optional): Reserve exclusive access to selected compute nodes.
        error_log_file (str, optional): Path to the error log file.
        run_time_max (int, optional): Maximum allowed execution time in seconds.
        priority (int, optional): Queuing system priority for the task.
        slurm_cmd_args (list[str], optional): Additional command line arguments for the srun call.
        submission_template (str, optional): Template for queuing system job submission scripts.
    """

    cores: int | None = None
    threads_per_core: int | None = None
    gpus_per_core: int | None = None
    cwd: str | None = None
    cache_key: str | None = None
    cache_directory: str | None = None
    num_nodes: int | None = None
    exclusive: bool | None = None
    error_log_file: str | None = None
    run_time_max: int | None = None
    priority: int | None = None
    slurm_cmd_args: list[str] | None = None
    submission_template: str | None = None

    if HAS_PYDANTIC:

        class Config:
            extra = Extra.forbid


if not HAS_PYDANTIC:
    ResourceDictValidation = dataclass(ResourceDictValidation)  # type: ignore


def _get_accepted_keys(class_type) -> list[str]:
    """
    Return a list of accepted field names from a Pydantic model or dataclass.

    Args:
        class_type: A Pydantic BaseModel subclass or a dataclass.

    Returns:
        list[str]: Field names declared on the class.

    Raises:
        TypeError: If the class type is neither a Pydantic model nor a dataclass.
    """
    if hasattr(class_type, "model_fields"):
        return list(class_type.model_fields.keys())
    elif hasattr(class_type, "__dataclass_fields__"):
        return list(class_type.__dataclass_fields__.keys())
    raise TypeError("Unsupported class type for validation")


def validate_resource_dict(resource_dict: dict) -> None:
    """
    Validate a resource dictionary against the declared fields of ResourceDictValidation.

    Args:
        resource_dict (dict): Dictionary of resource requirements to validate. Unknown keys raise
            a validation error when pydantic is available.

    Raises:
        ValidationError: If any key/value pair violates the ResourceDictValidation schema.
    """
    _ = ResourceDictValidation(**resource_dict)


def validate_resource_dict_with_optional_keys(resource_dict: dict) -> None:
    """
    Validate a resource dictionary, allowing unknown keys with a warning instead of an error.

    Known keys are validated against ResourceDictValidation. Unknown keys are collected and
    emitted as a UserWarning so callers can detect unsupported options without hard failure.

    Args:
        resource_dict (dict): Dictionary of resource requirements to validate.

    Raises:
        ValidationError: If any known key/value pair violates the ResourceDictValidation schema.
    """
    accepted_keys = _get_accepted_keys(class_type=ResourceDictValidation)
    optional_lst = [key for key in resource_dict if key not in accepted_keys]
    validate_dict = {
        key: value for key, value in resource_dict.items() if key in accepted_keys
    }
    _ = ResourceDictValidation(**validate_dict)
    if len(optional_lst) > 0:
        warnings.warn(
            f"The following keys are not recognized and cannot be validated: {optional_lst}",
            stacklevel=2,
        )
