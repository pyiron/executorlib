import warnings
from typing import Optional

from pydantic import BaseModel


class ResourceDictValidation(BaseModel):
    cores: Optional[int] = None
    threads_per_core: Optional[int] = None
    gpus_per_core: Optional[int] = None
    cwd: Optional[str] = None
    num_nodes: Optional[int] = None
    exclusive: Optional[bool] = None
    error_log_file: Optional[str] = None
    restart_limit: Optional[int] = None
    run_time_limit: Optional[int] = None
    priority: Optional[int] = None
    openmpi_oversubscribe: Optional[bool] = None
    pmi_mode: Optional[str] = None
    flux_executor_nesting: Optional[bool] = None
    flux_log_files: Optional[bool] = None
    slurm_cmd_args: Optional[list[str]] = None


def validate_resource_dict(resource_dict: dict) -> None:
    _ = ResourceDictValidation(**resource_dict)


def validate_resource_dict_with_optional_keys(resource_dict: dict) -> None:
    accepted_keys = ResourceDictValidation.model_fields.keys()
    optional_lst = [key for key in resource_dict if key not in accepted_keys]
    validate_dict = {
        key: value for key, value in resource_dict.items() if key in accepted_keys
    }
    _ = ResourceDictValidation(**validate_dict)
    warnings.warn(
        f"The following keys are not recognized and cannot be validated: {optional_lst}"
    )
