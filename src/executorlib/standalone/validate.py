import warnings
from typing import Optional

from pydantic import BaseModel, Extra


class ResourceDictValidation(BaseModel):
    cores: Optional[int] = None
    threads_per_core: Optional[int] = None
    gpus_per_core: Optional[int] = None
    cwd: Optional[str] = None
    cache_key: Optional[str] = None
    num_nodes: Optional[int] = None
    exclusive: Optional[bool] = None
    error_log_file: Optional[str] = None
    run_time_limit: Optional[int] = None
    priority: Optional[int] = None

    class Config:
        extra = Extra.forbid


def validate_resource_dict(resource_dict: dict) -> None:
    _ = ResourceDictValidation(**resource_dict)


def validate_resource_dict_with_optional_keys(resource_dict: dict) -> None:
    accepted_keys = ResourceDictValidation.model_fields.keys()
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
