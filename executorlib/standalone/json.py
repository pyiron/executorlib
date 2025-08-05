import os
from typing import Any, Optional

import jsonpickle


def dump_to_json(file_name: Optional[str], data_dict: dict) -> None:
    """
    Dump data dictionary into JSON file

    Args:
        file_name (str): file name of the JSON file as absolute path
        data_dict (dict): dictionary containing the python function to be executed {"fn": ..., "args": (), "kwargs": {}}
    """
    if file_name is not None:
        file_name_abs = os.path.abspath(file_name)
        os.makedirs(os.path.dirname(file_name_abs), exist_ok=True)
        if os.path.exists(file_name_abs):
            json_content = _read_json(file_name=file_name_abs)
            data_dict.update(
                {k: v for k, v in json_content.items() if k not in data_dict}
            )
        with open(file_name_abs, "w+") as f:
            f.write(jsonpickle.encode(data_dict))


def load_from_json(file_name: str) -> dict:
    """
    Load data dictionary from JSON file

    Args:
        file_name (str): file name of the JSON file as absolute path

    Returns:
        dict: dictionary containing the python function to be executed {"fn": ..., "args": (), "kwargs": {}}
    """
    default_dict = {
        "args": (),
        "kwargs": {},
    }
    json_content = _read_json(file_name=file_name)
    json_content.update({k:v for k, v in default_dict.items() if k not in json_content})
    if "fn" not in json_content:
        raise TypeError("Function not found in JSON file.")
    return json_content


def get_output_from_json(file_name: str) -> tuple[bool, bool, Any]:
    """
    Check if output is available in the JSON file

    Args:
        file_name (str): file name of the JSON file as absolute path

    Returns:
        Tuple[bool, bool, object]: boolean flag indicating if output is available and the output object itself
    """
    json_content = _read_json(file_name=file_name)
    if "output" in json_content:
        return True, True, json_content["output"]
    elif "error" in json_content:
        return True, False, json_content["error"]
    else:
        return False, False, None


def _read_json(file_name: str) -> dict:
    with open(file_name, "r+") as f:
        return jsonpickle.decode(f.read())