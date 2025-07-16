def backend_write_error_file(error: Exception, apply_dict: dict) -> None:
    """
    Write an error to a file if specified in the apply_dict.

    Args:
        error (Exception): The error to be written.
        apply_dict (dict): Dictionary containing additional parameters.

    Returns:
        None
    """
    if apply_dict.get("write_error_file", False):
        with open(apply_dict.get("error_file_name", "error.out"), "a") as f:
            if hasattr(error, "output"):
                f.write(error.output)
            else:
                f.write(str(error))
