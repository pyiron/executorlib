import traceback


def backend_write_error_file(error: Exception, apply_dict: dict) -> None:
    """
    Write an error to a file if specified in the apply_dict.

    Args:
        error (Exception): The error to be written.
        apply_dict (dict): Dictionary containing additional parameters.

    Returns:
        None
    """
    error_log_file = apply_dict.get("error_log_file")
    if error_log_file is not None:
        with open(error_log_file, "a") as f:
            f.write("function: " + str(apply_dict["fn"]) + "\n")
            f.write("args: " + str(apply_dict["args"]) + "\n")
            f.write("kwargs: " + str(apply_dict["kwargs"]) + "\n")
            traceback.print_exception(error, file=f)
