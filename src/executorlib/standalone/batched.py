from concurrent.futures import Future


def batched_futures(
    lst: list[Future], nested_skip_lst: list[Future[list]], n: int
) -> tuple[bool, list[Future]]:
    """
    Batch n completed future objects. If the number of completed futures is smaller than n and the end of the batch is
    not reached yet, then an empty list is returned. If n future objects are done, which are not included in the skip_set
    then they are returned as batch.

    Args:
        lst (list): list of all future objects
        nested_skip_lst (list): list of future objects, which contain the list of future objects ids which should be skipped for the batch
        n (int): batch size

    Returns:
        list: results of the batched futures
    """
    skip_set = {fid for f in nested_skip_lst for fid in f.result()}

    done_lst = []
    failed_lst = []
    n_expected = min(n, len(lst) - len(skip_set))
    for v in lst:
        if id(v) not in skip_set and v.done():
            if v.exception() is not None:
                failed_lst.append(v)
            elif id(v) not in skip_set and v.done():
                done_lst.append(v)
                if len(done_lst) == n_expected:
                    return True, done_lst
    if (len(lst) - len(skip_set)) == len(failed_lst):
        return False, failed_lst[:n_expected]  # raise the exception only after all futures have failed
    else:
        return True, []
