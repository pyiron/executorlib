from concurrent.futures import Future


def batched_futures(lst: list[Future], skip_set: set, n: int) -> list[list]:
    """
    Batch n completed future objects. If the number of completed futures is smaller than n and the end of the batch is
    not reached yet, then an empty list is returned. If n future objects are done, which are not included in the skip_set
    then they are returned as batch.

    Args:
        lst (list): list of all future objects
        skip_set (set): flat set of individual results already assigned to previous batches
        n (int): batch size

    Returns:
        list: results of the batched futures
    """
    done_lst = []
    n_expected = min(n, len(lst) - len(skip_set))
    for v in lst:
        if v.done() and v.result() not in skip_set:
            done_lst.append(v.result())
        if len(done_lst) == n_expected:
            return done_lst
    return []
