from concurrent.futures import Future


def batched_futures(lst: list[Future], skip_lst: list[list], n: int) -> list[list]:
    """
    Batch n completed future objects. If the number of completed futures is smaller than n and the end of the batch is
    not reached yet, then an empty list is returned. If n future objects are done, which are not included in the skip_lst
    then they are returned as batch.

    Args:
        lst (list): list of all future objects
        skip_lst (list): list of previous batches of future objects
        n (int): batch size

    Returns:
        list: results of the batched futures
    """
    skipped_ids = {id(item) for items in skip_lst for item in items}

    done_lst = []
    n_expected = min(n, len(lst) - len(skipped_ids))
    for v in lst:
        if v.done() and id(v.result()) not in skipped_ids:
            done_lst.append(v.result())
            if len(done_lst) == n_expected:
                return done_lst
    return []
