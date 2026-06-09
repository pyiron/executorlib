from concurrent.futures import Future

# Future objects we have already reported as failed -- so each failed job is logged once, not on
# every scheduler pass (batched_futures is re-evaluated many times until a batch fills).
_logged_failed_ids: set = set()


def batched_futures(lst: list[Future], skip_lst: list[list], n: int) -> list[list]:
    """
    Batch n completed future objects. If the number of completed futures is smaller than n and the end of the batch is
    not reached yet, then an empty list is returned. If n future objects are done, which are not included in the skip_lst
    then they are returned as batch.

    Futures that completed with an EXCEPTION (e.g. a labeling job that failed on a degenerate config, or a dead worker)
    are EXCLUDED from the batch rather than re-raised. Calling ``.result()`` on a failed future re-raises its exception;
    in the dependency scheduler that turns into ``set_exception`` on this batch future, which then cascades to every
    downstream task (combine_b / featurize / fit / cost / pareto) depending on the batch -- i.e. a single bad config
    silently kills the whole pipeline. Each failed future is logged once. When the entire input is resolved but a full
    batch of n cannot be formed (because some futures failed), the partial remainder is returned so the pipeline does
    not stall forever waiting for a batch that can never fill.

    Args:
        lst (list): list of all future objects
        skip_lst (list): list of previous batches of future objects
        n (int): batch size

    Returns:
        list: results of the batched futures
    """
    skipped_ids = {id(item) for items in skip_lst for item in items}

    done_lst: list = []
    all_resolved = True
    for v in lst:
        if v.done():
            if v.exception() is not None:
                if id(v) not in _logged_failed_ids:
                    _logged_failed_ids.add(id(v))
                    print(
                        f"[batched_futures] EXCLUDING failed future from batch: "
                        f"{type(v.exception()).__name__}: {v.exception()}",
                        flush=True,
                    )
                continue  # failed future: exclude instead of re-raising (which would poison all dependents)
            result = v.result()
            if id(result) not in skipped_ids:
                done_lst.append(result)
                if len(done_lst) == n:
                    return done_lst
        else:
            all_resolved = False
    if all_resolved and len(done_lst) > 0:
        return done_lst  # end of input reached; emit final (possibly short) batch
    return []
