from concurrent.futures import Future
from typing import Any, Optional


class FutureSelector(Future):
    def __init__(self, future: Future, selector: int | str):
        self._future = future
        self._selector = selector
        super().__init__()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._future, attr)

    def __setattr__(self, name: str, value: Any):
        if name in ["_future", "_selector"]:
            super().__setattr__(name, value)
        else:
            setattr(self._future, name, value)

    def result(self, timeout: Optional[float] = None) -> Any:
        result = self._future.result(timeout=timeout)
        if result is not None:
            return result[self._selector]
        else:
            return None


def split_future(future: Future, n: int) -> list[FutureSelector]:
    """
    Split a concurrent.futures.Future object which returns a tuple or list as result into individual future objects

    Args:
        future (Future): future object which returns a tuple or list as result
        n: number of elements expected in the future object

    Returns:
        list: List of future objects
    """
    return [FutureSelector(future=future, selector=i) for i in range(n)]


def get_item_from_future(future: Future, key: str) -> FutureSelector:
    """
    Get item from concurrent.futures.Future object which returns a dictionary as result by the corresponding dictionary
    key.

    Args:
        future (Future): future object which returns a dictionary as result
        key (str): dictionary key to get item from dictionary

    Returns:
        FutureSelector: Future object which returns the value corresponding to the key
    """
    return FutureSelector(future=future, selector=key)
