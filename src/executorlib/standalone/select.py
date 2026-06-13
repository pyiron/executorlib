from concurrent.futures import Future
from typing import Any, Optional


class FutureSelector(Future):
    """
    A Future wrapper that returns a single element from a result that is a tuple, list, or dict.

    This is used by split_future() and get_item_from_future() to give callers an individual
    Future-like object for each output of a function that returns a collection.

    Args:
        future (Future): The underlying future whose result is a collection.
        selector (int | str): Index (for sequences) or key (for mappings) used to extract
            the desired element from the result.
    """

    def __init__(self, future: Future[Any], selector: int | str):
        """
        Args:
            future (Future): The underlying future whose result is a collection.
            selector (int | str): Index or key used to extract an element from the result.
        """
        self._future = future
        self._selector = selector
        super().__init__()

    def __getattr__(self, attr: str) -> Any:
        """Delegate attribute access to the wrapped future."""
        return getattr(self._future, attr)

    def __setattr__(self, name: str, value: Any):
        """Set attributes on the wrapped future, except for the two private instance variables."""
        if name in ["_future", "_selector"]:
            super().__setattr__(name, value)
        else:
            setattr(self._future, name, value)

    def result(self, timeout: Optional[float] = None) -> Any:
        """
        Return the selected element from the underlying future's result.

        Args:
            timeout (float, optional): Maximum seconds to wait for the result. Defaults to None
                (wait indefinitely).

        Returns:
            Any: The element at position/key ``selector`` in the underlying result, or None if
                the underlying result is None.
        """
        result = self._future.result(timeout=timeout)
        if result is not None:
            if (
                isinstance(self._selector, int)
                and isinstance(result, (tuple, list))
                or isinstance(result, dict)
                and self._selector in result
            ):
                return result[self._selector]
            else:
                raise KeyError(
                    str(self._selector)
                    + " of type "
                    + str(type(self._selector))
                    + " is not in "
                    + str(result)
                    + " of type "
                    + str(type(result))
                )
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
