from concurrent.futures import Future
from typing import Any, Optional


class FutureSelector(Future):
    def __init__(self, future: Future, selector: int | str):
        super().__init__()
        self._future = future
        self._selector = selector

    def cancel(self) -> bool:
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def running(self) -> bool:
        return self._future.running()

    def done(self) -> bool:
        return self._future.done()

    def add_done_callback(self, fn) -> None:
        return self._future.add_done_callback(fn=fn)

    def result(self, timeout: Optional[float] = None) -> Any:
        result = self._future.result(timeout=timeout)
        if result is not None:
            return result[self._selector]
        else:
            return None

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        return self._future.exception(timeout=timeout)

    def set_running_or_notify_cancel(self) -> bool:
        return self._future.set_running_or_notify_cancel()

    def set_result(self, result: Any) -> None:
        return self._future.set_result(result=result)

    def set_exception(self, exception: Optional[BaseException]) -> None:
        return self._future.set_exception(exception=exception)


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
