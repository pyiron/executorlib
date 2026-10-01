import queue
from typing import Any


def put_front(que: queue.Queue, item: Any):
    """
    Insert an item at the front of the queue, ahead of any items already waiting, and wake a consumer blocked in
    get(). Equivalent to queue.Queue.put() except the item is placed at the front rather than the back; mutating
    que.queue directly instead would silently skip the notification, leaving a waiting consumer asleep forever.

    Args:
        que (queue.Queue): Queue with task objects which should be executed
        item (Any): item to place at the front of the queue
    """
    with que.not_full:
        que.queue.insert(0, item)
        que.unfinished_tasks += 1
        que.not_empty.notify()


def cancel_items_in_queue(que: queue.Queue):
    """
    Cancel items which are still waiting in the queue. If the executor is busy tasks remain in the queue, so the future
    objects have to be cancelled when the executor shuts down.

    Args:
        que (queue.Queue): Queue with task objects which should be executed
    """
    while True:
        try:
            item = que.get_nowait()
            if isinstance(item, dict) and "future" in item:
                item["future"].cancel()
                que.task_done()
        except queue.Empty:
            break
