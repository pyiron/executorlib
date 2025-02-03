import queue


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
