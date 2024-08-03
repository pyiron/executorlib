from concurrent.futures import Future
from typing import Tuple

import cloudpickle


def generate_nodes_and_edges(
    task_hash_dict: dict, future_hash_inverse_dict: dict
) -> Tuple[list, list]:
    """
    Generate nodes and edges for visualization.

    Args:
        task_hash_dict (dict): Dictionary mapping task hash to task information.
        future_hash_inverse_dict (dict): Dictionary mapping future hash to future object.

    Returns:
        Tuple[list, list]: Tuple containing the list of nodes and the list of edges.
    """
    node_lst, edge_lst = [], []
    hash_id_dict = {}

    def add_element(arg, link_to, label=""):
        """
        Add element to the node and edge lists.

        Args:
            arg: Argument to be added.
            link_to: ID of the node to link the element to.
            label (str, optional): Label for the edge. Defaults to "".
        """
        if isinstance(arg, Future):
            edge_lst.append(
                {
                    "start": hash_id_dict[future_hash_inverse_dict[arg]],
                    "end": link_to,
                    "label": label,
                }
            )
        elif isinstance(arg, list) and all([isinstance(a, Future) for a in arg]):
            for a in arg:
                add_element(arg=a, link_to=link_to, label=label)
        else:
            node_id = len(node_lst)
            node_lst.append({"name": str(arg), "id": node_id, "shape": "circle"})
            edge_lst.append({"start": node_id, "end": link_to, "label": label})

    for k, v in task_hash_dict.items():
        hash_id_dict[k] = len(node_lst)
        node_lst.append(
            {"name": v["fn"].__name__, "id": hash_id_dict[k], "shape": "box"}
        )
    for k, task_dict in task_hash_dict.items():
        for arg in task_dict["args"]:
            add_element(arg=arg, link_to=hash_id_dict[k], label="")

        for kw, v in task_dict["kwargs"].items():
            add_element(arg=v, link_to=hash_id_dict[k], label=str(kw))

    return node_lst, edge_lst


def generate_task_hash(task_dict: dict, future_hash_inverse_dict: dict) -> bytes:
    """
    Generate a hash for a task dictionary.

    Args:
        task_dict (dict): Dictionary containing task information.
        future_hash_inverse_dict (dict): Dictionary mapping future hash to future object.

    Returns:
        bytes: Hash generated for the task dictionary.
    """

    def convert_arg(arg, future_hash_inverse_dict):
        """
        Convert an argument to its hash representation.

        Args:
            arg: Argument to be converted.
            future_hash_inverse_dict (dict): Dictionary mapping future hash to future object.

        Returns:
            The hash representation of the argument.
        """
        if isinstance(arg, Future):
            return future_hash_inverse_dict[arg]
        elif isinstance(arg, list):
            return [
                convert_arg(arg=a, future_hash_inverse_dict=future_hash_inverse_dict)
                for a in arg
            ]
        else:
            return arg

    args_for_hash = [
        convert_arg(arg=arg, future_hash_inverse_dict=future_hash_inverse_dict)
        for arg in task_dict["args"]
    ]
    kwargs_for_hash = {
        k: convert_arg(arg=v, future_hash_inverse_dict=future_hash_inverse_dict)
        for k, v in task_dict["kwargs"].items()
    }
    return cloudpickle.dumps(
        {"fn": task_dict["fn"], "args": args_for_hash, "kwargs": kwargs_for_hash}
    )


def draw(node_lst: list, edge_lst: list):
    """
    Draw the graph visualization of nodes and edges.

    Args:
        node_lst (list): List of nodes.
        edge_lst (list): List of edges.
    """
    from IPython.display import SVG, display  # noqa
    import matplotlib.pyplot as plt  # noqa
    import networkx as nx  # noqa

    graph = nx.DiGraph()
    for node in node_lst:
        graph.add_node(node["id"], label=node["name"], shape=node["shape"])
    for edge in edge_lst:
        graph.add_edge(edge["start"], edge["end"], label=edge["label"])
    svg = nx.nx_agraph.to_agraph(graph).draw(prog="dot", format="svg")
    display(SVG(svg))
    plt.show()
