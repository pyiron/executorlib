import inspect
import json
import os.path
from concurrent.futures import Future
from typing import Optional

import cloudpickle
import numpy as np

from executorlib.standalone.select import FutureSelector


def generate_nodes_and_edges_for_plotting(
    task_hash_dict: dict, future_hash_inverse_dict: dict
) -> tuple[list, list]:
    """
    Generate nodes and edges for visualization.

    Args:
        task_hash_dict (dict): Dictionary mapping task hash to task information.
        future_hash_inverse_dict (dict): Dictionary mapping future hash to future object.

    Returns:
        Tuple[list, list]: Tuple containing the list of nodes and the list of edges.
    """
    node_lst: list = []
    edge_lst: list = []
    hash_id_dict: dict = {}

    def extend_args(funct_dict):
        sig = inspect.signature(funct_dict["fn"])
        args = sig.bind(*funct_dict["args"], **funct_dict["kwargs"])
        funct_dict["signature"] = args.arguments
        return funct_dict

    def add_element(arg, link_to, label=""):
        """
        Add element to the node and edge lists.

        Args:
            arg: Argument to be added.
            link_to: ID of the node to link the element to.
            label (str, optional): Label for the edge. Defaults to "".
        """
        if isinstance(arg, FutureSelector):
            edge_lst.append(
                {
                    "start": hash_id_dict[future_hash_inverse_dict[arg._future]],
                    "end": link_to,
                    "label": label + str(arg._selector),
                    "end_label": label,
                    "start_label": str(arg._selector),
                }
            )
        elif isinstance(arg, Future):
            edge_lst.append(
                {
                    "start": hash_id_dict[future_hash_inverse_dict[arg]],
                    "end": link_to,
                    "label": label,
                }
            )
        elif isinstance(arg, list) and any(isinstance(a, Future) for a in arg):
            lst_no_future = [a if not isinstance(a, Future) else "$" for a in arg]
            node_id = len(node_lst)
            node_lst.append(
                {
                    "name": str(lst_no_future),
                    "value": "python_workflow_definition.shared.get_list",
                    "id": node_id,
                    "type": "function",
                    "shape": "box",
                }
            )
            edge_lst.append({"start": node_id, "end": link_to, "label": label})
            for i, a in enumerate(arg):
                if isinstance(a, Future):
                    add_element(arg=a, link_to=node_id, label=str(i))
        elif isinstance(arg, dict) and any(isinstance(a, Future) for a in arg.values()):
            dict_no_future = {
                kt: vt if not isinstance(vt, Future) else "$" for kt, vt in arg.items()
            }
            node_id = len(node_lst)
            node_lst.append(
                {
                    "name": str(dict_no_future),
                    "value": "python_workflow_definition.shared.get_dict",
                    "id": node_id,
                    "type": "function",
                    "shape": "box",
                }
            )
            edge_lst.append({"start": node_id, "end": link_to, "label": label})
            for kt, vt in arg.items():
                add_element(arg=vt, link_to=node_id, label=kt)
        else:
            value_dict = {
                str(n["value"]): n["id"] for n in node_lst if n["type"] == "input"
            }
            if str(arg) not in value_dict:
                node_id = len(node_lst)
                node_lst.append(
                    {
                        "name": label,
                        "value": arg,
                        "id": node_id,
                        "type": "input",
                        "shape": "circle",
                    }
                )
            else:
                node_id = value_dict[str(arg)]
            edge_lst.append({"start": node_id, "end": link_to, "label": label})

    task_hash_modified_dict = {
        k: extend_args(funct_dict=v) for k, v in task_hash_dict.items()
    }

    for k, v in task_hash_modified_dict.items():
        hash_id_dict[k] = len(node_lst)
        node_lst.append(
            {
                "name": v["fn"].__name__,
                "type": "function",
                "value": v["fn"].__module__ + "." + v["fn"].__name__,
                "id": hash_id_dict[k],
                "shape": "box",
            }
        )
    for k, task_dict in task_hash_modified_dict.items():
        for kw, v in task_dict["signature"].items():
            add_element(arg=v, link_to=hash_id_dict[k], label=str(kw))

    return node_lst, edge_lst


def generate_task_hash_for_plotting(task_dict: dict, future_hash_dict: dict) -> bytes:
    """
    Generate a hash for a task dictionary.

    Args:
        task_dict (dict): Dictionary containing task information.
        future_hash_dict (dict): Dictionary mapping future hash to future object.

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
        if isinstance(arg, FutureSelector):
            if arg not in future_hash_inverse_dict:
                obj_dict = {
                    "args": (),
                    "kwargs": {
                        "future": future_hash_inverse_dict[arg._future],
                        "selector": arg._selector,
                    },
                }
                if isinstance(arg._selector, str):
                    obj_dict["fn"] = "get_item_from_future"
                else:
                    obj_dict["fn"] = "split_future"
                arg_hash = cloudpickle.dumps(obj_dict)
                future_hash_dict[arg_hash] = arg
                future_hash_inverse_dict[arg] = arg_hash
            return future_hash_inverse_dict[arg]
        elif isinstance(arg, Future):
            return future_hash_inverse_dict[arg]
        elif isinstance(arg, list):
            return [
                convert_arg(arg=a, future_hash_inverse_dict=future_hash_inverse_dict)
                for a in arg
            ]
        elif isinstance(arg, dict):
            return {
                k: convert_arg(arg=v, future_hash_inverse_dict=future_hash_inverse_dict)
                for k, v in arg.items()
            }
        else:
            return arg

    future_hash_inverted_dict = {v: k for k, v in future_hash_dict.items()}
    args_for_hash = [
        convert_arg(arg=arg, future_hash_inverse_dict=future_hash_inverted_dict)
        for arg in task_dict["args"]
    ]
    kwargs_for_hash = {
        k: convert_arg(arg=v, future_hash_inverse_dict=future_hash_inverted_dict)
        for k, v in task_dict["kwargs"].items()
    }
    return cloudpickle.dumps(
        {"fn": task_dict["fn"], "args": args_for_hash, "kwargs": kwargs_for_hash}
    )


def plot_dependency_graph_function(
    node_lst: list, edge_lst: list, filename: Optional[str] = None
):
    """
    Draw the graph visualization of nodes and edges.

    Args:
        node_lst (list): List of nodes.
        edge_lst (list): List of edges.
        filename (str): Name of the file to store the plotted graph in.
    """
    import networkx as nx  # noqa

    graph = nx.DiGraph()
    for node in node_lst:
        if node["type"] == "input":
            graph.add_node(node["id"], label=str(node["value"]), shape=node["shape"])
        else:
            graph.add_node(node["id"], label=str(node["name"]), shape=node["shape"])
    for edge in edge_lst:
        graph.add_edge(edge["start"], edge["end"], label=edge["label"])
    if filename is not None:
        file_format = os.path.splitext(filename)[-1][1:]
        with open(filename, "wb") as f:
            f.write(nx.nx_agraph.to_agraph(graph).draw(prog="dot", format=file_format))
    else:
        from IPython.display import SVG, display  # noqa

        display(SVG(nx.nx_agraph.to_agraph(graph).draw(prog="dot", format="svg")))


def export_dependency_graph_function(
    node_lst: list, edge_lst: list, file_name: str = "workflow.json"
):
    """
    Export the graph visualization of nodes and edges as a JSON dictionary.

    Args:
        node_lst (list): List of nodes.
        edge_lst (list): List of edges.
        file_name (str): Name of the file to store the exported graph in.
    """
    pwd_nodes_lst = []
    for n in node_lst:
        if n["type"] == "function":
            pwd_nodes_lst.append(
                {"id": n["id"], "type": n["type"], "value": n["value"]}
            )
        elif n["type"] == "input" and isinstance(n["value"], np.ndarray):
            pwd_nodes_lst.append(
                {
                    "id": n["id"],
                    "type": n["type"],
                    "value": n["value"].tolist(),
                    "name": n["name"],
                }
            )
        else:
            pwd_nodes_lst.append(
                {
                    "id": n["id"],
                    "type": n["type"],
                    "value": n["value"],
                    "name": n["name"],
                }
            )

    final_node = {"id": len(pwd_nodes_lst), "type": "output", "name": "result"}
    pwd_nodes_lst.append(final_node)
    pwd_edges_lst = [
        (
            {
                "target": e["end"],
                "targetPort": e["label"],
                "source": e["start"],
                "sourcePort": None,
            }
            if "start_label" not in e
            else {
                "target": e["end"],
                "targetPort": e["end_label"],
                "source": e["start"],
                "sourcePort": e["start_label"],
            }
        )
        for e in edge_lst
    ]
    pwd_edges_lst.append(
        {
            "target": final_node["id"],
            "targetPort": None,
            "source": max([e["target"] for e in pwd_edges_lst]),
            "sourcePort": None,
        }
    )
    pwd_dict = {
        "version": "0.1.0",
        "nodes": pwd_nodes_lst,
        "edges": pwd_edges_lst,
    }
    with open(file_name, "w") as f:
        json.dump(pwd_dict, f, indent=4)
