from concurrent.futures import Future

import cloudpickle


def generate_nodes_and_edges(task_hash_dict, future_hash_inverse_dict):
    node_lst, edge_lst = [], []
    hash_id_dict = {}
    for k, v in task_hash_dict.items():
        hash_id_dict[k] = len(node_lst)
        node_lst.append({"name": v["fn"].__name__, "id": hash_id_dict[k]})
    for k, task_dict in task_hash_dict.items():
        for arg in task_dict["args"]:
            if not isinstance(arg, Future):
                node_id = len(node_lst)
                node_lst.append({"name": str(arg), "id": node_id})
                edge_lst.append({"start": node_id, "end": hash_id_dict[k], "label": ""})
            else:
                edge_lst.append(
                    {
                        "start": hash_id_dict[future_hash_inverse_dict[arg]],
                        "end": hash_id_dict[k],
                        "label": "",
                    }
                )
        for kw, v in task_dict["kwargs"].items():
            if not isinstance(v, Future):
                node_id = len(node_lst)
                node_lst.append({"name": str(v), "id": node_id})
                edge_lst.append(
                    {"start": node_id, "end": hash_id_dict[k], "label": str(kw)}
                )
            else:
                edge_lst.append(
                    {
                        "start": hash_id_dict[future_hash_inverse_dict[v]],
                        "end": hash_id_dict[k],
                        "label": str(kw),
                    }
                )
    return node_lst, edge_lst


def generate_task_hash(task_dict, future_hash_inverse_dict):
    args_for_hash = [
        arg if not isinstance(arg, Future) else future_hash_inverse_dict[arg]
        for arg in task_dict["args"]
    ]
    kwargs_for_hash = {
        k: v if not isinstance(v, Future) else future_hash_inverse_dict[v]
        for k, v in task_dict["kwargs"].items()
    }
    return cloudpickle.dumps(
        {"fn": task_dict["fn"], "args": args_for_hash, "kwargs": kwargs_for_hash}
    )


def draw(node_lst, edge_lst):
    from IPython.display import SVG, display  # noqa
    import matplotlib.pyplot as plt  # noqa
    import networkx as nx  # noqa
    graph = nx.DiGraph()
    for node in node_lst:
        graph.add_node(node["id"], label=node["name"])
    for edge in edge_lst:
        graph.add_edge(edge["start"], edge["end"], label=edge["label"])
    svg = nx.nx_agraph.to_agraph(graph).draw(prog="dot", format="svg")
    display(SVG(svg))
    plt.show()
