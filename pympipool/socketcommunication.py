import cloudpickle


def send_dict(socket, input_dict):
    socket.send(cloudpickle.dumps(input_dict))


def receive_dict(socket):
    output = cloudpickle.loads(socket.recv())
    if "r" in output.keys():
        return output["r"]
    else:
        error_type = output["et"].split("'")[1]
        raise eval(error_type)(output["e"])


def send_and_receive_dict(socket, input_dict):
    send_dict(socket=socket, input_dict=input_dict)
    return receive_dict(socket=socket)
