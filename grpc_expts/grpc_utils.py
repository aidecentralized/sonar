from collections import OrderedDict
import io
import torch

def serialize_model(state_dict: OrderedDict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return buffer.read()

def deserialize_model(model_bytes: bytes) -> OrderedDict:
    buffer = io.BytesIO(model_bytes)
    buffer.seek(0)
    model_wts = torch.load(buffer)
    return model_wts
