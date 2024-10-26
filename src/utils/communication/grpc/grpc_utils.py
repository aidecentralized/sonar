from collections import OrderedDict
import io
from typing import Dict, Any
import torch


def serialize_model(state_dict: Dict[str, torch.Tensor]) -> bytes:
    # put every parameter on cpu first
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to("cpu")
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)  # type: ignore
    buffer.seek(0)
    return buffer.read()


def deserialize_model(model_bytes: bytes) -> OrderedDict[str, torch.Tensor]:
    buffer = io.BytesIO(model_bytes)
    buffer.seek(0)
    model_wts = torch.load(buffer)  # type: ignore
    return model_wts

def serialize_message(message: Dict[str, Any]) -> bytes:
    # assumes all tensors are on cpu
    buffer = io.BytesIO()
    torch.save(message, buffer)  # type: ignore
    buffer.seek(0)
    return buffer.read()


def deserialize_message(model_bytes: bytes) -> OrderedDict[str, Any]:
    buffer = io.BytesIO(model_bytes)
    buffer.seek(0)
    message = torch.load(buffer)  # type: ignore
    return message