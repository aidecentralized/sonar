from collections import OrderedDict
import io
import torch

def serialize_model(state_dict: OrderedDict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer) # type: ignore
    buffer.seek(0)
    return buffer.read()

def deserialize_model(model_bytes: bytes) -> OrderedDict[str, torch.Tensor]:
    buffer = io.BytesIO(model_bytes)
    buffer.seek(0)
    model_wts = torch.load(buffer) # type: ignore
    return model_wts
