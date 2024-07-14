from collections import OrderedDict
import io
import torch


def serialize_model(state_dict: OrderedDict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return buffer.read()


def deserialize_model(
    model_bytes: bytes, device: torch.device = torch.device("cpu")
) -> OrderedDict:
    buffer = io.BytesIO(model_bytes)
    buffer.seek(0)
    model_wts = torch.load(
        buffer, map_location=device
    )  # Specify map_location to load tensors to the desired device
    return model_wts
