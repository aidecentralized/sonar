import io
import torch

def serialize_model(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

def deserialize_model(model_bytes: bytes) -> torch.nn.Module:
    buffer = io.BytesIO(model_bytes)
    buffer.seek(0)
    model_wts = torch.load(buffer)
    return model_wts
