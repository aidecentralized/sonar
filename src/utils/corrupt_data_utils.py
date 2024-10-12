from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from typing import Tuple
from utils.corruptions import corrupt_mapping

# Custom dataset wrapper to apply corruption
class CorruptDataset(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, corruption_fn_name, severity: int = 1):
        print("Initialized CorruptDataset with corruption_fn_name: ", corruption_fn_name)
        self.dataset = dataset  # Original dataset
        self.corruption_fn = corrupt_mapping[corruption_fn_name]  # Corruption function
        self.severity = severity  # Corruption severity

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        data, target = self.dataset[index]  # Get a single image and label
        data_np = np.array(data)  # Convert image to NumPy
        data_np = self.corruption_fn(data_np, severity=self.severity)  # Apply corruption
        data = Image.fromarray(data_np)  # Convert back to PIL Image (optional, depends on your corruption)
        data = transforms.ToTensor()(data)  # Convert to tensor for PyTorch
        return data, target

    def __len__(self) -> int:
        return len(self.dataset)
    

# scratch work
# TODO: remove this
            # if (self.config.get("corrupt_fn", None) is not None) and (
            #     self.config.get("corrupt_severity", None) is not None
            # ):
            #     # Assuming data has shape [batch_size, channels, height, width]
            #     data_np = data.cpu().numpy()  # Move to CPU and convert to numpy for corruption
                
            #     # Apply corruption to each image in the batch
            #     for i in range(data_np.shape[0]):  # Loop over batch dimension
            #         # Example: Apply gaussian_noise corruption
            #         data_np[i] = corrupt(data_np[i], severity=self.config["corrupt_severity"], corruption_name=self.config["corrupt_fn"])
            #         gaussian_noise(data_np[i], severity=1)  # Corruption function

            #     # Convert the numpy data back to a tensor and move to the device
            #     data = torch.from_numpy(data_np).to(device)
            # else: