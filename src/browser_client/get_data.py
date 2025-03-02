import medmnist
import numpy as np
import json
import os
from tqdm import tqdm

def convert_medmnist_to_json(data_flag: str, output_dir: str):
    """
    Converts MedMNIST dataset to JSON files.

    Parameters:
    - data_flag (str): The dataset flag (e.g., 'bloodmnist').
    - output_dir (str): The directory where JSON files will be saved.

    Outputs:
    - Saves two JSON files: '<data_flag>_train.json' and '<data_flag>_test.json' in the output_dir.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    info = medmnist.INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load training and test datasets
    train_dataset = DataClass(split='train', download=True, as_rgb=True)
    test_dataset = DataClass(split='test', download=True, as_rgb=True)

    def process_and_save(dataset, split):
        samples = []
        print(f"Processing {split} data...")
        
        # Get first sample for verification
        first_image, first_label = dataset[0]
        print(f"First image shape: {np.array(first_image).shape}")  # Should be (28, 28, 3)
        print(f"First label: {first_label}")  # Should be a single integer
        print(f"First image min/max values: {np.array(first_image).min()}, {np.array(first_image).max()}")

        for idx in tqdm(range(len(dataset)), desc=f"Processing {split} samples"):
            image, label = dataset[idx]
            img_array = np.array(image)
            
            # Normalize pixel values to [0, 1] if they're in [0, 255]
            if img_array.max() > 1:
                img_array = img_array / 255.0
                
            flattened = img_array.flatten().tolist()
            
            # Verify dimensions of flattened array
            if idx == 0:
                print(f"Flattened image length: {len(flattened)}")  # Should be 2352
                
            samples.append({
                'image': flattened,
                'label': int(label[0]) if isinstance(label, np.ndarray) else int(label)
            })

        # Save to JSON
        output_file = os.path.join(output_dir, f"{data_flag}_{split}.json")
        with open(output_file, 'w') as f:
            json.dump(samples, f)
        print(f"Saved {split} data to {output_file}")

    # Process and save both train and test datasets
    process_and_save(train_dataset, 'train')
    process_and_save(test_dataset, 'test')

    print("Conversion completed successfully.")

if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Convert MedMNIST dataset to JSON.")
#     parser.add_argument('--data_flag', type=str, required=True, help="Dataset flag (e.g., 'bloodmnist').")
#     parser.add_argument('--output_dir', type=str, default='data', help="Directory to save JSON files.")
    
#     args = parser.parse_args()
    convert_medmnist_to_json("bloodmnist", "./public/datasets/imgs/bloodmnist/")
