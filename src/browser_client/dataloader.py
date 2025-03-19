import medmnist
import torchvision
import torchvision.transforms as transforms
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


def convert_cifar10_to_json(output_dir: str):
    """
    Converts CIFAR-10 dataset to JSON files with flattened 32x32x3 image arrays.

    Parameters:
    - output_dir (str): Directory to save JSON files.

    Outputs:
    - Saves 'cifar10_train.json' and 'cifar10_test.json'.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Define transformation to normalize the images and convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor()  # Converts to [0, 1] float tensors
        ])

        # Download CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(root='./rawdata', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./rawdata', train=False, download=True, transform=transform)

        def process_and_save(dataset, split):
            samples = []
            print(f"Processing {split} data... Total samples: {len(dataset)}")

            for idx in tqdm(range(len(dataset)), desc=f"Processing {split} samples"):
                image, label = dataset[idx]
                
                # Convert tensor to numpy array
                img_array = image.permute(1, 2, 0).numpy()  # Shape [32, 32, 3]
                flattened = img_array.flatten().tolist()  # Flatten to [3072]

                # Append to samples list
                samples.append({
                    'image': flattened,
                    'label': int(label)
                })

                # Verification on the first sample
                if idx == 0:
                    print(f"Original shape: {img_array.shape}, Flattened length: {len(flattened)}")
                    print(f"Label: {label}, Min/Max Pixel Values: {np.min(img_array)}, {np.max(img_array)}")

            # Save JSON file
            output_file = os.path.join(output_dir, f"cifar10_{split}.json")
            with open(output_file, 'w') as f:
                json.dump(samples, f)
            print(f"Saved {split} data to {output_file}")

        # Process train and test splits
        process_and_save(train_dataset, 'train')
        process_and_save(test_dataset, 'test')

        print("Conversion completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


def partition_cifar10_to_json(output_dir: str, num_clients: int, iid: bool = True, non_iid_strategy: str = "label_skew", 
                            classes_per_client: int = 2, alpha: float = 0.5):
    """
    Partitions CIFAR-10 dataset into multiple client datasets with options for IID and non-IID distributions.

    Parameters:
    - output_dir (str): Directory to save the partitioned JSON files.
    - num_clients (int): Number of clients to partition the data for.
    - iid (bool): If True, data is distributed randomly (IID). If False, uses non-IID strategy.
    - non_iid_strategy (str): Strategy for non-IID partitioning. Options:
        - "label_skew": Each client gets samples from a subset of classes.
        - "dirichlet": Uses Dirichlet distribution to create unbalanced class distributions.
    - classes_per_client (int): Number of classes per client when using "label_skew" (2-10).
    - alpha (float): Concentration parameter for Dirichlet distribution (smaller alpha = more skew).

    Outputs:
    - Saves 'cifar10_client_{i}_train.json' and 'cifar10_client_{i}_test.json' for each client i.
    """
    try:
        import numpy as np
        
        os.makedirs(output_dir, exist_ok=True)

        # Define transformation to normalize the images and convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor()  # Converts to [0, 1] float tensors
        ])

        # Download CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(root='./rawdata', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./rawdata', train=False, download=True, transform=transform)

        # Get class labels for all samples
        all_train_labels = np.array([label for _, label in train_dataset])
        all_test_labels = np.array([label for _, label in test_dataset])
        
        # Get indices for each class
        train_indices_per_class = [np.where(all_train_labels == c)[0] for c in range(10)]
        test_indices_per_class = [np.where(all_test_labels == c)[0] for c in range(10)]
        
        # Initialize partition assignments for each sample
        num_train_samples = len(train_dataset)
        num_test_samples = len(test_dataset)
        train_partition_indices = [[] for _ in range(num_clients)]
        test_partition_indices = [[] for _ in range(num_clients)]
        
        # Partition the data based on the specified distribution type
        if iid:
            # IID: Random partitioning
            train_indices = np.random.permutation(num_train_samples)
            test_indices = np.random.permutation(num_test_samples)
            
            # Calculate samples per client (make it as balanced as possible)
            train_samples_per_client = num_train_samples // num_clients
            test_samples_per_client = num_test_samples // num_clients
            
            # Distribute indices to clients
            for i in range(num_clients):
                start_train = i * train_samples_per_client
                end_train = (i + 1) * train_samples_per_client if i < num_clients - 1 else num_train_samples
                train_partition_indices[i] = train_indices[start_train:end_train].tolist()
                
                start_test = i * test_samples_per_client
                end_test = (i + 1) * test_samples_per_client if i < num_clients - 1 else num_test_samples
                test_partition_indices[i] = test_indices[start_test:end_test].tolist()
        
        elif non_iid_strategy == "label_skew":
            # Label-skewed non-IID: Each client gets a subset of classes
            classes_per_client = min(max(1, classes_per_client), 10)  # Ensure between 1 and 10 classes
            
            # Assign classes to clients
            class_assignments = []
            for i in range(num_clients):
                # Determine classes for this client (with potential overlap)
                client_classes = np.random.choice(10, classes_per_client, replace=False)
                class_assignments.append(client_classes)
                
                # Collect indices for assigned classes
                client_train_indices = []
                client_test_indices = []
                for c in client_classes:
                    client_train_indices.extend(train_indices_per_class[c])
                    client_test_indices.extend(test_indices_per_class[c])
                
                # Shuffle indices
                np.random.shuffle(client_train_indices)
                np.random.shuffle(client_test_indices)
                
                # Store indices
                train_partition_indices[i] = client_train_indices
                test_partition_indices[i] = client_test_indices
                
            # Print class distribution
            print("Non-IID (Label Skew) Class Distribution:")
            for i, classes in enumerate(class_assignments):
                print(f"Client {i}: Classes {classes}")
        
        elif non_iid_strategy == "dirichlet":
            # Dirichlet-based non-IID: Unbalanced class distribution
            # Each client gets samples from all classes but with different proportions
            train_proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha, size=10)
            test_proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha, size=10)
            
            # For each class, distribute samples according to the proportions
            for c in range(10):
                train_class_indices = train_indices_per_class[c]
                test_class_indices = test_indices_per_class[c]
                
                # Calculate how many samples of this class go to each client
                train_counts = (train_proportions[c] * len(train_class_indices)).astype(int)
                test_counts = (test_proportions[c] * len(test_class_indices)).astype(int)
                
                # Make sure we assign all samples
                train_counts[-1] = len(train_class_indices) - np.sum(train_counts[:-1])
                test_counts[-1] = len(test_class_indices) - np.sum(test_counts[:-1])
                
                # Shuffle indices
                np.random.shuffle(train_class_indices)
                np.random.shuffle(test_class_indices)
                
                # Distribute indices
                train_idx = 0
                test_idx = 0
                for i in range(num_clients):
                    if train_counts[i] > 0:
                        train_partition_indices[i].extend(
                            train_class_indices[train_idx:train_idx + train_counts[i]].tolist()
                        )
                        train_idx += train_counts[i]
                    
                    if test_counts[i] > 0:
                        test_partition_indices[i].extend(
                            test_class_indices[test_idx:test_idx + test_counts[i]].tolist()
                        )
                        test_idx += test_counts[i]
            
            # Print sample counts per client
            print("Non-IID (Dirichlet) Distribution:")
            for i in range(num_clients):
                train_class_counts = [np.sum(all_train_labels[train_partition_indices[i]] == c) for c in range(10)]
                print(f"Client {i} train samples: {len(train_partition_indices[i])}, " 
                      f"Class distribution: {train_class_counts}")
        
        else:
            raise ValueError(f"Unknown non-IID strategy: {non_iid_strategy}")
        
        # Process and save partitioned data for each client
        for client_idx in range(num_clients):
            # Process training data
            train_client_samples = []
            train_indices = train_partition_indices[client_idx]
            print(f"Processing client {client_idx} train data... Total samples: {len(train_indices)}")
            
            for idx in tqdm(train_indices, desc=f"Processing client {client_idx} train samples"):
                image, label = train_dataset[idx]
                
                # Convert tensor to numpy array
                img_array = image.permute(1, 2, 0).numpy()  # Shape [32, 32, 3]
                flattened = img_array.flatten().tolist()  # Flatten to [3072]
                
                # Append to samples list
                train_client_samples.append({
                    'image': flattened,
                    'label': int(label)
                })
            
            # Save training JSON file
            train_output_file = os.path.join(output_dir, f"cifar10_client_{client_idx}_train.json")
            with open(train_output_file, 'w') as f:
                json.dump(train_client_samples, f)
            print(f"Saved client {client_idx} train data to {train_output_file}")
            
            # Process test data
            test_client_samples = []
            test_indices = test_partition_indices[client_idx]
            print(f"Processing client {client_idx} test data... Total samples: {len(test_indices)}")
            
            for idx in tqdm(test_indices, desc=f"Processing client {client_idx} test samples"):
                image, label = test_dataset[idx]
                
                # Convert tensor to numpy array
                img_array = image.permute(1, 2, 0).numpy()  # Shape [32, 32, 3]
                flattened = img_array.flatten().tolist()  # Flatten to [3072]
                
                # Append to samples list
                test_client_samples.append({
                    'image': flattened,
                    'label': int(label)
                })
            
            # Save test JSON file
            test_output_file = os.path.join(output_dir, f"cifar10_client_{client_idx}_test.json")
            with open(test_output_file, 'w') as f:
                json.dump(test_client_samples, f)
            print(f"Saved client {client_idx} test data to {test_output_file}")
        
        print(f"Partitioning completed successfully. Created datasets for {num_clients} clients.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Convert MedMNIST dataset to JSON.")
#     parser.add_argument('--data_flag', type=str, required=True, help="Dataset flag (e.g., 'bloodmnist').")
#     parser.add_argument('--output_dir', type=str, default='data', help="Directory to save JSON files.")
    
#     args = parser.parse_args()
    # convert_medmnist_to_json("bloodmnist", "./public/datasets/imgs/bloodmnist/")
    # convert_cifar10_to_json("./public/datasets/imgs/cifar10/")
    
    # Example usage for IID partitioning
    partition_cifar10_to_json("./public/datasets/imgs/cifar10_iid/", num_clients=20, iid=True)
    
    # Example usage for non-IID label-skewed partitioning
    # partition_cifar10_to_json("./public/datasets/imgs/cifar10_non_iid/", 
    #                           num_clients=10, iid=False, non_iid_strategy="label_skew", classes_per_client=3)
    
    # Example usage for non-IID Dirichlet partitioning
    # partition_cifar10_to_json("./public/datasets/imgs/cifar10_dirichlet/", 
    #                           num_clients=5, iid=False, non_iid_strategy="dirichlet", alpha=0.5)
