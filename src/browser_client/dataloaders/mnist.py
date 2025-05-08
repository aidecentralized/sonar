import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
from tqdm import tqdm

def partition_mnist_dirichlet_fixed(output_dir: str, num_clients: int = 10, alpha: float = 0.5, test_size_per_client: int = 200):
    """
    Partitions mnist into non-IID training partitions using Dirichlet distribution.
    - Each client gets exactly **5,000** training samples.
    - Each client gets exactly **200** IID test samples.

    Parameters:
    - output_dir (str): Directory to save partitioned JSON files.
    - num_clients (int): Number of clients (default: 10).
    - alpha (float): Dirichlet concentration parameter (smaller = more skewed).
    - test_size_per_client (int): Number of IID test samples per client (default: 200).

    Outputs:
    - Saves 'mnist_client_{i}_train.json' for each client (5,000 images each).
    - Saves 'mnist_client_{i}_test.json' for each client (200 IID test images each).
    """

    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.MNIST(root='./rawdata', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./rawdata', train=False, download=True, transform=transform)

    # Get labels
    train_labels = np.array([label for _, label in train_dataset])
    test_labels = np.array([label for _, label in test_dataset])

    # Get indices per class
    train_indices_per_class = [np.where(train_labels == c)[0] for c in range(10)]
    test_indices_per_class = [np.where(test_labels == c)[0] for c in range(10)]

    # **Step 1: Create Dirichlet Distribution for Training Data**
    train_proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha, size=10)

    train_partition_indices = {i: [] for i in range(num_clients)}

    for c in range(10):  # For each class
        indices = train_indices_per_class[c]
        np.random.shuffle(indices)  # Shuffle for randomness
        proportions = (train_proportions[c] * 50000).astype(int)  # Distribute across 50,000 samples

        # Ensure each client gets **exactly** 5,000 images
        proportions = (proportions / proportions.sum() * 5000).astype(int)

        # Adjust last client to ensure exact sum of 5000
        proportions[-1] += 5000 - np.sum(proportions)

        idx = 0
        for client in range(num_clients):
            train_partition_indices[client].extend(indices[idx:idx + proportions[client]])
            idx += proportions[client]

    # **Step 2: Create IID Test Set (200 samples per client)**
    test_partition_indices = {i: [] for i in range(num_clients)}

    iid_test_indices = []
    test_samples_per_class = test_size_per_client // 10  # Distribute evenly across 10 classes

    for c in range(10):
        selected_indices = np.random.choice(test_indices_per_class[c], test_samples_per_class, replace=False)
        iid_test_indices.extend(selected_indices)

    np.random.shuffle(iid_test_indices)  # Shuffle test set

    for i in range(num_clients):
        test_partition_indices[i] = iid_test_indices

    # **Step 3: Save Partitions to JSON**
    def save_partition(client_idx, train_indices, test_indices):
        train_samples = [{"image": train_dataset[i][0].permute(1, 2, 0).numpy().flatten().tolist(), "label": int(train_dataset[i][1])}
                         for i in tqdm(train_indices, desc=f"Processing client {client_idx} train samples")]
        
        test_samples = [{"image": test_dataset[i][0].permute(1, 2, 0).numpy().flatten().tolist(), "label": int(test_dataset[i][1])}
                        for i in tqdm(test_indices, desc=f"Processing client {client_idx} IID test samples")]

        with open(os.path.join(output_dir, f"mnist_client_{client_idx}_train.json"), 'w') as f:
            json.dump(train_samples, f)

        with open(os.path.join(output_dir, f"mnist_client_{client_idx}_test.json"), 'w') as f:
            json.dump(test_samples, f)

        print(f"Saved client {client_idx} train and test data.")

    # Process each client
    for client_idx in range(num_clients):
        save_partition(client_idx, train_partition_indices[client_idx], test_partition_indices[client_idx])

    print(f"✅ Partitioning completed: {num_clients} clients with 5,000 Dirichlet-based training images and 200 IID test images each.")

def convert_mnist_to_json(output_dir: str):
    """
    Converts MNIST dataset to JSON files with flattened 28x28x1 image arrays.

    Parameters:
    - output_dir (str): Directory to save JSON files.

    Outputs:
    - Saves 'mnist_train.json' and 'mnist_test.json'.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Define transformation to normalize the images and convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor()  # Converts to [0, 1] float tensors
        ])

        # Download mnist dataset
        train_dataset = torchvision.datasets.MNIST(root='./rawdata', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./rawdata', train=False, download=True, transform=transform)

        def process_and_save(dataset, split):
            samples = []
            print(f"Processing {split} data... Total samples: {len(dataset)}")

            for idx in tqdm(range(len(dataset)), desc=f"Processing {split} samples"):
                image, label = dataset[idx]
                
                # Convert tensor to numpy array
                img_array = image.permute(1, 2, 0).numpy()  # Shape [28, 28, 1]
                flattened = img_array.flatten().tolist()  # Flatten to [784]

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
            output_file = os.path.join(output_dir, f"mnist_{split}.json")
            with open(output_file, 'w') as f:
                json.dump(samples, f)
            print(f"Saved {split} data to {output_file}")

        # Process train and test splits
        process_and_save(train_dataset, 'train')
        process_and_save(test_dataset, 'test')

        print("Conversion completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

def partition_mnist_to_json(output_dir: str, num_clients: int, iid: bool = True, non_iid_strategy: str = "label_skew", 
                              classes_per_client: int = 2, alpha: float = 1.0, test_iid: bool = True, generate_both_tests: bool = True):
    """
    Partitions MNIST dataset into multiple client datasets with options for IID and non-IID distributions.
    
    - Training data is partitioned based on IID or non-IID strategies.
    - Test data can be IID (random split) or non-IID (same partitioning as training).
    - Optionally, both test set types can be generated.

    Parameters:
    - output_dir (str): Directory to save the partitioned JSON files.
    - num_clients (int): Number of clients.
    - iid (bool): If True, training data is IID; otherwise, it's non-IID.
    - non_iid_strategy (str): Strategy for non-IID partitioning: "label_skew" or "dirichlet".
    - classes_per_client (int): Number of classes per client in label-skewed partitioning.
    - alpha (float): Dirichlet concentration parameter (lower = more skewed).
    - test_iid (bool): If True, test sets are IID; if False, they are non-IID.
    - generate_both_tests (bool): If True, saves both IID and non-IID test sets.
    """

    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.MNIST(root='./rawdata', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./rawdata', train=False, download=True, transform=transform)

    all_train_labels = np.array([label for _, label in train_dataset])
    all_test_labels = np.array([label for _, label in test_dataset])

    train_indices_per_class = [np.where(all_train_labels == c)[0] for c in range(10)]
    test_indices_per_class = [np.where(all_test_labels == c)[0] for c in range(10)]

    # **1. Partition Training Data (IID or non-IID)**
    train_partition_indices = [[] for _ in range(num_clients)]

    if iid:
        shuffled_indices = np.random.permutation(len(train_dataset))
        train_partition_indices = np.array_split(shuffled_indices, num_clients)
    
    elif non_iid_strategy == "label_skew":
        for client in range(num_clients):
            client_classes = np.random.choice(10, classes_per_client, replace=False)
            client_train_indices = []
            for c in client_classes:
                client_train_indices.extend(train_indices_per_class[c])
            np.random.shuffle(client_train_indices)
            train_partition_indices[client] = client_train_indices
    
    elif non_iid_strategy == "dirichlet":
        train_proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha, size=10)
        for c in range(10):
            train_class_indices = train_indices_per_class[c]
            train_counts = (train_proportions[c] * len(train_class_indices)).astype(int)
            train_counts[-1] += len(train_class_indices) - np.sum(train_counts)  # Ensure full coverage
            np.random.shuffle(train_class_indices)
            idx = 0
            for client in range(num_clients):
                train_partition_indices[client].extend(train_class_indices[idx:idx + train_counts[client]])
                idx += train_counts[client]

    # **2. Partition Test Data**
    test_partition_iid = [[] for _ in range(num_clients)]
    test_partition_non_iid = [[] for _ in range(num_clients)]

    if generate_both_tests or test_iid:
        shuffled_test_indices = np.random.permutation(len(test_dataset))
        test_partition_iid = np.array_split(shuffled_test_indices, num_clients)

    if generate_both_tests or not test_iid:
        if non_iid_strategy == "label_skew":
            for client in range(num_clients):
                client_classes = np.random.choice(10, classes_per_client, replace=False)
                client_test_indices = []
                for c in client_classes:
                    client_test_indices.extend(test_indices_per_class[c])
                np.random.shuffle(client_test_indices)
                test_partition_non_iid[client] = client_test_indices
        elif non_iid_strategy == "dirichlet":
            test_proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha, size=10)
            for c in range(10):
                test_class_indices = test_indices_per_class[c]
                test_counts = (test_proportions[c] * len(test_class_indices)).astype(int)
                test_counts[-1] += len(test_class_indices) - np.sum(test_counts)
                np.random.shuffle(test_class_indices)
                idx = 0
                for client in range(num_clients):
                    test_partition_non_iid[client].extend(test_class_indices[idx:idx + test_counts[client]])
                    idx += test_counts[client]

    # **3. Save Partitions to JSON**
    def save_partition(client_idx, train_indices, test_indices_iid=None, test_indices_non_iid=None):
        train_samples = [{"image": train_dataset[i][0].permute(1, 2, 0).numpy().flatten().tolist(), "label": int(train_dataset[i][1])}
                        for i in tqdm(train_indices, desc=f"Processing client {client_idx} train samples")]
        
        test_samples_iid = []
        test_samples_non_iid = []

        # ✅ FIX: Check length instead of `.size` to avoid AttributeError
        if test_indices_iid is not None and len(test_indices_iid) > 0:
            test_samples_iid = [{"image": test_dataset[i][0].permute(1, 2, 0).numpy().flatten().tolist(), "label": int(test_dataset[i][1])}
                                for i in tqdm(test_indices_iid, desc=f"Processing client {client_idx} IID test samples")]

        if test_indices_non_iid is not None and len(test_indices_non_iid) > 0:
            test_samples_non_iid = [{"image": test_dataset[i][0].permute(1, 2, 0).numpy().flatten().tolist(), "label": int(test_dataset[i][1])}
                                    for i in tqdm(test_indices_non_iid, desc=f"Processing client {client_idx} Non-IID test samples")]

        with open(os.path.join(output_dir, f"mnist_client_{client_idx}_train.json"), 'w') as f:
            json.dump(train_samples, f)

        if test_samples_iid:
            with open(os.path.join(output_dir, f"mnist_client_{client_idx}_test_iid.json"), 'w') as f:
                json.dump(test_samples_iid, f)

        if test_samples_non_iid:
            with open(os.path.join(output_dir, f"mnist_client_{client_idx}_test_noniid.json"), 'w') as f:
                json.dump(test_samples_non_iid, f)

        print(f"Saved client {client_idx} train and test data.")


    # Process each client
    for client_idx in range(num_clients):
        save_partition(client_idx, train_partition_indices[client_idx],
                       test_partition_iid[client_idx] if generate_both_tests or test_iid else None,
                       test_partition_non_iid[client_idx] if generate_both_tests or not test_iid else None)

    print(f"Partitioning completed: {num_clients} clients with non-IID training and both IID & non-IID test sets.")

def partition_mnist_unique_labels(output_dir: str, num_clients: int = 10, test_size_per_client: int = 100):

    """
    Partitions mnist into exactly 10 non-IID training partitions where each client gets one unique class.
    The test set is IID and shared across all clients.

    Parameters:
    - output_dir (str): Directory to save partitioned JSON files.
    - num_clients (int): Number of clients (default: 10 for mnist classes).
    - test_size_per_client (int): Number of IID test samples per client.

    Outputs:
    - Saves 'mnist_client_{i}_train.json' for each client (one unique class).
    - Saves 'mnist_client_{i}_test.json' for each client (IID test set).
    """

    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.MNIST(root='./rawdata', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./rawdata', train=False, download=True, transform=transform)

    # Get labels
    train_labels = np.array([label for _, label in train_dataset])
    test_labels = np.array([label for _, label in test_dataset])

    # Get indices per class
    train_indices_per_class = [np.where(train_labels == c)[0] for c in range(num_clients)]
    test_indices_per_class = [np.where(test_labels == c)[0] for c in range(num_clients)]

    # **Step 1: Assign Each Client One Unique Class for Training**
    train_partition_indices = {i: train_indices_per_class[i].tolist() for i in range(num_clients)}

    # **Step 2: Create a Small IID Test Set**
    test_partition_indices = {i: [] for i in range(num_clients)}

    # Collect a small subset from the test set (equal distribution from all classes)
    test_samples_per_class = test_size_per_client // num_clients  # Evenly distribute test samples

    iid_test_indices = []
    for c in range(num_clients):
        selected_indices = np.random.choice(test_indices_per_class[c], test_samples_per_class, replace=False)
        iid_test_indices.extend(selected_indices)

    np.random.shuffle(iid_test_indices)  # Shuffle test set

    # Assign the same IID test set to all clients
    for i in range(num_clients):
        test_partition_indices[i] = iid_test_indices

    # **Step 3: Save Partitions to JSON**
    def save_partition(client_idx, train_indices, test_indices):
        train_samples = [{"image": train_dataset[i][0].permute(1, 2, 0).numpy().flatten().tolist(), "label": int(train_dataset[i][1])}
                         for i in tqdm(train_indices, desc=f"Processing client {client_idx} train samples")]
        
        test_samples = [{"image": test_dataset[i][0].permute(1, 2, 0).numpy().flatten().tolist(), "label": int(test_dataset[i][1])}
                        for i in tqdm(test_indices, desc=f"Processing client {client_idx} IID test samples")]

        with open(os.path.join(output_dir, f"mnist_client_{client_idx}_train.json"), 'w') as f:
            json.dump(train_samples, f)

        with open(os.path.join(output_dir, f"mnist_client_{client_idx}_test.json"), 'w') as f:
            json.dump(test_samples, f)

        print(f"Saved client {client_idx} train and test data.")

    # Process each client
    for client_idx in range(num_clients):
        save_partition(client_idx, train_partition_indices[client_idx], test_partition_indices[client_idx])

    print(f"✅ Partitioning completed: {num_clients} clients with unique-label training and IID test sets.")

if __name__ == "__main__":
    # convert_mnist_to_json("../public/datasets/imgs/mnist/")
    
    # Example usage for IID partitioning
    # partition_mnist_to_json("../public/datasets/imgs/mnist_iid_split10/", num_clients=10, iid=True)
    
    # Example usage for non-IID label-skewed partitioning
    # partition_mnist_to_json("../public/datasets/imgs/mnist_non_iid/", 
    #                           num_clients=10, iid=False, non_iid_strategy="label_skew", classes_per_client=3)
    
    # Example usage for non-IID Dirichlet partitioning
    # partition_mnist_to_json("../public/datasets/imgs/mnist_dirichlet/", 
    #                           num_clients=5, iid=False, non_iid_strategy="dirichlet", alpha=0.5)

    # BOTH iid and non-iid test sets:
    # partition_mnist_to_json(output_dir="../public/datasets/imgs/mnist_non_iid_10clients_2classes/", num_clients=10, iid=False, 
    #                       non_iid_strategy='label_skew', classes_per_client=2, generate_both_tests=True)

    # partition_mnist_unique_labels(output_dir="../public/datasets/imgs/mnist_non_iid_unique_labels/", num_clients=10, test_size_per_client=200)

    # partition_mnist_dirichlet_fixed(output_dir="../public/datasets/imgs/mnist_non_iid_dirichlet/", num_clients=10, alpha=0.5, test_size_per_client=200)

        # Example usage for IID partitioning
    partition_mnist_to_json("../public/datasets/imgs/mnist/mnist_iid_split10/", num_clients=10, iid=True, generate_both_tests=False)
    
    # Example usage for non-IID label-skewed partitioning
    partition_mnist_to_json("../public/datasets/imgs/mnist/mnist_label_skew_3/", 
                              num_clients=10, iid=False, non_iid_strategy="label_skew", classes_per_client=3, generate_both_tests=True)
    
    # Example usage for non-IID Dirichlet partitioning
    partition_mnist_to_json("../public/datasets/imgs/mnist/mnist_dirichlet_0.5/", 
                              num_clients=10, iid=False, non_iid_strategy="dirichlet", alpha=0.5, generate_both_tests=True)
