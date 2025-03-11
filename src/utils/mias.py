import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from typing import List, Dict

"""
LOSS-based mia attack based on 
https://github.com/sacs-epfl/shatter/blob/main/src/virtualNodes/attacks/MIA/LOSSCIFARTestSet.py#L4
"""

class LOSSMIA:
    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )

    def _model_eval(self, model, data_samples, epsilon=10e-9):
        with torch.no_grad():
            model = model.to(self.device)
            data, targets = data_samples
            data = data.to(self.device)
            targets = targets.to(self.device)
            output = model(data)
            loss_val = (
                F.cross_entropy(output, targets, reduction="none").detach().clone()
            )

            nan_mask = torch.isnan(loss_val)
            loss_val[nan_mask] = torch.tensor(1 / epsilon).to(self.device)
            inf_mask = torch.isinf(loss_val)
            loss_val[inf_mask] = torch.tensor(1 / epsilon).to(self.device)

            return loss_val

    def attack_dataset(
        self,
        victim_model,
        in_dataloader,
        out_dataloader,
        in_size=50000,
        out_size=10000,
        epsilon=10e-9,
    ):
        victim_model.eval()
        loss_vals = {
            "in": torch.zeros((in_size,), dtype=torch.float32, device=self.device),
            "out": torch.zeros((out_size,), dtype=torch.float32, device=self.device),
        }
        with torch.no_grad():
            # Process in_dataloader (members)
            last = 0
            for data_samples in in_dataloader:
                loss_in = -self._model_eval(victim_model, data_samples, epsilon=epsilon)
                batch_size = len(data_samples[1])  # Get the actual batch size
                loss_vals["in"][last : last + batch_size] = loss_in[:batch_size]  # Ensure sizes match
                last += batch_size
            loss_vals["in"] = loss_vals["in"][:last].cpu()  # Trim to actual size

            # Process out_dataloader (non-members)
            last = 0
            for data_samples in out_dataloader:
                loss_out = -self._model_eval(victim_model, data_samples, epsilon=epsilon)
                batch_size = len(data_samples[1])  # Get the actual batch size
                loss_vals["out"][last : last + batch_size] = loss_out[:batch_size]  # Ensure sizes match
                last += batch_size
            loss_vals["out"] = loss_vals["out"][:last].cpu()  # Trim to actual size

        return loss_vals

    def calculate_roc_auc_score(self, loss_vals: List[float]) -> Dict[str, float]:
        """
        Calculate ROC-AUC score for the given loss values
        """
        results = {}
        # Combine loss values for members and non-members
        all_losses = torch.cat([loss_vals["in"], loss_vals["out"]])
        # Create labels: 1 for members, 0 for non-members
        labels = torch.cat([torch.ones_like(loss_vals["in"]), torch.zeros_like(loss_vals["out"])])
        # Compute ROC-AUC score
        auc_score = roc_auc_score(labels.numpy(), all_losses.numpy())
        results["loss"] = auc_score
        return results
    
# MIA for baseline from https://github.com/antibloch/mia_attacks/blob/main/baseline/baseline_conf.py 
    
def calculate_accuracy(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total


# Function to train a model
def train_model_with_raw_tensors(model, train_data, train_labels, epochs=100, lr=0.01,bs=128*2, device='cuda'):
    dataset= torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in train_loader:
            img, label = batch
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
    return model


def train_model_with_loader(model, train_loader, training_epochs=100, lr=0.01, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    for epochy in range(training_epochs):
        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data) 
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def get_softmax_scores(model, dataloader, device):
    scores = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            # Apply softmax to get probability distributions
            softmax_outputs = F.softmax(outputs, dim=1)
            # Get the maximum probability for each prediction
            max_probs, _ = torch.max(softmax_outputs, dim=1)
            scores.append(max_probs.cpu())
    return torch.cat(scores).unsqueeze(1)

def shadow_zone(target_model, train_loader, test_loader, device='cuda'):
    """Get confidence scores and/or losses for train and test sets"""
    train_metrics = []
    test_metrics = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    target_model.eval()
    with torch.no_grad():
        # Get metrics for training data
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = target_model(data)
            
            # Get confidence (max probability)
            probs = F.softmax(outputs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            
            # Get loss
            loss = criterion(outputs, labels)
            
            # Get entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            
            # Get probability of correct class
            correct_probs = probs[torch.arange(len(labels)), labels]
            
            metrics = {
                'confidence': confidence.cpu(),
                'loss': loss.cpu(),
                'entropy': entropy.cpu(),
                'correct_prob': correct_probs.cpu()
            }
            train_metrics.append(metrics)
            
        # Get metrics for test data
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = target_model(data)
            
            probs = F.softmax(outputs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            loss = criterion(outputs, labels)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            correct_probs = probs[torch.arange(len(labels)), labels]
            
            metrics = {
                'confidence': confidence.cpu(),
                'loss': loss.cpu(),
                'entropy': entropy.cpu(),
                'correct_prob': correct_probs.cpu()
            }
            test_metrics.append(metrics)
    
    # Combine metrics
    combined_train = {k: torch.cat([m[k] for m in train_metrics]) for k in train_metrics[0].keys()}
    combined_test = {k: torch.cat([m[k] for m in test_metrics]) for k in test_metrics[0].keys()}
    
    return combined_train, combined_test

def MIA(target_model, train_loader, test_loader, device='cuda'):
    """Perform MIA using different metrics"""
    # Get metrics for train and test sets
    target_model.eval()
    train_metrics, test_metrics = shadow_zone(target_model, train_loader, test_loader, device)
    
    results = {}
    log = {}
    for metric_name in train_metrics.keys():
        print(f"DEBUG: {metric_name}")
        # Combine values and create labels
        values = torch.cat([train_metrics[metric_name], test_metrics[metric_name]]).numpy()
        labels = np.concatenate([np.ones(len(train_metrics[metric_name])), 
                               np.zeros(len(test_metrics[metric_name]))])
        labels = np.concatenate([np.ones(len(train_metrics[metric_name])), 
                               np.zeros(len(test_metrics[metric_name]))])
    
        # Check for NaN or Inf values
        nan_mask = np.isnan(values) | np.isinf(values)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN/Inf values in {metric_name}")
            raise
            # Replace NaNs with a large value or median
            # values[nan_mask] = np.nanmedian(values)
            values[nan_mask] = 1e6
        
        # For loss and entropy, lower values indicate membership
        if metric_name in ['loss', 'entropy']:
            auc_score = roc_auc_score(labels, -values)
        else:
            auc_score = roc_auc_score(labels, values)
            
        results[metric_name] = {
            'auc_score': auc_score,
            'train_metrics': train_metrics[metric_name].cpu().numpy().tolist(),
            'test_metrics': test_metrics[metric_name].cpu().numpy().tolist(),
            'train_mean': train_metrics[metric_name].mean().item(),
            'train_std': train_metrics[metric_name].std().item(),
            'test_mean': test_metrics[metric_name].mean().item(),
            'test_std': test_metrics[metric_name].std().item()
        }
        log[metric_name] = auc_score
        
        print(f"\n{metric_name.upper()} Statistics:")
        print(f"Train - mean: {results[metric_name]['train_mean']:.4f}, std: {results[metric_name]['train_std']:.4f}")
        print(f"Test  - mean: {results[metric_name]['test_mean']:.4f}, std: {results[metric_name]['test_std']:.4f}")
        print(f"ROC-AUC: {results[metric_name]['auc_score']:.4f}")
    
    target_model.train()
    return log, results