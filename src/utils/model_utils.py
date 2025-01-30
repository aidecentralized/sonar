from collections import OrderedDict
from typing import Any, Tuple, List, Dict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

import resnet
import resnet_in

import yolo
from utils.types import ConfigType

class ModelUtils:
    def __init__(self, device: torch.device, config: ConfigType) -> None:
        self.device = device
        self.dset = None

        self.models_layers_idx = {
            "resnet10": {"l1": 17, "l2": 35, "l3": 53, "l4": 71, "fc": 73},
            "resnet18": {"l1": 29, "l2": 59, "l3": 89, "l4": 119, "fc": 121},
        }

        self.config = config
        self.malicious_type = config.get("malicious_type", "normal")

    def get_model(
        self,
        model_name: str,
        dset: str,
        device: torch.device,
        pretrained: bool = False,
        **kwargs: Any,
    ):
        self.dset = dset
        # TODO: add support for loading checkpointed models
        model_name = model_name.lower()
        if model_name == "resnet6":
            if pretrained:
                raise ValueError("Pretrained model not available for resnet6")
            model = resnet.resnet6(**kwargs)
        elif model_name == "resnet10":
            if pretrained:
                raise ValueError("Pretrained model not available for resnet10")
            model = resnet.resnet10(**kwargs)
        elif model_name == "resnet18":
            model = (
                resnet_in.resnet18(pretrained=True, **kwargs)
                if pretrained
                else resnet.resnet18(**kwargs)
            )
        elif model_name == "resnet34":
            model = (
                resnet_in.resnet34(pretrained=True, **kwargs)
                if pretrained
                else resnet.resnet34(**kwargs)
            )
        elif model_name == "resnet50":
            model = (
                resnet_in.resnet50(pretrained=True, **kwargs)
                if pretrained
                else resnet.resnet50(**kwargs)
            )
        elif model_name == "yolo":
            model = yolo.yolo(pretrained=False)
        else:
            raise ValueError(f"Model name {model_name} not supported")
        model = model.to(device)
        return model

    def train(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        dloader: DataLoader[Any],
        loss_fn: Any,
        device: torch.device,
        test_loader: DataLoader[Any] | None = None,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """TODO: generate docstring"""
        model.train()

        if self.dset == "pascal":
            mean_loss, acc = self.train_object_detection(
                model, optim, dloader, loss_fn, device, test_loader=None, **kwargs
            )
            return mean_loss, acc
        elif self.malicious_type == "backdoor_attack" or self.malicious_type == "gradient_attack":
            train_loss, acc = self.train_classification_malicious(
                model, optim, dloader, loss_fn, device, test_loader=None, **kwargs
            )
            return train_loss, acc
        else:
            train_loss, acc = self.train_classification(
                model, optim, dloader, loss_fn, device, test_loader=None, **kwargs
            )
            return train_loss, acc

    def train_object_detection(
        self,
        model: nn.Module,
        optim,
        dloader,
        loss_fn,
        device: torch.device,
        test_loader=None,
        **kwargs,
    ) -> Tuple[float, float]:
        losses = []  # Initialize the list to store the losses

        for batch_idx, (data, target) in enumerate(dloader):
            data = data.to(device)
            y0, y1, y2 = (
                target[0].to(device),
                target[1].to(device),
                target[2].to(device),
            )

            image_size = kwargs.get("image_size", 416)
            # Grid cell sizes
            s = [image_size // 32, image_size // 16, image_size // 8]

            # Calculating the loss at each scale
            scaled_anchors = kwargs.get(
                "scaled_anchors",
                (
                    torch.tensor(
                        [
                            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
                        ]
                    )
                    * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
                ).to(device),
            )

            with torch.cuda.amp.autocast(enabled=False):
                # Getting the model predictions
                outputs = model(data)

                loss = (
                    loss_fn(outputs[0], y0, scaled_anchors[0])
                    + loss_fn(outputs[1], y1, scaled_anchors[1])
                    + loss_fn(outputs[2], y2, scaled_anchors[2])
                )

            # Add the loss to the list
            losses.append(loss.item())

            # Reset gradients
            optim.zero_grad()
            scaler = kwargs.get("scaler", torch.cuda.amp.GradScaler())
            # Backpropagate the loss
            scaler.scale(loss).backward()

            # Optimization step
            scaler.step(optim)

            # Update the scaler for next iteration
            scaler.update()

            # Calculate the mean loss dynamically after each batch
            mean_loss = sum(losses) / len(losses)
            print("batch ", batch_idx, "loss ", mean_loss)

        return mean_loss, 0  # Optionally return the mean loss at the end

    def train_classification(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        dloader: DataLoader[Any],
        loss_fn: Any,
        device: torch.device,
        test_loader: DataLoader[Any] | None = None,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        model.train()
        correct = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data = data.to(device)
            target = target.to(device)

            optim.zero_grad()

            position = kwargs.get("position", 0)
            output = model(data, position=position)

            if kwargs.get("apply_softmax", False):
                print("here, applying softmax")
                output = nn.functional.log_softmax(output, dim=1)  # type: ignore
            if kwargs.get("gia", False):
                from inversefed.reconstruction_algorithms import loss_steps
                
                # Sum the loss and create gradient graph like in loss_steps
                # Use modified loss_steps function that returns loss
                model.eval()
                param_updates = loss_steps(
                    model, 
                    data, 
                    target, 
                    loss_fn=loss_fn,
                    lr=3e-4,
                    local_steps=1,
                    use_updates=True,  # Must be True to get parameter differences
                    batch_size=10
                )
                
                # save parameter update for sanity check
                # with open(f"param_updates_{node_id}.pkl", "wb") as f:
                #     pickle.dump(param_updates, f)
                model.train()

                # Apply the updates to the model parameters
                with torch.no_grad():
                    for param, update in zip(model.parameters(), param_updates):
                        param.data.add_(update)  # Directly add the update differences
                
                # Compute loss for tracking (without gradients since we've already updated)
                with torch.no_grad():
                    position = kwargs.get("position", 0)
                    output = model(data, position=position)
                    if kwargs.get("apply_softmax", False):
                        output = nn.functional.log_softmax(output, dim=1)
                    loss = loss_fn(output, target)
                    train_loss += loss.item()
 
            else:
                # Standard training procedure
                optim.zero_grad()
                position = kwargs.get("position", 0)
                output = model(data, position=position)
                
                if kwargs.get("apply_softmax", False):
                    output = nn.functional.log_softmax(output, dim=1)
                
                loss = loss_fn(output, target)
                loss.backward()
                optim.step()
                train_loss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                output = model(data, position=position)
                pred = output.argmax(dim=1, keepdim=True)
                if len(target.size()) > 1:
                    target = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            if test_loader is not None:
                test_loss, test_acc = self.test(model, test_loader, loss_fn, device)
                print(
                    f"Train Loss: {train_loss/(batch_idx+1):.6f} | "
                    f"Train Acc: {correct/((batch_idx+1)*len(data)):.6f} | "
                    f"Test Loss: {test_loss:.6f} | "
                    f"Test Acc: {test_acc:.6f}"
                )

        acc = correct / len(dloader.dataset)
        return train_loss, acc
    
    def train_classification_malicious(
        self,
        model: nn.Module,
        optim,
        dloader,
        loss_fn,
        device: torch.device,
        test_loader=None,
        **kwargs,
    ) -> Tuple[float, float]:
        correct = 0
        train_loss = 0

        for batch_idx, (data, target) in enumerate(dloader):
            data = data.to(device)
            target = target.to(device)

            optim.zero_grad()

            position = kwargs.get("position", 0)
            output = model(data, position=position)

            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1)  # type: ignore

            if self.malicious_type == "backdoor_attack":
                target_labels = self.config.get("target_labels", [])
                additional_loss = self.config.get("additional_loss", 10)
                
                # Modify loss if target labels are part of the current batch
                loss = loss_fn(output, target)  # Initial standard loss computation

                for label in target_labels:
                    # Check if current target is within target labels
                    target_mask = (target == label)
                    if target_mask.any():  # If any of the targets match the backdoor target label
                        # Add malicious behavior: Forcing the model to incorrectly classify specific labels
                        # Manipulate loss by adding a large penalty when the target label is present
                        loss += additional_loss  # You can tune this value to control the severity of the attack

                # Perform backpropagation with modified loss
                loss.backward()
            elif self.malicious_type == "label_flip":
                # permutation = torch.tensor(self.config.get("permutation", [i for i in range(10)]))
                permute_labels = self.config.get("permute_labels", 10)
                permutation = torch.randperm(permute_labels)
                permutation = permutation.to(target.device)

                target = permutation[target] # flipped targets
                loss = loss_fn(output, target)
                loss.backward()

            else:
                loss = loss_fn(output, target)
                loss.backward()

            # scale the gradients if this is a gradient attack:
            if self.malicious_type == "gradient_attack":
                scaling_factor = self.config.get("scaling_factor", None)
                noise_factor = self.config.get("noise_factor", None)

                for param in self.model.parameters():
                    if param.grad is not None:
                        if scaling_factor:
                            param.grad.data *= scaling_factor  # Scaling the gradient maliciously
                        if noise_factor:
                            noise = torch.randn(param.grad.size()).to(self.device) * noise_factor
                            param.grad.data += noise

            optim.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are
            # the same
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if test_loader is not None:
                # TODO: implement test loader for pascal
                test_loss, test_acc = self.test(model, test_loader, loss_fn, device)
                print(
                    f"Train Loss: {train_loss/(batch_idx+1):.6f} | Train Acc: {correct/((batch_idx+1)*len(data)):.6f} | Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.6f}"
                )

        acc = correct / len(dloader.dataset)
        return train_loss, acc

    def train_mask(
        self,
        model: nn.Module,
        mask,
        optim,
        dloader,
        loss_fn,
        device: torch.device,
        **kwargs,
    ) -> Tuple[float, float]:
        """TODO: generate docstring"""
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            position = kwargs.get("position", 0)
            output = model(data, position=position)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1)  # type: ignore
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            for name, param in model.named_parameters():
                if name in mask:
                    param.data *= mask[name].to(device)
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are
            # the same
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(dloader.dataset)
        return train_loss, acc

    def deep_mutual_train(
        self, models, optim, dloader, device: torch.device, **kwargs
    ) -> Tuple[list, list]:
        """TODO: generate docstring"""
        num_net = len(models)
        for model in models:
            model.train()

        optimizers = [optim] * num_net
        criterion_CE = nn.CrossEntropyLoss()
        criterion_KLD = nn.KLDivLoss(reduction="batchmean")
        train_loss = [0] * num_net
        pred = [0] * num_net
        correct = [0] * num_net
        train_accuracy = [0] * num_net

        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            output = []
            losses = []
            KLD_loss = []
            CE_loss = []
            for i in range(num_net):
                output.append(models[i](data))
            for k in range(num_net):
                CE_loss.append(criterion_CE(output[k], target))
                KLD_loss.append(0)
                for l in range(num_net):
                    if l != k:
                        KLD_loss[k] += criterion_KLD(
                            F.log_softmax(output[k], dim=1),
                            F.softmax(output[l], dim=1).detach(),
                        )
                losses.append(CE_loss[k] + KLD_loss[k] / (num_net - 1))
            for i in range(num_net):
                train_loss[i] = losses[i].item()
                pred[i] = output[i].max(1, keepdim=True)[1]
                correct[i] += pred[i].eq(target.view_as(pred[i])).sum().item()
            for i in range(num_net):
                optimizers[i].zero_grad()
                losses[i].backward()
                optimizers[i].step()
        for i in range(num_net):
            train_accuracy[i] = 100.0 * correct[i] / len(dloader.dataset)
        return train_loss, train_accuracy

    def test(
        self,
        model: nn.Module,
        dloader: DataLoader[Any],
        loss_fn: Any,
        device: torch.device,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """TODO: generate docstring"""
        model.eval()
        test_loss, acc = 0, 0
        if self.dset == "pascal":
            test_loss, acc = self.test_object_detect(
                model, dloader, loss_fn, device, **kwargs
            )
        else:
            test_loss, acc = self.test_classification(
                model, dloader, loss_fn, device, **kwargs
            )
        return test_loss, acc

    def test_object_detect(
        self, model, dloader, loss_fn, device, **kwargs
    ) -> Tuple[float, float]:
        losses = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(dloader):
                data = data.to(device)
                y0, y1, y2 = (
                    target[0].to(device),
                    target[1].to(device),
                    target[2].to(device),
                )

                image_size = kwargs.get("image_size", 416)
                # Grid cell sizes
                s = [image_size // 32, image_size // 16, image_size // 8]
                # Calculating the loss at each scale
                scaled_anchors = kwargs.get(
                    "scaled_achors",
                    (
                        torch.tensor(
                            [
                                [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                                [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                                [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
                            ]
                        )
                        * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
                    ).to(device),
                )

                with torch.cuda.amp.autocast(enabled=False):
                    # Getting the model predictions
                    outputs = model(data)

                    loss = (
                        loss_fn(outputs[0], y0, scaled_anchors[0])
                        + loss_fn(outputs[1], y1, scaled_anchors[1])
                        + loss_fn(outputs[2], y2, scaled_anchors[2])
                    )

                # Add the loss to the list
                losses.append(loss.item())

                # train loss will be average loss
            loss = sum(losses) / len(losses)
        return loss, 0

    def test_classification(
        self, model, dloader, loss_fn, device, **kwargs
    ) -> Tuple[float, float]:
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for idx, (data, target) in enumerate(dloader):
                data, target = data.to(device), target.to(device)
                position = kwargs.get("position", 0)
                output = model(data, position=position)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                # view_as() is used to make sure the shape of pred and target
                # are the same
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(dloader.dataset)
        return test_loss, acc

    def save_model(self, model: nn.Module, path: str) -> None:
        if isinstance(model, DataParallel):
            model_ = model.module
        else:
            model_ = model
        torch.save(model_.state_dict(), path)

    def move_to_device(
        self, items: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device
    ) -> list:
        # Expects a list of tuples with each tupe containing two tensors
        return [[item[0].to(device), item[1].to(device)] for item in items]

    def substract_model_weights(self, model1: OrderedDict[str, Tensor], model2: OrderedDict[str, Tensor]):
        res: OrderedDict[str, Tensor] = {}
        for key, param in model1.items():
            res[key] = param - model2[key]
        return res

    def get_last_layer_keys(self, model_wts: OrderedDict[str, Tensor]) -> List[str]:
        # Assume one layer is composed of multiple weights named as "layer_name.weight_name"

        reversed_model_wts = reversed(model_wts)
        last_key = next(reversed_model_wts)
        last_layer = last_key.split(".")[0]

        last_layer_keys: List[str] = []

        while last_key is not None and last_key.startswith(last_layer + "."):
            last_layer_keys.append(last_key)
            last_key = next(reversed_model_wts, None)
        return last_layer_keys

    def filter_model_weights(
        self, model_wts: Dict[str, Tensor], key_to_ignore: List[str]
    ) -> Dict[str, Tensor]:
        # Assume one layer is composed of multiple weights named as "layer_name.weight_name"

        filtered_model_wts: Dict[str, Tensor] = {}
        for key, param in model_wts.items():
            if key not in key_to_ignore:
                filtered_model_wts[key] = param
        return filtered_model_wts

    def get_memory_usage(self):
        """
        Get the memory usage
        """
        return torch.cuda.memory_allocated(self.device)