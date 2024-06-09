from collections import OrderedDict
from typing import Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

import resnet
import resnet_in


class ModelUtils():
    def __init__(self) -> None:
        pass
    
        self.models_layers_idx = {
            "resnet10":{
                "l1":17,
                "l2":35,
                "l3":53,
                "l4":71,
                "fc":73
            },
            "resnet18":{
                "l1":29,
                "l2":59,
                "l3":89,
                "l4":119,
                "fc":121
            }
        }

    def get_model(self, model_name:str, dset:str, device:torch.device, device_ids:list, pretrained=False, **kwargs) -> DataParallel:
        #TODO: add support for loading checkpointed models
        model_name = model_name.lower()
        if model_name == "resnet10":
            if pretrained:
                raise ValueError("Pretrained model not available for resnet10")
            model = resnet.ResNet10(**kwargs)
        elif model_name == "resnet18":
            model = resnet_in.resnet18(pretrained=True, **kwargs) if pretrained else resnet.ResNet18(**kwargs)
        elif model_name == "resnet34":
            model = resnet_in.resnet34(pretrained=True, **kwargs) if pretrained else resnet.ResNet34(**kwargs)
        elif model_name == "resnet50":
            model = resnet_in.resnet50(pretrained=True, **kwargs) if pretrained else resnet.ResNet50(**kwargs)
        else:
            raise ValueError(f"Model name {model_name} not supported")
        model = model.to(device)
        return model

    def train(self, model:nn.Module, optim, dloader, loss_fn, device: torch.device, test_loader=None, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            position = kwargs.get("position", 0)
            output = model(data, position=position)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1) # type: ignore
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are the same
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()  
            
            if test_loader is not None:
                test_loss, test_acc = self.test(model, test_loader, loss_fn, device)
                print(f"Train Loss: {train_loss/(batch_idx+1):.6f} | Train Acc: {correct/((batch_idx+1)*len(data)):.6f} | Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.6f}")   
                  
        acc = correct / len(dloader.dataset)
        return train_loss, acc
    
    def train_mask(self, model:nn.Module, mask,optim, dloader, loss_fn, device: torch.device, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            position = kwargs.get("position", 0)
            output = model(data, position=position)
            if kwargs.get("apply_softmax", False):
                output = nn.functional.log_softmax(output, dim=1) # type: ignore
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            for name, param in model.named_parameters():
                if name in mask:
                    param.data *= mask[name].to(device)
            pred = output.argmax(dim=1, keepdim=True)
            # view_as() is used to make sure the shape of pred and target are the same
            if len(target.size()) > 1:
                target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(dloader.dataset)
        return train_loss, acc

    def deep_mutual_train(self, models,optim, dloader, device: torch.device, **kwargs) -> Tuple[list, list]:
        """TODO: generate docstring
        """
        num_net = len(models)   
        for model in models:
            model.train()

        optimizers = [optim]*num_net
        criterion_CE = nn.CrossEntropyLoss()
        criterion_KLD = nn.KLDivLoss(reduction='batchmean')   
        train_loss = [0]*num_net
        pred = [0]*num_net
        correct = [0]*num_net
        train_accuracy=[0]*num_net

        for batch_idx, (data, target) in enumerate(dloader):
            data, target = data.to(device), target.to(device)
            output=[]
            losses=[]
            KLD_loss=[]
            CE_loss=[]
            for i in range(num_net):
                output.append(models[i](data))
            for k in range(num_net):
                CE_loss.append(criterion_CE(output[k],target))
                KLD_loss.append(0)
                for l in range(num_net):
                    if l!=k:
                        KLD_loss[k]+=criterion_KLD(F.log_softmax(output[k],dim=1),F.softmax(output[l],dim=1).detach())
                losses.append(CE_loss[k]+KLD_loss[k]/(num_net-1))
            for i in range(num_net):
                train_loss[i]=losses[i].item()
                pred[i] = output[i].max(1, keepdim = True)[1]
                correct[i] += pred[i].eq(target.view_as(pred[i])).sum().item()
            for i in range(num_net):
                optimizers[i].zero_grad()
                losses[i].backward()
                optimizers[i].step()
        for i in range(num_net):
            train_accuracy[i] = 100.*correct[i]/len(dloader.dataset)
        return train_loss,train_accuracy

    def test(self, model, dloader, loss_fn, device, **kwargs) -> Tuple[float, float]:
        """TODO: generate docstring
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dloader:
                data, target = data.to(device), target.to(device)
                position = kwargs.get("position", 0)
                output = model(data, position=position)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                # view_as() is used to make sure the shape of pred and target are the same
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(dloader.dataset)
        return test_loss, acc

    def save_model(self, model, path):
        if type(model) == DataParallel:
            model_ = model
        else:
            model_ = model
        torch.save(model_.state_dict(), path)

    def move_to_device(self, items: List[Tuple[torch.Tensor, torch.Tensor]],
                       device: torch.device) -> list:
        # Expects a list of tuples with each tupe containing two tensors
        return [[item[0].to(device), item[1].to(device)] for item in items]
    
    def substract_model_weights(self, model1, model2):
        res = {}
        for key, param in model1.items():
            res[key] = param - model2[key]
        return res
    
    def get_last_layer_keys(self, model_wts: OrderedDict[str, Tensor]):
        # Assume one layer is composed of multiple weights named as "layer_name.weight_name" 

        reversed_model_wts = reversed(model_wts)
        last_key = next(reversed_model_wts)
        last_layer = last_key.split(".")[0]
        
        last_layer_keys = []

        while(last_key is not None and last_key.startswith(last_layer + ".")):
            last_layer_keys.append(last_key)
            last_key = next(reversed_model_wts, None)
        return last_layer_keys
    
    def filter_model_weights(self, model_wts: OrderedDict[str, Tensor], key_to_ignore: List[str]):
        # Assume one layer is composed of multiple weights named as "layer_name.weight_name" 

        filtered_model_wts = OrderedDict()
        for key, param in model_wts.items():
            if key not in key_to_ignore:
                filtered_model_wts[key] = param
        return filtered_model_wts
