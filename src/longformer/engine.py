from .dataset import TextDataset
from .model import LongformerClass

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

import numpy as np


def train_fn(model: LongformerClass,
             optimizer,
             dataloader: DataLoader,
             device: str):             
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        labels = data['labels'].to(device, dtype = torch.long)

        output = model(ids, mask, token_type_ids, labels)

        loss = output.loss.sum()
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
        
    final_loss /= len(dataloader)   
    
    return final_loss


def valid_fn(model, dataloader, device):
    with torch.no_grad():
        final_loss = 0
        valid_preds = []

        for data in dataloader:
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.long)      

            output = model(ids, mask, token_type_ids, labels)
            preds = F.softmax(output.logits, dim=1).detach().cpu().numpy()
            valid_preds.append(preds)

            # loss = loss_fn(output, target)
            loss = output.loss.sum().item()

            final_loss += loss


        final_loss /= len(dataloader)
        valid_preds = np.concatenate(valid_preds)
        # valid_preds = np.argmax(valid_preds, axis=1)

        return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    preds = []
    with torch.no_grad():
        for data in dataloader:
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)                
            labels = data['labels'].to(device, dtype = torch.long)     
#             labels = torch.tensor([1, 1]).view((-1, 1, 1)).to(device, dtype = torch.long)
            output = model(ids, mask, token_type_ids, labels)
            pred = F.softmax(output.logits, dim=1).detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds)[:, 1]      

    return preds
    
