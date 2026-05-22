import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import cohen_kappa_score
import numpy as np

def get_class_weights(df, device):
    counts = df['diagnosis'].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float).to(device)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss,correct,total = 0,0,0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += len(labels)
    
    return total_loss/len(loader), correct/total

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss,correct,total = 0,0,0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            preds = outputs.argmax(1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return total_loss / len(loader), correct / total, kappa