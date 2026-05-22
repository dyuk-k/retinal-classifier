import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=5, freeze_backbone=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3),nn.Linear(in_features, 256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256, num_classes))
    
    return model