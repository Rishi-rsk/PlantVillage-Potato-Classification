import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

def get_model(num_classes=3):
    # Using default weights (NO DOWNLOAD after first time)
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model
