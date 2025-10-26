import torch
import torch.nn as nn
import torchvision.models as models
from simple_config import *


class LightweightFineTuned(nn.Module):
    
    def __init__(self, num_classes=4):
        super(LightweightFineTuned, self).__init__()
        
        print("📥 Loading pre-trained ResNet18...")
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        
        # Adapt first layer for grayscale spectrograms (1 channel instead of 3)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1, keepdim=True))
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Freeze early layers (keep general pattern recognition)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        
        # Train later layers (adapt to underwater sounds)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_confidence=False):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)
        
        if return_confidence:
            confidence = self.confidence_head(features)
            return logits, confidence
        
        return logits


def create_model(num_classes=4):
    print("🧠 Creating underwater sound classifier...")
    print("🚀 Fine-tuning pre-trained ResNet18")
    return LightweightFineTuned(num_classes=num_classes)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    model = create_model(num_classes=4)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Percentage trainable: {100*trainable/total:.1f}%")
    
    x = torch.randn(2, 1, 128, 1024)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print("✓ Model test successful!")
