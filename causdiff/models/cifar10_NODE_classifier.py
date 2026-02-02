import torch
import torch.nn as nn


# Convolutional Feature Extractor for CIFAR-10
class CIFAR10FeatureExtractor(nn.Module):
    def __init__(self):
        super(CIFAR10FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            # Block 1: [N,3, 32, 32] -> [N,64, 16, 16]
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: [N,64, 16, 16] -> [N,128, 8, 8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class ODEFunc(nn.Module):
    def __init__(self, channels):
        super(ODEFunc, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels), nn.ReLU(), nn.Linear(channels, channels)
        )

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, t, x):
        t_tensor = torch.tensor([[t]], device=x.device, dtype=x.dtype)  # [1, 1]
        t_emb = self.time_embed(t_tensor)  # [1, channels]
        t_emb = t_emb.view(1, -1, 1, 1)  # Reshape to [1, channels, 1, 1]

        # Add the time embedding to x (broadcasting over batch, height, and width)
        x = x + t_emb
        return self.net(x)


# Regular Euler Integration Block: simulate continuous evolution via fixed Euler steps.
class EulerODEBlock(nn.Module):
    def __init__(self, odefunc, t0=0.0, t1=1.0, steps=10_000):
        super(EulerODEBlock, self).__init__()
        self.odefunc = odefunc
        self.t0 = t0
        self.t1 = t1
        self.steps = steps

    def forward(self, x):
        dt = (self.t1 - self.t0) / self.steps
        t = self.t0
        out = x
        outputs = [x]  # Collect initial state
        for _ in range(self.steps):
            out = out + dt * self.odefunc(t, out)  # Euler update: x = x + dt * f(t, x)
            t += dt
            outputs.append(out)
        return torch.stack(outputs, dim=0)  # [steps+1, batch, channels, H, W]


class NeuralODEClassifier(nn.Module):
    def __init__(self, num_classes=10, steps=5, device=None):
        super(NeuralODEClassifier, self).__init__()
        self.feature_extractor = CIFAR10FeatureExtractor()
        # ODE function operates on 128-channel feature maps.
        self.odeblock = EulerODEBlock(
            ODEFunc(channels=128), t0=0.0, t1=1.0, steps=steps
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

        # If a device is provided, move the model there.
        if device is not None:
            self.to(device)

    def forward(self, x):
        features = self.feature_extractor(x)  # [batch, 128, 8, 8]
        ode_states = self.odeblock(features)  # [steps+1, batch, 128, 8, 8]
        final_state = ode_states[-1]  # Use the final state (at t1)
        pooled = self.avgpool(final_state)  # [batch, 128, 1, 1]
        flat = pooled.view(pooled.size(0), -1)  # [batch, 128]
        logits = self.classifier(flat)  # [batch, num_classes]
        return logits
