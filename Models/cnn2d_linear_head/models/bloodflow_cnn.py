import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, List

# =========================================================
# SE Block
# =========================================================
class SEBlock2D(nn.Module):
    """2D Squeeze-and-Excitation block for adaptive channel weighting."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        squeeze = x.view(b, c, -1).mean(dim=2)  # Global average pooling
        excitation = F.relu(self.fc1(squeeze), inplace=True)
        excitation = torch.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)
        return x * excitation

# =========================================================
# BloodFlow CNN 2D
# =========================================================
class BloodFlowCNN2D(nn.Module):
    """
    2D CNN for speckle pattern blood flow regression.
    Input: (B, 1, H, W)
    Output: (B,) scalar flow value
    """
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        reduction: int = 16,
        dropout_rate: float = 0.4,
        input_norm: bool = True,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(input_channels) if input_norm else nn.Identity()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(inplace=True),

            SEBlock2D(base_channels * 4, reduction=reduction),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(inplace=True),
        )

        # Global pooling + regression head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(base_channels * 8, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        x = self.input_norm(x)
        x = self.encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)  # (B,)

    def _initialize_weights(self):
        """Kaiming-normal initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _validate_input(x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.size(1)} channels")

# =========================================================
# Factory function
# =========================================================
def get_model(**kwargs) -> BloodFlowCNN2D:
    return BloodFlowCNN2D(**kwargs)

# =========================================================
# Utilities for plotting
# =========================================================
def plot_loss_curve(train_losses: List[float], val_losses: List[float], save_path: Optional[str]=None):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_predictions_vs_truth(y_true: List[float], y_pred: List[float], save_path: Optional[str]=None, flow_rates: List[float]=[0,5,50,100,150,400]):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.6, edgecolors='k')
    plt.plot([min(flow_rates), max(flow_rates)], [min(flow_rates), max(flow_rates)], 'r--', lw=2, label="Ideal")
    plt.xticks(flow_rates)
    plt.yticks(flow_rates)
    plt.xlabel("True Flowrate (uL/min)")
    plt.ylabel("Predicted Flowrate (uL/min)")
    plt.title("Predictions vs True")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# =========================================================
# Quick sanity check
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    dummy_input = torch.randn(2,1,128,128).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)  # (B,)