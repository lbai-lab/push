import torchvision
import torch.nn as nn


class Modifiedvit_b_16(nn.Module):
    def __init__(self):
        super(Modifiedvit_b_16, self).__init__()
        # self.vit_b_16 = torchvision.models.VisionTransformer(image_size=28, patch_size=14, num_classes=10, num_heads=4, num_layers=20, mlp_dim=1024, hidden_dim=128)
        self.vit_b_16 = torchvision.models.VisionTransformer(image_size=28, patch_size=14, num_classes=10, num_heads=8, num_layers=16, mlp_dim=1280, hidden_dim=320)

    def forward(self, x):
        return self.vit_b_16(x)


class Modifiedvit_b_16_adjust(nn.Module):
    def __init__(self, num_heads, num_layers, mlp_dim, hidden_dim):
        super(Modifiedvit_b_16_adjust, self).__init__()
        # self.vit_b_16 = torchvision.models.VisionTransformer(image_size=28, patch_size=14, num_classes=10, num_heads=4, num_layers=20, mlp_dim=1024, hidden_dim=128)
        self.vit_b_16 = torchvision.models.VisionTransformer(image_size=28, patch_size=14, num_classes=10, num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        return self.vit_b_16(x)
