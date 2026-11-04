import torch.nn as nn
from diffusers import UNet2DModel

WIDTHS = {
    "small": (64, 128, 128, 128),
    "standard": (128, 256, 256, 256),
}


class UNet(nn.Module):
    def __init__(self, size="small", img_size=32):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=WIDTHS[size],
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, t):
        return self.unet(x, t).sample

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
