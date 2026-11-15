import torch.nn as nn
from diffusers import UNet2DModel

WIDTHS = {
    "tiny": (32, 64, 64, 64),
    "small": (64, 128, 128, 128),
    "standard": (128, 256, 256, 256),
}


class UNet(nn.Module):
    def __init__(self, size="small", img_size=32):
        super().__init__()
        widths = WIDTHS[size]
        down = ("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D")
        if img_size == 64:
            # extra full-width level in front so the 32x32 model sits unchanged below it
            # (attention still at 16x16 resolution, same depth per resolution)
            widths, down = (widths[0],) + widths, ("DownBlock2D",) + down
        elif img_size != 32:
            raise ValueError(img_size)
        up = tuple(b.replace("Down", "Up") for b in reversed(down))
        self.img_size = img_size
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=widths,
            down_block_types=down,
            up_block_types=up,
        )

    def forward(self, x, t):
        return self.unet(x, t).sample

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
