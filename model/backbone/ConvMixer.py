import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1,kernel_size), groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        y = self.Resnet(x)
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=5, patch_size=5, n_classes=60):
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1,patch_size), stride=(1,patch_size)),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.ConvMixer_blocks = nn.ModuleList([])
        self.conv_transpose =nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=(1, 5), stride=(1, 5)))
        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=dim, kernel_size=kernel_size))

        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(dim, n_classes)
        # )

    def forward(self, x):
        x = self.conv2d1(x)
        for ConvMixer_block in self.ConvMixer_blocks:
            x = ConvMixer_block(x)

        # x = self.head(x)

        return self.conv_transpose(x)


if __name__ == '__main__':
    x = torch.randn(1, 64, 300,25)
    convmixer = ConvMixer(dim=64, depth=1)
    out = convmixer(x)
    print(out.shape)  # [1, 1000]