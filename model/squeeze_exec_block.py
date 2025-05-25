import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gap_layer = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Sequential(
            nn.Linear(channels,channels//16),
            nn.ReLU(inplace=True),
            nn.Linear(channels//16,channels),
            nn.Sigmoid()
        )

    def forward(self, in_tensor):
        b,c,h,w = in_tensor.size()

        avg_tensor = self.gap_layer(in_tensor)
        avg_tensor = avg_tensor.view(b,c)

        out_tensor = self.fc_layer(avg_tensor)
        out_tensor = out_tensor.view(b,c,1,1)

        scaled_tensor = out_tensor.expand_as(in_tensor)
        scaled_tensor *= in_tensor

        return scaled_tensor


