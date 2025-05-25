import torch
import torch.nn as nn
import torch.nn.functional as func

class MSAttBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        quarter_channel = channels // 4

        self.scale_1 = nn.Sequential(
            nn.Conv2d(channels, quarter_channel, kernel_size=1),
            nn.BatchNorm2d(quarter_channel),
            nn.ReLU(inplace=True)
        )
        self.scale_2 = nn.Sequential(
            nn.Conv2d(channels, quarter_channel, kernel_size=1),
            nn.BatchNorm2d(quarter_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(quarter_channel,quarter_channel, kernel_size=3,padding=1),
            nn.BatchNorm2d(quarter_channel),
            nn.ReLU(inplace=True)
        )
        self.scale_3 = nn.Sequential(
            nn.Conv2d(channels, quarter_channel, kernel_size=1),
            nn.BatchNorm2d(quarter_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(quarter_channel,quarter_channel, kernel_size=3,padding=1),
            nn.BatchNorm2d(quarter_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(quarter_channel,quarter_channel, kernel_size=3,padding=1),
            nn.BatchNorm2d(quarter_channel),
            nn.ReLU(inplace=True),
        )

        self.scale_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, quarter_channel, kernel_size=1),
            nn.BatchNorm2d(quarter_channel),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, in_tensor):
        tensor_1 = self.scale_1(in_tensor)
        tensor_2 = self.scale_2(in_tensor)
        tensor_3 = self.scale_3(in_tensor)
        tensor_4 = self.scale_4(in_tensor)

        size=(in_tensor.shape[2], in_tensor.shape[3])
        tensor_4 = func.interpolate(tensor_4, size=size, mode='bilinear', align_corners=False)

        result = torch.cat((tensor_1, tensor_2, tensor_3, tensor_4), dim=1)
        result = self.fusion(result)
        result *= in_tensor

        return result
