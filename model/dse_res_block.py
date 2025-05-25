import torch.nn as nn
from squeeze_exec_block import SEBlock

class DSEResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,expand_ratio=4):
        super().__init__()

        self.dropout = nn.Dropout2d(0.2)
        self.expand_ratio = expand_ratio

        expanded_channel = in_channel*expand_ratio
        if in_channel==out_channel and stride==1:
            self.use_residual = True
        else:
            self.use_residual = False

        self.expand_layer=nn.Sequential(
            nn.Conv2d(in_channel,expanded_channel,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(expanded_channel),
            nn.ReLU(inplace=True)
        )

        self.depthwise_layer = nn.Sequential(
            nn.Conv2d(expanded_channel,expanded_channel,kernel_size=3,stride=stride,bias=False),
            nn.BatchNorm2d(expanded_channel),
            nn.ReLU(inplace=True),
        )

        self.se_block = SEBlock(expanded_channel)

        self.project_layer = nn.Sequential(
            nn.Conv2d(expanded_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel)
        )



    def forward(self, in_tensor):
        if self.expand_ratio!=1:
            expand_tensor = self.expand_layer(in_tensor)
        else:
            expand_tensor = in_tensor

        out_tensor = self.depthwise_layer(expand_tensor)
        out_tensor = self.se_block(out_tensor)
        out_tensor = self.project_layer(out_tensor)
        if self.use_residual:
            out_tensor += self.dropout(out_tensor)

        return out_tensor