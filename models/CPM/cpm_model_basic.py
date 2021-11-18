import torch.nn as nn
import torch.nn.functional as F
import torch

# class conv_bn(nn.Module):
#
#      def __init__(self, in_chan, out_chan, kernel_size, padding = 0):
#         super(conv_bn, self).__init__()
#         self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, padding=padding)
#         self.bn = nn.BatchNorm2d(out_chan)
#
#      def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x

class VGG(nn.Module):
    def __init__(self, outputs, ):
        super(VGG, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier[0].in_features, outputs)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, num_classes),
        # )
    def forward(self, x):
        return self.model(x)

if __name__=='__main__':
    VGG()