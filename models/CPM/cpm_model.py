import torch.nn as nn
import torch.nn.functional as F
import torch

class conv_bn(nn.Module):

     def __init__(self, in_chan, out_chan, kernel_size, padding = 0):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_chan)

     def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CPM(nn.Module):
    def __init__(self, k):
        super(CPM, self).__init__()
        self.k = k
        self.in_chan = 1#3
        self.sigmoid = nn.Sigmoid()
        self.conv1_stage1 = conv_bn(self.in_chan, 256, kernel_size=3, padding=1)#9, padding=4)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = conv_bn(256, 128, kernel_size=3, padding=1)#9, padding=4)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = conv_bn(128, 128, kernel_size=3, padding=1)#9, padding=4)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage1 = conv_bn(128, 32, kernel_size=3, padding=1)#5, padding=2)
        self.conv5_stage1 = conv_bn(32, 512, kernel_size=3, padding=1)#9, padding=4)
        self.conv6_stage1 = conv_bn(512, 512, kernel_size=1)
        self.conv7_stage1 = conv_bn(512, self.k + 1, kernel_size=1)

        self.conv1_stage2 = conv_bn(self.in_chan, 256, kernel_size=3, padding=1)#9, padding=4)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = conv_bn(256, 128, kernel_size=3, padding=1)#9, padding=4)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage2 = conv_bn(128, 128, kernel_size=3, padding=1)#9, padding=4)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage2 = conv_bn(128, 32, kernel_size=3, padding=1)#5, padding=2)

        self.Mconv1_stage2 = conv_bn(32 + self.k + 1, 256, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv2_stage2 = conv_bn(256, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv3_stage2 = conv_bn(128, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv4_stage2 = conv_bn(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = conv_bn(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage3 = conv_bn(128, 32, kernel_size=3, padding=1)#5, padding=2)

        self.Mconv1_stage3 = conv_bn(32 + self.k + 1, 256, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv2_stage3 = conv_bn(256, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv3_stage3 = conv_bn(128, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv4_stage3 = conv_bn(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage3 = conv_bn(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage4 = conv_bn(128, 32, kernel_size=3, padding=1)#5, padding=2)

        self.Mconv1_stage4 = conv_bn(32 + self.k + 1, 256, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv2_stage4 = conv_bn(256, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv3_stage4 = conv_bn(128, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv4_stage4 = conv_bn(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage4 = conv_bn(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage5 = conv_bn(128, 32, kernel_size=3, padding=1)#5, padding=2)

        self.Mconv1_stage5 = conv_bn(32 + self.k + 1, 256, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv2_stage5 = conv_bn(256, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv3_stage5 = conv_bn(128, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv4_stage5 = conv_bn(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage5 = conv_bn(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage6 = conv_bn(128, 32, kernel_size=3, padding=1)#5, padding=2)

        self.Mconv1_stage6 = conv_bn(32 + self.k + 1, 256, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv2_stage6 = conv_bn(256, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv3_stage6 = conv_bn(128, 128, kernel_size=3, padding=1)#11, padding=5)
        self.Mconv4_stage6 = conv_bn(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage6 = conv_bn(128, self.k + 1, kernel_size=1, padding=0)

    #def rgb2fourier(self, image):
        

    def _stage1(self, image):
        x = self.pool1_stage1(F.relu(self.conv1_stage1(image)))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x)))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x)))
        x = F.relu(self.conv4_stage1(x))
        x = F.relu(self.conv5_stage1(x))
        x = F.relu(self.conv6_stage1(x))
        x = self.sigmoid(self.conv7_stage1(x))

        return x

    def _middle(self, image):
        x = self.pool1_stage2(F.relu(self.conv1_stage2(image)))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x)))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x)))
        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map):

        x = F.relu(self.conv4_stage2(pool3_stage2_map))
        x = torch.cat([x, conv7_stage1_map], dim=1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.sigmoid(self.Mconv5_stage2(x))

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map):

        x = F.relu(self.conv1_stage3(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage2_map], dim=1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.sigmoid(self.Mconv5_stage3(x))

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map):

        x = F.relu(self.conv1_stage4(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage3_map], dim=1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.sigmoid(self.Mconv5_stage4(x))

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map):

        x = F.relu(self.conv1_stage5(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage4_map], dim=1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.sigmoid(self.Mconv5_stage5(x))

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map):
        
        x = F.relu(self.conv1_stage6(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage5_map], dim=1)
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x))
        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x))
        x = self.sigmoid(self.Mconv5_stage6(x))

        return x



    def forward(self, image):
    	#fourier_features = self.rgb2fourier(image)
        conv7_stage1_map = self._stage1(image)
        pool3_stage2_map = self._middle(image)
        #conv7_stage1_map = self._stage1(image)
        #pool3_stage2_map = self._middle(image)

        Mconv5_stage2_map = self._stage2(pool3_stage2_map, conv7_stage1_map)
        Mconv5_stage3_map = self._stage3(pool3_stage2_map, Mconv5_stage2_map)
        Mconv5_stage4_map = self._stage4(pool3_stage2_map, Mconv5_stage3_map)
        Mconv5_stage5_map = self._stage5(pool3_stage2_map, Mconv5_stage4_map)
        Mconv5_stage6_map = self._stage6(pool3_stage2_map, Mconv5_stage5_map)

        return conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map, Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map

