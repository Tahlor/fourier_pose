import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.linear_model import LinearRegression

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

class CPM(nn.Module):
    """ VGG outputs a bunch of positions
            # position * projection = x*a+y*b
            # prediction = sin ( 2pi * position * projection )
            # arccos(prediction) / (2pi) = x*a+y*b
            # arcsine(prediction) / (2pi) = x*a+y*b
                # Network just learns that these both need to be x*a + y*b
                # Projection Matrix * [x y] = Predictions
            # X = torch.linalg.solve(A, [pred_arcsine, pred_arccos])

    """
    def __init__(self, k, mapping_size=32, hidden_size=4096, baseline=False):
        """ k = number of keypoints
        """
        super().__init__()
        print("baseline",baseline)
        self.B_gauss = np.random.normal(size=(mapping_size, 2))
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        if baseline:
            output_dim = k*2
        else:
            output_dim = k * mapping_size * 2
        if False:
            self.model.classifier = nn.Linear(self.model.classifier[0].in_features, k*mapping_size*2)
        else:
            self.model.classifier = nn.Sequential(
                     nn.Linear(self.model.classifier[0].in_features, hidden_size),
                     nn.ReLU(),
                     nn.Linear(hidden_size, output_dim)
            )
        print("IN FEATURES", self.model.classifier[0].in_features)
        print("")
        # self.freeze(max_layer=20)
        # print(list(self.model.modules()))

    def freeze(self, max_layer=20):
        layer_counter = 0
        for (name, module) in self.model.named_children():
            if name == 'features':
                for layer in module.children():
                    for param in layer.parameters():
                        param.requires_grad = False

                    print('Layer "{}" in module "{}" was frozen!'.format(layer_counter, name))
                    layer_counter += 1
                    if layer_counter > max_layer:
                        return

    def forward(self, x):
        if True:
            return torch.tanh(self.model(x))
        else:
            return self.model(x)

    def calc_best_guess(self,preds):
        return calc_best_guess(preds, self.B_gauss)

regressor = LinearRegression()

def calc_best_guess(preds,B_guass):
    # split preds
    # preds1 = torch.arcsin(preds1) / (2 * np.pi * B_gauss)
    # preds2 = torch.arccos(preds2) / (2 * np.pi * B_gauss)
    # recombine preds
    solution = regressor.fit(B_guass, preds)
    return solution

if __name__=='__main__':
    CPM()

