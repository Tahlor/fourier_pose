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
    """ VGG outputs a bunch of positions
            # position * projection = x*a+y*b
            # prediction = sin ( 2pi * position * projection )
            # arcsine(prediction) / (2pi * projection) = position

            # arccos(prediction) / (2pi) = x*a+y*b
            # arcsine(prediction) / (2pi) = x*a+y*b
                # Network just learns that these both need to be x*a + y*b
                # Projection Matrix * [x y] = Predictions
            # X = torch.linalg.solve(A, [pred_arcsine, pred_arccos])

    """
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

def calc_best_guess(preds,B_guass):
    # split preds
    preds1 = torch.arcsin(preds1) / (2 * np.pi * B_gauss)
    preds2 = torch.arccos(preds2) / (2 * np.pi * B_gauss)
    # recombine preds

    regressor = LinearRegression()

    solution = regressor.fit(B_guass, preds)
    return solution

if __name__=='__main__':
    VGG()

    import numpy as np
    from sklearn.linear_model import LinearRegression

    mapping_size = 256
    B_gauss = np.random.normal((mapping_size, 2))
