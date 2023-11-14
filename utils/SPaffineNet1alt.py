import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from utils.SuperPoint import SuperPointFrontend
from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import transform_points_DVF
from pytorch_model_summary import summary

image_size = 256

# define model
class SP_AffineNet1alt(nn.Module):
    def __init__(self, model_params):
        super(SP_AffineNet1alt, self).__init__()
        self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
                          conf_thresh=0.015, nn_thresh=0.7, cuda=True)
        self.affineNet = AffineNet1_alt()
        self.nn_thresh = 0.7
        self.model_params = model_params
        print("\nRunning new version (not run SP on source image)")
        
        inputs = torch.rand((1, 1, image_size, image_size)), torch.rand((1, 1, image_size, image_size))
        summary(self.affineNet, *inputs, show_input=True, show_hierarchical=True, print_summary=True)

    def forward(self, source_image, target_image):
        # source_image = source_image.to(device)
        # target_image = target_image.to(device)

        # print('source_image: ', source_image.shape)
        # print('target_image: ', target_image.shape)
        points1, desc1, heatmap1 = self.superpoint(source_image[0, 0, :, :].cpu().numpy())
        points2, desc2, heatmap2 = self.superpoint(target_image[0, 0, :, :].cpu().numpy())

        if self.model_params.heatmaps == 0:
            affine_params = self.affineNet(source_image, target_image)
        elif self.model_params.heatmaps == 1:
            print("This part is not yet implemented.")
            # affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)

        # transform the source image using the affine parameters
        # using F.affine_grid and F.grid_sample
        transformed_source_affine = tensor_affine_transform(source_image, affine_params)
        points1_2, desc1_2, heatmap1_2 = self.superpoint(transformed_source_affine[0, 0, :, :].detach().cpu().numpy())

        # match the points between the two images
        tracker = PointTracker(5, nn_thresh=0.7)
        try:
            matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
        except:
            # print('No matches found')
            # TODO: find a better way to do this
            try:
                while matches.shape[1] < 3 and self.nn_thresh > 0.1:
                    self.nn_thresh = self.nn_thresh - 0.1
                    matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
            except:
                return transformed_source_affine, affine_params, [], [], [], [], [], [], []

        # take the elements from points1 and points2 using the matches as indices
        matches1 = np.array(points1[:2, matches[0, :].astype(int)])
        matches2 = np.array(points2[:2, matches[1, :].astype(int)])
        # matches1_2 = np.array(points1_2[:2, matches[0, :].astype(int)])
        # print('matches1', matches1)
        # print('matches2', matches2)
        # print('matches1_2', matches1_2)

        # try:
        #     matches1_2 = points1_2[:2, matches[0, :].astype(int)]
        # except:
        # print(affine_params.cpu().detach().shape, transformed_source_affine.shape)
        matches1_2 = transform_points_DVF(torch.tensor(matches1), 
                        affine_params.cpu().detach(), transformed_source_affine)

        # transform the points using the affine parameters
        # matches1_transformed = transform_points(matches1.T[None, :, :], affine_params.cpu().detach())
        return transformed_source_affine, affine_params, matches1, matches2, matches1_2, \
            desc1_2, desc2, heatmap1_2, heatmap2

class AffineNet1_alt(nn.Module):
    def __init__(self):
        super(AffineNet1_alt, self).__init__()
        self.filter = [64, 128, 256, 512]
        self.conv1  = nn.Conv2d(1,              self.filter[0], 3, padding=1, padding_mode='zeros')
        self.conv1s = nn.Conv2d(self.filter[0], self.filter[0], 2, stride=2, padding_mode='zeros')
        self.conv2  = nn.Conv2d(self.filter[0], self.filter[1], 3, padding=1, padding_mode='zeros')
        self.conv2s = nn.Conv2d(self.filter[1], self.filter[1], 2, stride=2, padding_mode='zeros')
        self.conv3  = nn.Conv2d(self.filter[1], self.filter[2], 3, padding=1, padding_mode='zeros')
        self.conv3s = nn.Conv2d(self.filter[2], self.filter[2], 2, stride=2, padding_mode='zeros')

        self.fc1 = nn.Linear(self.filter[2]*2, 6)

        self.dropout = nn.Dropout(p=0.7)
        self.aPooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.ReLU = nn.PReLU()
        self.ReLU = nn.LeakyReLU()
        self.Act1 = nn.GroupNorm(int(self.filter[0]/2), self.filter[0], eps=1e-05, affine=True)
        self.Act2 = nn.GroupNorm(int(self.filter[1]/2), self.filter[1], eps=1e-05, affine=True)
        self.Act3 = nn.GroupNorm(int(self.filter[2]/2), self.filter[2], eps=1e-05, affine=True)

        # self.flow = torch.nn.Parameter(torch.zeros(1, 2, image_size, image_size).to(device), requires_grad=True)
        # self.img = torch.nn.Parameter(torch.zeros(1, 1, image_size, image_size).to(device), requires_grad=True)

    def forward(self, source_image, target_image):

        # print(source_image.size(), target_image.size())
        # x = self.conv1(source_image)
        x = self.Act1(self.ReLU(self.conv1s(self.Act1(self.ReLU(self.conv1(source_image))))))
        y = self.Act1(self.ReLU(self.conv1s(self.Act1(self.ReLU(self.conv1(target_image))))))
        # print(x.shape, y.shape)
        x = self.Act2(self.ReLU(self.conv2s(self.Act2(self.ReLU(self.conv2(x))))))
        y = self.Act2(self.ReLU(self.conv2s(self.Act2(self.ReLU(self.conv2(y))))))
        # print(x.shape, y.shape)
        x = self.Act3(self.ReLU(self.conv3s(self.Act3(self.ReLU(self.conv3(x))))))
        y = self.Act3(self.ReLU(self.conv3s(self.Act3(self.ReLU(self.conv3(y))))))
        # print(x.shape, y.shape)
        x = self.aPooling(x)
        y = self.aPooling(y)
        # print(x.shape, y.shape)
        t = torch.cat((x, y), dim=1)
        # print(t.shape)
        t = self.fc1(t.flatten())
        t = t.view(-1, 2, 3)
        # print(t.shape)

        return t

    