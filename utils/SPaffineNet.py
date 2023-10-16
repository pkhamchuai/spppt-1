import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.SuperPoint import SuperPointFrontend
from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import transform_points_DVF

image_size = 256

# class SP_AffineNet(nn.Module):
#     def __init__(self, model_params):
#         super(SP_AffineNet, self).__init__()
#         self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
#                           conf_thresh=0.015, nn_thresh=0.7, cuda=True)
#         self.affineNet = AffineNet()
#         self.nn_thresh = 0.7
#         self.model_params = model_params

#     def forward(self, source_image, target_image):
#         # source_image = source_image.to(device)
#         # target_image = target_image.to(device)

#         # print('source_image: ', source_image.shape)
#         # print('target_image: ', target_image.shape)
#         points1, desc1, heatmap1 = self.superpoint(source_image[0, 0, :, :].cpu().numpy())
#         points2, desc2, heatmap2 = self.superpoint(target_image[0, 0, :, :].cpu().numpy())

#         if self.model_params.heatmaps == 0:
#             affine_params = self.affineNet(source_image, target_image)
#         elif self.model_params.heatmaps == 1:
#             affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)

#         transformed_source_affine = tensor_affine_transform(source_image, affine_params)

#         # match the points between the two images
#         tracker = PointTracker(5, nn_thresh=0.7)
#         try:
#             matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
#         except:
#             print('No matches found')
#             # TODO: find a better way to do this
#             try:
#                 while matches.shape[1] < 3 and self.nn_thresh > 0.1:
#                     self.nn_thresh = self.nn_thresh - 0.1
#                     matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
#             except:
#                 return transformed_source_affine, affine_params, [], [], [], [], [], [], []

#         # take the elements from points1 and points2 using the matches as indices
#         matches1 = points1[:2, matches[0, :].astype(int)]
#         matches2 = points2[:2, matches[1, :].astype(int)]

#         # transform the points using the affine parameters
#         matches1_transformed = cv2.transform(matches1.T[None, :, :], affine_params[0].cpu().detach().numpy(), (image_size, image_size))
#         return transformed_source_affine, affine_params, matches1, matches2, matches1_transformed, desc1, desc2, heatmap1, heatmap2

#         # return transformed_source_affine, affine_params

# define model
class SP_AffineNet(nn.Module):
    def __init__(self, model_params):
        super(SP_AffineNet, self).__init__()
        self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
                          conf_thresh=0.015, nn_thresh=0.7, cuda=True)
        self.affineNet = AffineNet()
        self.nn_thresh = 0.7
        self.model_params = model_params
        print("\nRunning new version (not run SP on source image)")

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
                    matches = tracker.nn_match_two_way(desc1_2, desc2, nn_thresh=self.nn_thresh)
            except:
                return transformed_source_affine, affine_params, [], [], [], [], [], [], []

        # take the elements from points1 and points2 using the matches as indices
        matches1 = points1[:2, matches[0, :].astype(int)]
        matches2 = points2[:2, matches[1, :].astype(int)]
        try:
            matches1_2 = points1_2[:2, matches[0, :].astype(int)]
        except:
            matches1_2 = transform_points_DVF(matches1.T[None, :, :], 
                                              affine_params.cpu().detach(), image_size)

        # transform the points using the affine parameters
        # matches1_transformed = transform_points(matches1.T[None, :, :], affine_params.cpu().detach())
        return transformed_source_affine, affine_params, matches1, matches2, matches1_2, \
            desc1_2, desc2, heatmap1_2, heatmap2


# class SP_AffineNet(nn.Module):
#     def __init__(self, model_params):
#         super(SP_AffineNet, self).__init__()
#         self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
#                           conf_thresh=0.015, nn_thresh=0.7, cuda=True)
#         self.affineNet = AffineNet()
#         self.nn_thresh = 0.7
#         self.model_params = model_params

#     def forward(self, source_image, target_image):
#         # source_image = source_image.to(device)
#         # target_image = target_image.to(device)

#         # print('source_image: ', source_image.shape)
#         # print('target_image: ', target_image.shape)
#         points1, desc1, heatmap1 = self.superpoint(source_image[0, 0, :, :].cpu().numpy())
#         points2, desc2, heatmap2 = self.superpoint(target_image[0, 0, :, :].cpu().numpy())

#         if self.model_params.heatmaps == 0:
#             affine_params = self.affineNet(source_image, target_image)
#         elif self.model_params.heatmaps == 1:
#             affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)

#         # transform the source image using the affine parameters
#         # using F.affine_grid and F.grid_sample
#         transformed_source_affine = tensor_affine_transform(source_image, affine_params)

#         # match the points between the two images
#         tracker = PointTracker(5, nn_thresh=0.7)
#         try:
#             matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
#         except:
#             print('No matches found')
#             # TODO: find a better way to do this
#             try:
#                 while matches.shape[1] < 3 and self.nn_thresh > 0.1:
#                     self.nn_thresh = self.nn_thresh - 0.1
#                     matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
#             except:
#                 return transformed_source_affine, affine_params, [], [], [], [], [], [], []

#         # take the elements from points1 and points2 using the matches as indices
#         matches1 = points1[:2, matches[0, :].astype(int)]
#         matches2 = points2[:2, matches[1, :].astype(int)]

#         # transform the points using the affine parameters
#         matches1_transformed = transform_points(matches1.T[None, :, :], affine_params.cpu().detach())
#         return transformed_source_affine, affine_params, matches1, matches2, matches1_transformed, desc1, desc2, heatmap1, heatmap2

#         # return transformed_source_affine, affine_params


# class SP_AffineNet(nn.Module):
#     def __init__(self, model_params):
#         super(SP_AffineNet, self).__init__()
#         self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
#                           conf_thresh=0.015, nn_thresh=0.7, cuda=True)
#         self.affineNet = AffineNet()
#         self.nn_thresh = 0.7
#         self.model_params = model_params

#     def forward(self, source_image, target_image):
#         # source_image = source_image.to(device)
#         # target_image = target_image.to(device)

#         # print('source_image: ', source_image.shape)
#         # print('target_image: ', target_image.shape)
#         points1, desc1, heatmap1 = self.superpoint(source_image[0, 0, :, :].cpu().numpy())
#         points2, desc2, heatmap2 = self.superpoint(target_image[0, 0, :, :].cpu().numpy())

#         if self.model_params.heatmaps == 0:
#             affine_params = self.affineNet(source_image, target_image)
#         elif self.model_params.heatmaps == 1:
#             affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)

#         transformed_source_affine = tensor_affine_transform(source_image, affine_params)

#         # match the points between the two images
#         tracker = PointTracker(5, nn_thresh=0.7)
#         try:
#             matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
#         except:
#             print('No matches found')
#             # TODO: find a better way to do this
#             try:
#                 while matches.shape[1] < 3 and self.nn_thresh > 0.1:
#                     self.nn_thresh = self.nn_thresh - 0.1
#                     matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
#             except:
#                 return transformed_source_affine, affine_params, [], [], [], [], [], [], []

#         # take the elements from points1 and points2 using the matches as indices
#         matches1 = points1[:2, matches[0, :].astype(int)]
#         matches2 = points2[:2, matches[1, :].astype(int)]

#         # transform the points using the affine parameters
#         matches1_transformed = cv2.transform(matches1.T[None, :, :], affine_params[0].cpu().detach().numpy(), (image_size, image_size))
#         return transformed_source_affine, affine_params, matches1, matches2, matches1_transformed, desc1, desc2, heatmap1, heatmap2

#         # return transformed_source_affine, affine_params


# class Transformer(nn.Module):
#     def __init__(self, image_size):
#         super(Transformer, self).__init__()
#         self.image_size = image_size

#     def forward(self, source1, grad_grid, affine_grid=None):
#         # image_size = source1.size(-1)
#         y_, x_ = torch.meshgrid(
#             [torch.arange(0, self.image_size).float(), torch.arange(0, self.image_size).float()])
#         y_, x_ = 2.0 * y_ / (self.image_size - 1) - 1.0, 2.0 * x_ / (self.image_size - 1) - 1.0

#         x = grad_grid[0, 0].type(torch.FloatTensor) + x_
#         x = x[np.newaxis, ..., np.newaxis]
#         y = grad_grid[0, 1].type(torch.FloatTensor) + y_
#         y = y[np.newaxis, ..., np.newaxis]
#         grad_grid = torch.cat((x, y), dim=-1)

#         '''xy = torch.stack([x_, y_], 2)
#         xy[:,:,0], \
#         xy[:,:,1]'''
#         x = F.grid_sample(source1.cpu(), grad_grid, align_corners=True) # add grid_x and grid_y ? TODO
#         if affine_grid is not None:
#             x = F.grid_sample(x, affine_grid, align_corners=True)
#         return x

class AffineNet(nn.Module):
    def __init__(self):
        super(AffineNet, self).__init__()
        self.conv1f = 64
        self.conv2f = 128
        self.conv3f = 256
        self.conv4f = 512
        self.conv5f = 512
        self.conv1 = nn.Conv2d(1, self.conv1f, 3, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(self.conv1f, self.conv2f, 3, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(self.conv2f, self.conv3f, 3, padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(self.conv3f, self.conv4f, 3, padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(self.conv4f, self.conv5f, 3, padding=1, padding_mode='zeros')

        self.conv1s = nn.Conv2d(self.conv1f, self.conv1f, 2, stride=2, padding_mode='zeros')
        self.conv2s = nn.Conv2d(self.conv2f, self.conv2f, 2, stride=2, padding_mode='zeros')
        self.conv3s = nn.Conv2d(self.conv3f, self.conv3f, 2, stride=2, padding_mode='zeros')
        self.conv4s = nn.Conv2d(self.conv4f, self.conv4f, 2, stride=2, padding_mode='zeros')
        self.fc1 = nn.Linear(self.conv5f*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 6)

        self.aPooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.ReLU = nn.PReLU()
        self.ReLU = nn.LeakyReLU()
        self.Act1 = nn.GroupNorm(int(self.conv1f/2), self.conv1f, eps=1e-05, affine=True)
        self.Act2 = nn.GroupNorm(int(self.conv2f/2), self.conv2f, eps=1e-05, affine=True)
        self.Act3 = nn.GroupNorm(int(self.conv3f/2), self.conv3f, eps=1e-05, affine=True)
        self.Act4 = nn.GroupNorm(int(self.conv4f/2), self.conv4f, eps=1e-05, affine=True)
        self.Act5 = nn.GroupNorm(int(self.conv5f/2), self.conv5f, eps=1e-05, affine=True)
        '''self.Act1 = nn.BatchNorm2d(self.conv1f)
        self.Act2 = nn.BatchNorm2d(self.conv2f)
        self.Act3 = nn.BatchNorm2d(self.conv3f)
        self.Act4 = nn.BatchNorm2d(self.conv4f)
        self.Act5 = nn.BatchNorm2d(self.conv5f)'''
        '''
        self.pooling = nn.AvgPool2d(2, ceil_mode=False)
        self.elu = nn.ELU()
        self.GN = nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)'''

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
        x = self.Act4(self.ReLU(self.conv4s(self.Act4(self.ReLU(self.conv4(x))))))
        y = self.Act4(self.ReLU(self.conv4s(self.Act4(self.ReLU(self.conv4(y))))))
        # print(x.shape, y.shape)
        x = self.aPooling(self.Act5(self.ReLU(self.conv5(x))))
        y = self.aPooling(self.Act5(self.ReLU(self.conv5(y))))
        # print(x.shape, y.shape)
        t = torch.cat((x, y), dim=1)
        # print(t.shape)
        t = self.fc1(t.flatten())
        # print(t.shape)
        t = self.fc4(self.fc3(self.fc2(t)))
        t = t.view(-1, 2, 3)
        # print(t.shape)

        return t
        # transformed_source_affine = tensor_affine_transform(source_image, t)
        # return transformed_source_affine, t
    