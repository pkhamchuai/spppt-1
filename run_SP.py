import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
import matplotlib.pyplot as plt
import random
import math
from skimage.metrics import structural_similarity as ssim
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, AffineTransform
# Suppress the specific warning
import warnings
import csv
import sys
from IPython.utils.capture import capture_output
from datetime import datetime

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

from utils.SuperPoint import SuperPointFrontend, PointTracker, load_image
from utils.datagen import datagen
from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, print_summary, DL_affine_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

image_size = 256

def process_image(image):
    # squeeze dimensions 0 and 1
    image = image.squeeze(0).squeeze(0)
    # convert to numpy array
    image = image.cpu().numpy()
    # normalize image to range 0 to 1
    image = (image/np.max(image)).astype('float32')
    return image

def run(model_params):
    # Initialize SuperPointFrontend
    superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
                          conf_thresh=0.015,
                          nn_thresh=0.7, cuda=True)
    
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Create output directory
    output_dir = f"output/{args.model}_{model_params.get_model_code()}_{timestamp}_test"
    os.makedirs(output_dir, exist_ok=True)

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

    testbar = tqdm(test_dataset, desc=f'Testing:')
    for i, data in enumerate(testbar, 0):
        # Get images and affine parameters
        if model_params.sup:
            source_image, target_image, affine_params_true = data
        else:
            source_image, target_image = data
            affine_params_true = None
            
        # process images
        source_image = process_image(source_image)
        target_image = process_image(target_image)

        # Process the first image
        points1, desc1, heatmap1 = superpoint(source_image)
        # Process the second image
        points2, desc2, heatmap2 = superpoint(target_image)

        # match the points between the two images
        tracker = PointTracker(5, nn_thresh=0.7)
        matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=0.7)

        # take the elements from points1 and points2 using the matches as indices
        matches1 = points1[:2, matches[0, :].astype(int)]
        matches2 = points2[:2, matches[1, :].astype(int)]

        # create affine transform matrix from points1 to points2
        # and apply it to points1
        affine_transform1 = cv2.estimateAffinePartial2D(matches1.T, matches2.T)
        matches1_transformed = cv2.transform(matches1.T[None, :, :], affine_transform1[0])
        matches1_transformed = matches1_transformed[0].T

        # transform image 1 and 2 using the affine transform matrix
        transformed_source_affine = cv2.warpAffine(source_image, affine_transform1[0], (256, 256))

        mse12 = np.mean((matches1_transformed - matches2)**2)
        tre12 = np.mean(np.sqrt(np.sum((matches1_transformed - matches2)**2, axis=0)))

        if i < 100:
            plot_ = True
        else:
            plot_ = False

        results = DL_affine_plot(f"{i+1}", output_dir,
                f"{i}", "_", source_image, target_image, \
                transformed_source_affine, \
                matches1, matches2, matches1_transformed, desc1, desc2, 
                affine_params_true=affine_params_true,
                affine_params_predict=affine_transform1, 
                heatmap1=heatmap1, heatmap2=heatmap2, plot=plot_)


        # calculate metrics
        # matches1_transformed = results[0]
        mse_before = results[1]
        mse12 = results[2]
        tre_before = results[3]
        tre12 = results[4]
        mse12_image_before = results[5]
        mse12_image = results[6]
        ssim12_image_before = results[7]
        ssim12_image = results[8]

        # append metrics to metrics list
        metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, 
            mse12_image, ssim12_image_before, ssim12_image, matches2.shape[-1]])
        
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", "mse12_image_before", "mse12_image", "ssim12_image_before", "ssim12_image", "num_points"])
        for i in range(len(metrics)):
            writer.writerow(metrics[i])
        # write the average and std of the metrics
        metrics = np.array(metrics)
        nan_mask = np.isnan(metrics).any(axis=1)
        metrics = metrics[~nan_mask]
        avg = ["average", np.mean(metrics[:, 1]), np.mean(metrics[:, 2]), np.mean(metrics[:, 3]), np.mean(metrics[:, 4]), 
            np.mean(metrics[:, 5]), np.mean(metrics[:, 6]), np.mean(metrics[:, 7]), np.mean(metrics[:, 8])]
        std = ["std", np.std(metrics[:, 1]), np.std(metrics[:, 2]), np.std(metrics[:, 3]), np.std(metrics[:, 4]), 
            np.std(metrics[:, 5]), np.std(metrics[:, 6]), np.std(metrics[:, 7]), np.std(metrics[:, 8])]
        writer.writerow(avg)
        writer.writerow(std)

    print(f"The test results are saved in {csv_file}")

    # delete all txt files in output_dir
    # for file in os.listdir(output_dir):
    #     if file.endswith(".txt"):
    #         os.remove(os.path.join(output_dir, file))

    print_summary(args.model, None, model_params, 
                  None, timestamp, True)


# if main
# from utils.utils1 import transform_points_DVF
if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='image used for training')
    parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model', type=str, default='SP', help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    args = parser.parse_args()

    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, heatmaps=args.heatmaps, 
                               loss_image=args.loss_image, num_epochs=args.num_epochs, 
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run(model_params)