import argparse
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import ransac
# from skimage.transform import FundamentalMatrixTransform, AffineTransform

# Suppress the specific warning
# import warnings
import csv
# import sys
# from IPython.utils.capture import capture_output
from datetime import datetime
from tqdm import tqdm

import torch
# from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F

from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, DL_affine_plot
from utils.SPaffineNet import SP_AffineNet
from utils.datagen import datagen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

def test(model, model_params, timestamp):
    # Set model to training mode
    model.eval()

    # Create output directory
    # print(model_params.get_model_code())
    output_dir = f"output/{model_params.get_model_code()}_{timestamp}_test"
    print(f'Output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # Validate model
    # validation_loss = 0.0

    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # matches1_transformed.shape[-1], mse_before, mse12, tre_before, tre12, \
        # mse12_image, ssim12_image, 
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", "mse12_image_before", "mse12_image", "ssim12_image_before", "ssim12_image"])

    with torch.no_grad():
        testbar = tqdm(test_dataset, desc=f'Testing:')
        for i, data in enumerate(testbar, 0):
            # Get images and affine parameters
            if model_params.sup:
                source_image, target_image, affine_params_true = data
            else:
                source_image, target_image = data
            source_image = source_image.to(device)
            target_image = target_image.to(device)

            # Forward pass
            outputs = model(source_image, target_image)
            # for i in range(len(outputs)):
            #     print(i, outputs[i].shape)
            transformed_source_affine = outputs[0]
            affine_params_predicted = outputs[1]
            points1 = outputs[2]
            points2 = outputs[3]
            points1_affine = np.array(outputs[4])
            try:
                points1_affine = points1_affine.reshape(points1_affine.shape[2], points1_affine.shape[1])
            except:
                pass
            desc1 = outputs[5]
            desc2 = outputs[6]
            heatmap1 = outputs[7]
            heatmap2 = outputs[8]
            
            if i < 50:
                plot_ = True
            else:
                plot_ = False

            results = DL_affine_plot(f"{i+1}", output_dir,
                    f"{i}", "_", source_image[0, 0, :, :].cpu().numpy(), target_image[0, 0, :, :].cpu().numpy(), \
                    transformed_source_affine[0, 0, :, :].cpu().numpy(), \
                    points1, points2, points1_affine, desc1, desc2, \
                    affine_params=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=plot_)

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

            # write metrics to csv file
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file) # TODO: might need to export true & predicted affine parameters too
                writer.writerow([i, mse_before, mse12, tre_before, tre12, mse12_image_before, mse12_image, ssim12_image_before, ssim12_image])

    # delete all txt files in output_dir
    for file in os.listdir(output_dir):
        if file.endswith(".txt"):
            os.remove(os.path.join(output_dir, file))

    print('Finished testing')
    print(f'Images and metrics saved in {output_dir}')


if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')
    parser.add_argument('--model_path', type=str, help='path to model to load')
    args = parser.parse_args()
    print('args.model_path: ', args.model_path)

    model_params = ModelParams.model_code_from_model_path(args.model_path)
    
    train_dataset = datagen(model_params.dataset, True, model_params.sup)
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Get sample batch
    print('Train set: ', [x.shape for x in next(iter(train_dataset))])
    print('Test set: ', [x.shape for x in next(iter(test_dataset))])

    model = SP_AffineNet(model_params).to(device)
    print(model)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, model_params.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_params.decay_rate ** epoch)
    model_path = args.model_path

    # if a model is loaded, the training will continue from the epoch it was saved at
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        model_params.start_epoch = int(model_path.split('/')[-1].split('_')[2])
        print(f'Loaded model from {model_path}\nStart testing from epoch {model_params.start_epoch}')
    else:
        print('A model path needed to be given to test a model')
        exit()

    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print("Test model +++++++++++++++++++++++++++++")

    test(model, model_params, timestamp)

