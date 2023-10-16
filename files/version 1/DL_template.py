import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import ransac
# from skimage.transform import FundamentalMatrixTransform, AffineTransform

# Suppress the specific warning
import warnings
import csv
import sys
from IPython.utils.capture import capture_output
from datetime import datetime
from tqdm import tqdm

import torch
from torchvision import transforms
from torch import nn, optim
import torch.nn.functional as F
# from torch.utils import data

# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor

from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, DL_affine_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

model_params = ModelParams(sup=0, dataset=1, image=1, heatmaps=0, 
                           loss_image=2, num_epochs=30)
model_params.print_explanation()

from utils.SuperPoint import SuperPointFrontend
from utils.SPaffineNet import AffineNet

# define model
class SP_AffineNet(nn.Module):
    def __init__(self):
        super(SP_AffineNet, self).__init__()
        self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
                          conf_thresh=0.015, nn_thresh=0.7, cuda=True)
        self.affineNet = AffineNet()
        self.nn_thresh = 0.7

    def forward(self, source_image, target_image):
        # source_image = source_image.to(device)
        # target_image = target_image.to(device)

        # print('source_image: ', source_image.shape)
        # print('target_image: ', target_image.shape)
        points1, desc1, heatmap1 = self.superpoint(source_image[0, 0, :, :].cpu().numpy())
        points2, desc2, heatmap2 = self.superpoint(target_image[0, 0, :, :].cpu().numpy())

        if model_params.heatmaps == 0:
            affine_params = self.affineNet(source_image, target_image)
        elif model_params.heatmaps == 1:
            affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)

        transformed_source_affine = tensor_affine_transform(source_image, affine_params)

        # match the points between the two images
        tracker = PointTracker(5, nn_thresh=0.7)
        try:
            matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
        except:
            print('No matches found')
            # TODO: find a better way to do this
            try:
                while matches.shape[1] < 3 and self.nn_thresh > 0.1:
                    self.nn_thresh = self.nn_thresh - 0.1
                    matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
            except:
                return transformed_source_affine, affine_params, [], [], [], [], [], [], []

        # take the elements from points1 and points2 using the matches as indices
        matches1 = points1[:2, matches[0, :].astype(int)]
        matches2 = points2[:2, matches[1, :].astype(int)]

        # transform the points using the affine parameters
        matches1_transformed = cv2.transform(matches1.T[None, :, :], affine_params[0].cpu().detach().numpy(), (image_size, image_size))
        return transformed_source_affine, affine_params, matches1, matches2, matches1_transformed, desc1, desc2, heatmap1, heatmap2

        # return transformed_source_affine, affine_params

        
from utils.datagen import datagen
        
train_dataset = datagen(model_params.dataset, True, model_params.sup)
test_dataset = datagen(model_params.dataset, False, model_params.sup)

# Get sample batch
print('Train set: ', [x.shape for x in next(iter(train_dataset))])
print('Test set: ', [x.shape for x in next(iter(test_dataset))])

model = SP_AffineNet().to(device)
print(model)

parameters = model.parameters()
optimizer = optim.Adam(parameters, model_params.learning_rate)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_params.decay_rate ** epoch)
model_path = 'trained_models/01102_0.001_20_1_20230930-091532.pth'

# if a model is loaded, the training will continue from the epoch it was saved at
try:
    model.load_state_dict(torch.load(model_path))
    model_params.start_epoch = int(model_path.split('/')[-1].split('_')[2])
    print(f'Loaded model from {model_path}\nStarting at epoch {model_params.start_epoch}')
except:
    model_params.start_epoch = 0
    print('No model loaded, starting from scratch')

# print case
print(model_params)
model_params.print_explanation()

# Define training function
def train(model, model_params, timestamp):
    # Set model to training mode
    model.train()

    # Define loss function based on supervised or unsupervised learning
    criterion = model_params.loss_image
    if model_params.sup:
        criterion_affine = nn.MSELoss()
        # TODO: add loss for points1_affine and points2, Euclidean distance

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=model_params.learning_rate)

    # Create empty list to store epoch number, train loss and validation loss
    epoch_loss_list = []

    # Create output directory
    output_dir = f"output/{model_params.get_model_code()}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Train model
    for epoch in range(model_params.start_epoch, model_params.num_epochs):
        running_loss = 0.0
        train_bar = tqdm(train_dataset, desc=f'Training Epoch {epoch+1}/{model_params.num_epochs}')
        for i, data in enumerate(train_bar):
            # Get images and affine parameters
            if model_params.sup:
                source_image, target_image, affine_params_true = data
            else:
                source_image, target_image = data
            source_image = source_image.to(device)
            target_image = target_image.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(source_image, target_image)
            # for i in range(len(outputs)):
            #         print(i, outputs[i].shape)
            # 0 torch.Size([1, 1, 256, 256])
            # 1 torch.Size([1, 2, 3])
            # 2 (2, 4)
            # 3 (2, 4)
            # 4 (1, 4, 2)
            # 5 (256, 9)
            # 6 (256, 16)
            # 7 (256, 256)
            # 8 (256, 256)
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

            loss = criterion(transformed_source_affine, target_image)
            if model_params.sup:
                loss_affine = criterion_affine(affine_params_true, affine_params_predicted)
                # TODO: add loss for points1_affine and points2, Euclidean distance
                # loss_points = criterion_points(points1_affine, points2)

                loss += loss_affine
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            train_bar.set_postfix({'loss': running_loss / (i+1)})
        print(f'Training Epoch {epoch+1}/{model_params.num_epochs} loss: {running_loss / len(train_dataset)}')

        # Validate model
        validation_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_dataset, 0):
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

                loss = criterion(transformed_source_affine, target_image)
                if model_params.sup:
                    loss_affine = criterion_affine(affine_params_true, affine_params_predicted)
                    # TODO: add loss for points1_affine and points2, Euclidean distance
                    # loss_points = criterion_points(points1_affine, points2)

                    loss += loss_affine

                # Add to validation loss
                validation_loss += loss.item()

                # Plot images if i < 5
                
                if i < 5:
                    output_file = f"{output_dir}/validation_epoch{epoch+1}.txt"
                    try: # need to do this because the warning cannot be suppressed, fix clipping makes the image all black
                        # Open the file in append mode to save the print statements
                        with open(output_file, 'a') as file:
                            # Redirect sys.stdout to the file
                            sys.stdout = file

                            # Redirect warnings to the file
                            # warnings.filterwarnings('always')  # Capture all warnings
                            warnings_file = open(output_file, 'a')
                            warnings.showwarning = lambda message, category, filename, lineno, file=warnings_file, line=None: \
                                file.write(warnings.formatwarning(message, category, filename, lineno, line))
                            
                            # Plot images
                            with capture_output():
                                DL_affine_plot(f"epoch{epoch+1}", output_dir,
                                    f"{i}", "_", source_image[0, 0, :, :].cpu().numpy(), target_image[0, 0, :, :].cpu().numpy(), \
                                    transformed_source_affine[0, 0, :, :].cpu().numpy(),
                                    points1, points2, points1_affine, desc1, desc2, \
                                    affine_params=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=True)
                                
                            # Reset sys.stdout to the console
                            sys.stdout = sys.__stdout__
                            warnings_file.close()
                    except Exception as e:
                        # Handle exceptions, if any
                        print(f"An error occurred: {e}")

        # Print validation statistics
        validation_loss /= len(test_dataset)
        print(f'Validation Epoch {epoch+1}/{model_params.num_epochs} loss: {validation_loss}')

        # Append epoch number, train loss and validation loss to epoch_loss_list
        epoch_loss_list.append([epoch+1, running_loss / len(train_dataset), validation_loss])

    print('Finished Training')

    # Extract epoch number, train loss and validation loss from epoch_loss_list
    epoch = [x[0] for x in epoch_loss_list]
    train_loss = [x[1] for x in epoch_loss_list]
    val_loss = [x[2] for x in epoch_loss_list]

    save_plot_name = f"{output_dir}/loss_{model_params.get_model_code()}_epoch{model_params.num_epochs}_{timestamp}.png"

    # Plot train loss and validation loss against epoch number
    plt.plot(epoch, train_loss, label='Train Loss')
    plt.plot(epoch, val_loss, label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()

    plt.savefig(save_plot_name)
    plt.show()

    # Return epoch_loss_list
    return epoch_loss_list


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
loss_list = train(model, model_params, timestamp)

print("Training output:")
for i in range(len(loss_list)):
    print(loss_list[i])

model_save_path = "trained_models/"
model_name_to_save = model_save_path + f"{model_params.get_model_code()}_{timestamp}.pth"
print(model_name_to_save)
torch.save(model.state_dict(), model_name_to_save)

