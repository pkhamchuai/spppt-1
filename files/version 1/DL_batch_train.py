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
from utils.SPaffineNet import SP_AffineNet
from utils.datagen import datagen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

# Define training function
def train(model, model_params, train_dataset, test_dataset, timestamp, model_path=None):

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, model_params.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_params.decay_rate ** epoch)
    # model_path = args.model_path

    # if a model is loaded, the training will continue from the epoch it was saved at
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        model_params.start_epoch = int(model_path.split('/')[-1].split('_')[2])
        print(f'Loaded model from {model_path}\nStarting at epoch {model_params.start_epoch}')
    else:
        model_params.start_epoch = 0
        print('No model loaded, starting from scratch')

    # print case
    print(model_params)
    model_params.print_explanation()

    # Set model to training mode
    model.train()

    # Define loss function based on supervised or unsupervised learning
    criterion = model_params.loss_image
    # extra = loss_extra()

    if model_params.sup:
        criterion_affine = nn.MSELoss()
        # TODO: add loss for points1_affine and points2, Euclidean distance

    # Create empty list to store epoch number, train loss and validation loss
    epoch_loss_list = []
    
    # Create output directory
    output_dir = f"output/{model_params.get_model_code()}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Train model
    for epoch in range(model_params.start_epoch, model_params.num_epochs):
        running_loss = 0.0
        # Zero the parameter gradients
        optimizer.zero_grad()
        train_bar = tqdm(train_dataset, desc=f'Training Epoch {epoch+1}/{model_params.num_epochs}')
        for i, data in enumerate(train_bar):
            # Get images and affine parameters
            if model_params.sup:
                source_image, target_image, affine_params_true = data
            else:
                source_image, target_image = data
            source_image = source_image.to(device)
            target_image = target_image.to(device)

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
            # loss += extra(affine_params_predicted)
            if model_params.sup:
                loss_affine = criterion_affine(affine_params_true.view(1, 2, 3), affine_params_predicted.cpu())
                # TODO: add loss for points1_affine and points2, Euclidean distance
                # loss_points = criterion_points(points1_affine, points2)

                loss += loss_affine
            loss.backward() # should it be here?
            optimizer.step()

            # Plot images if i < 5
            if i < 5:
                DL_affine_plot(f"epoch{epoch+1}_train", output_dir,
                    f"{i}", "_", source_image[0, 0, :, :].detach().cpu().numpy(), target_image[0, 0, :, :].detach().cpu().numpy(), 
                    transformed_source_affine[0, 0, :, :].detach().cpu().numpy(),
                    points1, points2, points1_affine, desc1, desc2, \
                        affine_params=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=True)

            
            # Print statistics
            running_loss += loss.item()
            train_bar.set_postfix({'loss': running_loss / (i+1)})
        print(f'Training Epoch {epoch+1}/{model_params.num_epochs} loss: {running_loss / len(train_dataset)}')
        
        scheduler.step()

        # Validate model
        validation_loss = 0.0
        model.eval()
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
                # loss += extra(affine_params_predicted)
                if model_params.sup:
                    loss_affine = criterion_affine(affine_params_true.view(1, 2, 3), affine_params_predicted.cpu())
                    # TODO: add loss for points1_affine and points2, Euclidean distance
                    # loss_points = criterion_points(points1_affine, points2)
                    loss += loss_affine

                # Add to validation loss
                validation_loss += loss.item()

                # Plot images if i < 5
                if i < 5:
                    DL_affine_plot(f"epoch{epoch+1}_valid", output_dir,
                        f"{i}", "_", source_image[0, 0, :, :].cpu().numpy(), target_image[0, 0, :, :].cpu().numpy(), transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        points1, points2, points1_affine, desc1, desc2, \
                            affine_params=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=True)

        # Print validation statistics
        validation_loss /= len(test_dataset)
        print(f'Validation Epoch {epoch+1}/{model_params.num_epochs} loss: {validation_loss}')

        # Append epoch number, train loss and validation loss to epoch_loss_list
        epoch_loss_list.append([epoch+1, running_loss / len(train_dataset), validation_loss])

    print('Finished Training')

    # delete all txt files in output_dir
    for file in os.listdir(output_dir):
        if file.endswith(".txt"):
            os.remove(os.path.join(output_dir, file))

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
    # plt.show()

    # Return epoch_loss_list
    return epoch_loss_list


def test(model, model_params, test_dataset, timestamp):
    # Set model to training mode
    model.eval()

    # Create output directory
    output_dir = f"output/{model_params.get_model_code()}_{timestamp}_test"
    os.makedirs(output_dir, exist_ok=True)

    # Validate model
    # validation_loss = 0.0

    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # matches1_transformed.shape[-1], mse_before, mse12, tre_before, tre12, \
        # mse12_image, ssim12_image, 
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", 
                         "mse12_image_before", "mse12_image", "ssim12_image_before", "ssim12_image"])

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
                writer.writerow([i, mse_before, mse12, tre_before, tre12, mse12_image_before, mse12_image, \
                                 ssim12_image_before, ssim12_image])

    # delete all txt files in output_dir
    for file in os.listdir(output_dir):
        if file.endswith(".txt"):
            os.remove(os.path.join(output_dir, file))

def batch_training(model_params):
    train_dataset = datagen(model_params.dataset, True, model_params.sup)
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Get sample batch
    print('Train set: ', [x.shape for x in next(iter(train_dataset))])
    print('Test set: ', [x.shape for x in next(iter(test_dataset))])

    model = SP_AffineNet(model_params).to(device)
    print(model)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_list = train(model, model_params, train_dataset, test_dataset, timestamp)

    print("Training output:")
    for i in range(len(loss_list)):
        print(loss_list[i])

    model_save_path = "trained_models/"
    model_name_to_save = model_save_path + f"{model_params.get_model_code()}_{timestamp}.pth"
    print(model_name_to_save)
    torch.save(model.state_dict(), model_name_to_save)

    print("Test model +++++++++++++++++++++++++++++")

    # model = SPmodel = SP_AffineNet().to(device)
    # print(model)

    # parameters = model.parameters()
    # optimizer = optim.Adam(parameters, model_params.learning_rate)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_params.decay_rate ** epoch)

    # model.load_state_dict(torch.load(model_name_to_save))

    test(model, model_params, test_dataset, timestamp)
    print("Test model finished +++++++++++++++++++++++++++++")


if __name__ == '__main__':
    # test datagen for all datasets and training and testing
    count = 0
    for dataset in range(2): # let's do 2 datasets first
        for loss in range(5):
            for sup in [0, 1]:
                print(f'dataset: {dataset}, sup: {sup}, loss: {loss}')
                
                if sup==1 and dataset==0:
                    print('skipping')
                    pass
                else:
                    try:
                        model_params = ModelParams(dataset=dataset, sup=sup, image=1, 
                            num_epochs=1, learning_rate=1e-4, decay_rate=0.9, heatmaps=0, loss_image=loss)
                        batch_training(model_params)
                        count += 1
                    except ValueError:
                        print('ValueError')
                        continue

                print('\n')

    print(f'Finish training: {count} models')
