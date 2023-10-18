import argparse
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.measure import ransac
# from skimage.transform import FundamentalMatrixTransform, AffineTransform

import csv
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn, optim

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
def train(model, model_params, timestamp):
    # Define loss function based on supervised or unsupervised learning
    criterion = model_params.loss_image
    extra = loss_extra()

    if model_params.sup:
        criterion_affine = nn.MSELoss()
        # TODO: add loss for points1_affine and points2, Euclidean distance

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=model_params.learning_rate)

    # Create empty list to store epoch number, train loss and validation loss
    epoch_loss_list = []
    running_loss_list = []
    
    # Create output directory
    output_dir = f"output/{model_params.get_model_code()}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    save_plot_name = f"{output_dir}/loss_{model_params.get_model_code()}_epoch{model_params.num_epochs}_{timestamp}.png"

    # Train model
    for epoch in range(model_params.start_epoch, model_params.num_epochs):
        # Set model to training mode
        model.train()
        
        running_loss = 0.0
        train_bar = tqdm(train_dataset, desc=f'Training Epoch {epoch+1}/{model_params.num_epochs}')
        for i, data in enumerate(train_bar):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Get images and affine parameters
            if model_params.sup:
                source_image, target_image, affine_params_true = data
            else:
                source_image, target_image = data
                affine_params_true = None
            source_image = source_image.to(device)
            target_image = target_image.to(device)

            # Forward + backward + optimize
            outputs = model(source_image, target_image)
            # for i in range(len(outputs)):
            #         print(i, outputs[i].shape)
            # 0 torch.Size([1, 1, 256, 256])
            # 1 torch.Size([1, 2, 3])
            # 2 (2, 0)
            # 3 (2, 0)
            # 4 (256, 100)
            # 5 (256, 115)
            # 6 (256, 256)
            # 7 (256, 256)
            transformed_source_affine = outputs[0] # image
            affine_params_predicted = outputs[1] # affine parameters
            points1 = np.array(outputs[2])
            points2 = np.array(outputs[3])
            points1_transformed = np.array(outputs[4])

            # print(f"affine_params_true: {affine_params_true}")
            # print(f"affine_params_predicted: {affine_params_predicted}\n")

            try:
                points1_affine = points1_affine.reshape(
                    points1_transformed.shape[2], points1_transformed.shape[1])
            except:
                pass
            desc1_2 = outputs[5]
            desc2 = outputs[6]
            heatmap1 = outputs[7]
            heatmap2 = outputs[8]

            loss = criterion(transformed_source_affine, target_image)
            loss += extra(affine_params_predicted)
            if model_params.sup:
                loss_affine = criterion_affine(affine_params_true.view(1, 2, 3), affine_params_predicted.cpu())
                # TODO: add loss for points1_affine and points2, Euclidean distance
                # loss_points = criterion_points(points1_affine, points2)

                loss += loss_affine
            # loss.backward()
            optimizer.step()
            

            # Plot images if i < 5
            if i % 100 == 0:
                DL_affine_plot(f"epoch{epoch+1}_train", output_dir,
                    f"{i}", "_", source_image[0, 0, :, :].detach().cpu().numpy(), target_image[0, 0, :, :].detach().cpu().numpy(), 
                    transformed_source_affine[0, 0, :, :].detach().cpu().numpy(),
                    points1, points2, points1_transformed, desc1_2, desc2, affine_params_true=affine_params_true,
                        affine_params_predict=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=True)

            # Print statistics
            running_loss += loss.item()
            running_loss_list.append([epoch+((i+1)/len(train_dataset)), loss.item()])
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
                    affine_params_true = None
                source_image = source_image.to(device)
                target_image = target_image.to(device)

                # Forward pass
                outputs = model(source_image, target_image)
                # for i in range(len(outputs)):
                #     print(i, outputs[i].shape)
                transformed_source_affine = outputs[0]
                affine_params_predicted = outputs[1]
                points1 = np.array(outputs[2])
                points2 = np.array(outputs[3])
                points1_transformed = np.array(outputs[4])

                try:
                    points1_affine = points1_transformed.reshape(points1_transformed.shape[2], points1_transformed.shape[1])
                except:
                    pass

                desc1_2 = outputs[5]
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
                if i % 50 == 0:
                    DL_affine_plot(f"epoch{epoch+1}_valid", output_dir,
                        f"{i}", "_", source_image[0, 0, :, :].cpu().numpy(), target_image[0, 0, :, :].cpu().numpy(), transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        points1, points2, points1_transformed, desc1_2, desc2, affine_params_true=affine_params_true,
                            affine_params_predict=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=True)

        # Print validation statistics
        validation_loss /= len(test_dataset)
        print(f'Validation Epoch {epoch+1}/{model_params.num_epochs} loss: {validation_loss}')

        # Append epoch number, train loss and validation loss to epoch_loss_list
        epoch_loss_list.append([epoch+1, running_loss / len(train_dataset), validation_loss])

        # Extract epoch number, train loss and validation loss from epoch_loss_list
        epoch = [x[0] for x in epoch_loss_list]
        train_loss = [x[1] for x in epoch_loss_list]
        val_loss = [x[2] for x in epoch_loss_list]
        step = [x[0] for x in running_loss_list]
        running_train_loss = [x[1] for x in running_loss_list]

        # Plot train loss and validation loss against epoch number
        plt.figure()
        plt.plot(step, running_train_loss, label='Running Train Loss', alpha=0.5)
        plt.plot(epoch, train_loss, label='Train Loss', linewidth=3)
        plt.plot(epoch, val_loss, label='Validation Loss', linewidth=3)
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()

        plt.savefig(save_plot_name)
        # plt.show()

    print('Finished Training')

    # delete all txt files in output_dir
    for file in os.listdir(output_dir):
        if file.endswith(".txt"):
            os.remove(os.path.join(output_dir, file))

    # Return epoch_loss_list
    return epoch_loss_list


def test(model, model_params, timestamp):
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
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", "mse12_image_before", "mse12_image", "ssim12_image_before", "ssim12_image"])

    with torch.no_grad():
        testbar = tqdm(test_dataset, desc=f'Testing:')
        for i, data in enumerate(testbar, 0):
            # Get images and affine parameters
            if model_params.sup:
                source_image, target_image, affine_params_true = data
            else:
                source_image, target_image = data
                affine_params_true = None
            source_image = source_image.to(device)
            target_image = target_image.to(device)

            # Forward pass
            outputs = model(source_image, target_image)
            # for i in range(len(outputs)):
            #     print(i, outputs[i].shape)
            transformed_source_affine = outputs[0]
            affine_params_predicted = outputs[1]
            points1 = np.array(outputs[2])
            points2 = np.array(outputs[3])
            points1_transformed = np.array(outputs[4])

            try:
                points1_affine = points1_transformed.reshape(points1_transformed.shape[2], points1_transformed.shape[1])
            except:
                pass

            desc1_2 = outputs[5]
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
                points1, points2, points1_transformed, desc1_2, desc2, affine_params_true=affine_params_true,
                affine_params_predict=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=plot_)


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


if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='image used for training')
    parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=2, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    args = parser.parse_args()


    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, heatmaps=args.heatmaps, 
                               loss_image=args.loss_image, num_epochs=args.num_epochs, 
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    model_params.print_explanation()
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
        # print(model_path.split('/')[-1].split('_')[3])
        model_params.start_epoch = int(model_path.split('/')[-1].split('_')[3])
        print(f'Loaded model from {model_path}\nStarting at epoch {model_params.start_epoch}')
        if model_params.start_epoch >= model_params.num_epochs:
            model_params.num_epochs += model_params.start_epoch
    else:
        model_params.start_epoch = 0
        print('No model loaded, starting from scratch')

    # print case
    print(model_params)
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_list = train(model, model_params, timestamp)

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

    metrics = test(model, model_params, timestamp)
    print(metrics)
    print("Test model finished +++++++++++++++++++++++++++++")
