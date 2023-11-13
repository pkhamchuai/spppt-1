# import sys
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
# from torchvision import transforms
from torch import nn, optim
# import torch.nn.functional as F
# from torch.utils import data
# from torchsummary import summary
# from pytorch_model_summary import summary

from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, DL_affine_plot, print_summary
from utils.datagen import datagen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF

# Define training function
def train(model_name, model_params, timestamp):
    # Define loss function based on supervised or unsupervised learning
    criterion = model_params.loss_image
    extra = loss_extra()

    if model_params.sup:
        criterion_affine = nn.MSELoss()
        # TODO: add loss for points1_affine and points2, Euclidean distance

    if model_name == 'SP_AffineNet1':
        torch.manual_seed(9793047918980052389)
        from utils.SPaffineNet1 import SP_AffineNet1
        model = SP_AffineNet1(model_params).to(device)
    elif model_name == 'SP_AffineNet1_alt':
        torch.manual_seed(9793047918980052389)
        from utils.SPaffineNet1_alt import SP_AffineNet1_alt
        model = SP_AffineNet1_alt(model_params).to(device)
    elif model_name == 'SP_AffineNet2':
        from utils.SPaffineNet2 import SP_AffineNet2
        model = SP_AffineNet2(model_params).to(device)
    elif model_name == 'SP_AffineNet2_alt':
        from utils.SPaffineNet2_alt import SP_AffineNet2_alt
        model = SP_AffineNet2_alt(model_params).to(device)
    elif model_name == 'SP_AffineNet3':
        from utils.SPaffineNet3 import SP_AffineNet3
        model = SP_AffineNet3(model_params).to(device)
    elif model_name == 'SP_AffineNet4':
        from utils.SPaffineNet4 import SP_AffineNet4
        model = SP_AffineNet4(model_params).to(device)
    else:
        print('Input a valid model name')
        sys.exit()

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, model_params.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_params.decay_rate ** epoch)
    model_path = args.model_path

    # if a model is loaded, the training will continue from the epoch it was saved at
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        # print(model_path.split('/')[-1].split('_'))
        model_params.start_epoch = int(model_path.split('/')[-1].split('_')[3])
        print(f'Loaded model from {model_path}\nStarting at epoch {model_params.start_epoch}')
        if model_params.start_epoch >= model_params.num_epochs:
            model_params.num_epochs += model_params.start_epoch
    else:
        model_params.start_epoch = 0
        print('No model loaded, starting from scratch')

    # print case
    # print(model_params)
    model_params.print_explanation()

    # Create empty list to store epoch number, train loss and validation loss
    epoch_loss_list = []
    running_loss_list = []
    
    print('Seed:', torch.seed())
    
    # Create output directory
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}"
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
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Plot images if i < 5
            if i % 50 == 0:
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
        plt.plot(step, running_train_loss, label='Running Train Loss', alpha=0.3)
        plt.plot(epoch, train_loss, label='Train Loss', linewidth=3)
        plt.plot(epoch, val_loss, label='Validation Loss', linewidth=3)
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.yscale('log')
        plt.tight_layout()
        plt.savefig(save_plot_name)
        # plt.show()

    print('\nFinished Training')

    # delete all txt files in output_dir
    for file in os.listdir(output_dir):
        if file.endswith(".txt"):
            os.remove(os.path.join(output_dir, file))

    # Save model
    model_save_path = "trained_models/"
    model_name_to_save = model_save_path + f"{model_name}_{model_params.get_model_code()}_{timestamp}.pth"
    torch.save(model.state_dict(), model_name_to_save)
    print(f'Model saved in: {model_name_to_save}')

    # save the output of print_explanation() and loss_list to a txt file
    print_summary(model_name, model_name_to_save, 
                  model_params, epoch_loss_list, timestamp, False)

    # Return epoch_loss_list
    return model, epoch_loss_list


def test(model_name, model, model_params, timestamp):
    # Set model to training mode
    model.eval()

    # Create output directory
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}_test"
    os.makedirs(output_dir, exist_ok=True)

    # Validate model
    # validation_loss = 0.0

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

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

            if i < 10:
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

            # append metrics to metrics list
            metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, mse12_image, ssim12_image_before, ssim12_image])

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", "mse12_image_before", "mse12_image", "ssim12_image_before", "ssim12_image"])
        for i in range(len(metrics)):
            writer.writerow(metrics[i])
        # write the average and std of the metrics
        metrics = np.array(metrics)
        writer.writerow(["average", np.mean(metrics[:, 1]), np.mean(metrics[:, 2]), np.mean(metrics[:, 3]), np.mean(metrics[:, 4]), np.mean(metrics[:, 5]), np.mean(metrics[:, 6]), np.mean(metrics[:, 7]), np.mean(metrics[:, 8])])
        writer.writerow(["std", np.std(metrics[:, 1]), np.std(metrics[:, 2]), np.std(metrics[:, 3]), np.std(metrics[:, 4]), np.std(metrics[:, 5]), np.std(metrics[:, 6]), np.std(metrics[:, 7]), np.std(metrics[:, 8])])

    print(f"The test results are saved in {csv_file}")

    # delete all txt files in output_dir
    # for file in os.listdir(output_dir):
    #     if file.endswith(".txt"):
    #         os.remove(os.path.join(output_dir, file))

    print_summary(model_name, None, model_params, 
                  None, timestamp, True)


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
    parser.add_argument('--model', type=str, default=None, help='which model to use')
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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model, loss_list = train(args.model, model_params, timestamp)

    print("\nTesting the trained model +++++++++++++++++++++++")

    test(args.model, model, model_params, timestamp)
    
    print("Test model finished +++++++++++++++++++++++++++++")
