import os
import torch
import yaml
from torch import nn, optim
from shutil import copyfile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models  # Added models import
from tqdm import tqdm
import argparse

# Define a function for argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Training script for ResNet50')
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs to use')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='Name of the training session')
    parser.add_argument('--train_all', action='store_true', help='Use all training data')
    parser.add_argument('--batchsize', default=32, type=int, help='Batch size')
    parser.add_argument('--data_dir', default='data', type=str, help='Path to the data directory')
    parser.add_argument('--erasing_p', default=0.5, type=float, help='Probability of random erasing')
    return parser.parse_args()

# Training function
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, use_gpu=False):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # Save the model after each epoch
        model_save_path = os.path.join('models', f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

    return model

# Add your training loop here
