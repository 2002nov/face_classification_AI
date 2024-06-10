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

                now_batch_size = inputs.size(0)
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

    return model

if __name__ == '__main__':
    # Parse arguments
    opt = parse_args()

    # Set up data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(opt.data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(opt.data_dir, 'val'), data_transforms['val']),
    }

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=opt.batchsize, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=opt.batchsize, shuffle=False, num_workers=4),
    }

    # Dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device(f"cuda:{opt.gpu_ids}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Initialize model, criterion, optimizer, and scheduler
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train the model
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25, use_gpu=use_gpu)

