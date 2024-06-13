from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io
import math
from tqdm import tqdm
import shutil
from PIL import Image

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir', default='/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/Market', type=str, help='./test_data')
parser.add_argument('--name', default='best150', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--custom_model_path', default='/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/models/best150.pt', type=str, help='path to the custom model file')

opt = parser.parse_args()

# Set linear_num to 2048 as a default value for ResNet50
opt.linear_num = 2048

str_ids = opt.gpu_ids.split(',')
gpu_ids = [int(id) for id in str_ids if int(id) >= 0]

ms = [math.sqrt(float(s)) for s in opt.ms.split(',')]

# set gpu ids
if gpu_ids:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
h, w = 256, 128
data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning 0 as a placeholder for the label

data_dir = opt.test_dir

# Load the dataset directory from 'gt_bbox'
image_datasets = CustomImageDataset(os.path.join(data_dir, 'gt_bbox'), data_transforms)
dataloaders = DataLoader(image_datasets, batch_size=opt.batchsize, shuffle=False, num_workers=16)
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_custom_model(model_path, num_classes=751):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def fuse_all_conv_bn(model):
    # Dummy function, add real implementation if needed
    return model

######################################################################
# Extract feature
# ----------------------
def extract_feature(model, dataloaders):
    pbar = tqdm()
    if opt.linear_num <= 0:
        opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, _ = data
        n, c, h, w = img.size()
        pbar.update(n)
        ff = torch.FloatTensor(n, opt.linear_num).zero_().cuda()

        input_img = Variable(img.cuda())
        for scale in ms:
            if scale != 1:
                input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
            outputs = model(input_img)
            ff += outputs

        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if iter == 0:
            features = torch.FloatTensor(len(dataloaders.dataset), ff.shape[1])
        start = iter * opt.batchsize
        end = min((iter + 1) * opt.batchsize, len(dataloaders.dataset))
        features[start:end, :] = ff
    pbar.close()
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1] if 'c' in filename else '0'
        labels.append(int(label) if label[0:2] != '-1' else -1)
        camera_id.append(int(camera[0]))
    return camera_id, labels

query_path = image_datasets.image_paths

query_cam, query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
model = load_custom_model(opt.custom_model_path)

# Remove the final fc layer and classifier layer
model.fc = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    query_feature = extract_feature(model, dataloaders)

# Save to Matlab for check
result = {'query_f': query_feature.cpu().numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('pytorch_result.mat', result)

# Re-ID and save images to respective folders
def reid_and_save_images(query_feature, query_path):
    dist = torch.cdist(query_feature, query_feature)
    indices = torch.argmin(dist, dim=1)

    output_dir = '/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/Market/reid_output3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, match_idx in enumerate(indices):
        query_img_path = query_path[idx]
        matched_img_path = query_path[match_idx]

        person_id = os.path.basename(matched_img_path).split('_')[0]
        person_dir = os.path.join(output_dir, person_id)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        # Print details for debugging
        #print(f'Copying {query_img_path} to {person_dir}')

        shutil.copy(query_img_path, os.path.join(person_dir, os.path.basename(query_img_path)))

reid_and_save_images(query_feature, query_path)

