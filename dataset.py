import os
import scipy.io
import numpy as np

def load_cuhk03_dataset(dataset_path):
    labeled_path = os.path.join(dataset_path, 'labeled')
    detected_path = os.path.join(dataset_path, 'detected')

    if not os.path.exists(labeled_path) or not os.path.exists(detected_path):
        raise FileNotFoundError('CUHK03 dataset folders not found in the specified path.')

    labeled_files = sorted([os.path.join(labeled_path, f) for f in os.listdir(labeled_path) if f.endswith('.mat')])
    detected_files = sorted([os.path.join(detected_path, f) for f in os.listdir(detected_path) if f.endswith('.mat')])

    print(f'Found {len(labeled_files)} labeled files and {len(detected_files)} detected files.')

    labeled_data = [scipy.io.loadmat(f) for f in labeled_files]
    detected_data = [scipy.io.loadmat(f) for f in detected_files]

    return labeled_data, detected_data

def main(dataset_path):
    labeled_data, detected_data = load_cuhk03_dataset(dataset_path)
    print('Labeled data and Detected
