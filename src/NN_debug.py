import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
from skimage import io, transform
import pandas as pd
import time
import copy
from torchvision.models import resnet18


class MeltpoolDataset(Dataset):
    """Dataset for Meltpool Images and Process Parameters"""

    def __init__(self, xlsx_file, root_dir, transform=None):
        """
        Args:
            xlsx_file (string): file with process parameters and labels
            root_dir (string): image directory
            transform (callable, optional): transform(s) to apply
        """

        print('Loading Process Parameters...')
        data_frame = pd.read_excel(xlsx_file, sheet_name='Sheet1', engine='openpyxl')
        self.images = np.array(data_frame['image_name'])
        self.labels = np.array(data_frame['label'])
        self.process_parameters = np.array(data_frame[data_frame.columns[2:]])

        print('Updating Image File Names...')
        for ii in range(self.images.shape[0]):
            layer = self.images[ii][0:self.images[ii].find('_')]
            self.images[ii] = layer + '/' + self.images[ii]

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = io.imread(img_name)
        pp = self.process_parameters[idx, :]
        pp = pp.astype('float')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'process_parameters': pp, 'label': label}

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, process_parameters = sample['image'], sample['process_parameters']
        label = sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img, 'process_parameters': process_parameters, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, process_parameters = sample['image'], sample['process_parameters']
        label = sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'process_parameters': process_parameters, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, process_parameters = sample['image'], sample['process_parameters']
        label = sample['label']
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, dim=-1)  # Add a dimension to end to indicate a single channel
        image = image.permute((2, 0, 1))

        pp = torch.from_numpy(process_parameters)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #         image = image.transpose((2, 0, 1))
        return {'image': image, 'process_parameters': pp, 'label': label}


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # The base directory to images
    DATA_DIR = '../../../In-situ Meas Data/In-situ Meas Data/Melt Pool Camera Preprocessed PNG/'

    dataset = MeltpoolDataset('neural_network_data/test_labels_pp.xlsx', DATA_DIR)
    print(dataset.__len__())