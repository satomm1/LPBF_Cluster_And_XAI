import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
import pandas as pd
import copy
import cv2
from PIL import Image as im
from torchvision.models import resnet18
import openpyxl
from matplotlib.legend_handler import HandlerTuple
from tqdm import tqdm

class MeltPoolNetwork(nn.Module):
    """Neural Network for Melt Pool Shape Prediction"""

    def __init__(self, imageModel, num_classes=10, num_param=10):
        """
        Args:
            imageModel (A pytorch model): the CNN to use for melt pool image encoding
            num_classes (int): Number of different melt pool classes to predict
            num_param (int): Number of process parameters available
        """

        super().__init__()
        # The image encoder CNN
        self.ImageModel = imageModel

        # The process parameter encoder layers
        self.paramLayer1 = nn.Sequential(nn.Linear(num_param, 10), nn.Tanh())
        self.paramLayer2 = nn.Sequential(nn.Linear(10, 10), nn.Tanh())
        self.paramLayer3 = nn.Sequential(nn.Linear(10, 10), nn.Tanh())
        self.paramLayer4 = nn.Sequential(nn.Linear(10, 10), nn.Tanh())

        # prediction head layers
        self.prediction1 = nn.Sequential(nn.Linear(512 + 10, 100), nn.Tanh())
        self.prediction2 = nn.Linear(100, num_classes)

        # Initialize Model Weights
        tanh_gain = torch.nn.init.calculate_gain('tanh', param=None)
        torch.nn.init.xavier_normal_(self.paramLayer1[0].weight, gain=tanh_gain)
        torch.nn.init.xavier_normal_(self.paramLayer2[0].weight, gain=tanh_gain)
        torch.nn.init.xavier_normal_(self.paramLayer3[0].weight, gain=tanh_gain)
        torch.nn.init.xavier_normal_(self.paramLayer4[0].weight, gain=tanh_gain)
        torch.nn.init.xavier_normal_(self.prediction1[0].weight, gain=tanh_gain)
        torch.nn.init.kaiming_normal_(self.prediction2.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, img, pp):
        """
        Args:
            img (tensor): The melt pool image
            pp  (tensor): The process parameters
        """

        # Image CNN
        x = self.ImageModel(img)

        # PP NN
        y = self.paramLayer1(pp)
        y = self.paramLayer2(y)
        y = self.paramLayer3(y)
        y = self.paramLayer4(y)
        y = y.view(y.size(0), -1)

        # Prediction Head
        # y = torch.squeeze(y)  # remove any dimensions of 1
        z = torch.cat((x, y), dim=1)
        z = self.prediction1(z)
        z = self.prediction2(z)
        return z


class MeltpoolDataset(Dataset):
    """Dataset for Meltpool Images and Process Parameters"""

    def __init__(self, xlsx_file, root_dir, transform=None):
        """
        Args:
            xlsx_file (string): file with process parameters and labels
            root_dir (string): image directory
            transform (callable, optional): transform(s) to apply
        """

        print('************** Loading Data **************')
        print(xlsx_file)

        # Load the excel file and separate into image file names, labels, and process parameters
        if xlsx_file.find('xlsx') >= 0:
            data_frame = pd.read_excel(xlsx_file, sheet_name='Sheet1', engine='openpyxl')
        elif xlsx_file.find('csv') >= 0:
            data_frame = pd.read_csv(xlsx_file)
        self.images = np.array(data_frame['image_name'])
        self.labels = np.array(data_frame['label'])
        self.process_parameters = np.array(data_frame[data_frame.columns[2:]])

        # We need to modify the image file names
        for ii in range(self.images.shape[0]):
            layer = self.images[ii][0:self.images[ii].find('_')]
            self.images[ii] = layer + '/' + self.images[ii]

        # Store some important information
        self.root_dir = root_dir
        self.transform = transform
        self.PIL_transform = transforms.ToPILImage()
        print('************ Finished Loading ************')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image and convert to a PIL image
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = io.imread(img_name)
        image = self.PIL_transform(image).convert('RGB')

        # Apply transforms to the image
        if self.transform:
            image = self.transform(image)

        # Load the process parameters
        pp = self.process_parameters[idx, :]
        pp = pp.astype('float')

        # Load the label
        label = self.labels[idx]

        return {'image': image, 'process_parameters': pp, 'label': label}

class LRP_Combined(nn.Module):

    def __init__(self, model, eps=1.0e-9, gamma=0.1):
        super().__init__()
        self.model = model
        self.model.eval()

        self.eps = eps
        self.gamma = gamma
        self.layers = self.get_layers()

    def get_layers(self):

        # Builds list of all layers in the neural network
        # Works specifically for this CNN
        img_model_layers = nn.ModuleList()
        for module in self.model.ImageModel.children():
            if isinstance(module, nn.Sequential):
                for module2 in module.children():
                    # for module3 in module2.children():
                    #     if isinstance(module3, nn.Sequential):
                    #         for module4 in module3.children():
                    #             layers.append(module4)
                    #     else:
                    #         layers.append(module3)
                    img_model_layers.append(module2)
            else:
                img_model_layers.append(module)
        #         print(layers)

        param_layers = nn.ModuleList()
        final_layers = nn.ModuleList()
        for module in self.model.children():
            if not isinstance(module, torchvision.models.resnet.ResNet):
                if isinstance(module, nn.Sequential):
                    if module[0].in_features < 512:
                        for module2 in module.children():
                            param_layers.append(module2)
                    else:
                        for module2 in module.children():
                            final_layers.append(module2)
                else:
                    final_layers.append(module)

        layers = {"image": img_model_layers, "pp": param_layers, "final": final_layers}
        return layers

    def evaluate(self, sample):
        img = sample['image']
        x = img.to(device=device, dtype=torch.float)
        pp = sample['process_parameters']
        y = pp.to(device=device, dtype=torch.float)

        img_act = []
        pp_act = []
        final_act = []

        with torch.no_grad():
            img_act.append(torch.ones_like(x))
            for layer in self.layers["image"]:
                #                 print(layer)
                if isinstance(layer, nn.Linear):
                    x = x.squeeze(dim=2)
                    x = x.squeeze(dim=2)
                x = layer(x)
                # print(x.shape)
                img_act.append(x)

            pp_act.append(torch.ones_like(pp))
            for layer in self.layers["pp"]:
                y = layer(y)
                pp_act.append(y)

            # y = y.view(y.size(0), -1)
            x = torch.squeeze(x)
            y = torch.squeeze(y)
            z = torch.cat((x, y), dim=-1)
            for layer in self.layers["final"]:
                z = layer(z)
                final_act.append(z)

        img_act = img_act[::-1]  # reverse order
        img_act = [a.requires_grad_(True) for a in img_act]

        pp_act = pp_act[::-1]
        pp_act = [a.requires_grad_(True) for a in pp_act]

        final_act = final_act[::-1]
        final_act = [a.requires_grad_(True) for a in final_act]

        R = torch.softmax(final_act.pop(0), dim=-1)
        R_final = R

        R_final_list = []
        R_final_list.append(R)

        img_layers = self.layers['image']
        pp_layers = self.layers['pp']
        final_layers = self.layers['final']

        reverse_img_layers = img_layers[::-1]
        reverse_pp_layers = pp_layers[::-1]
        reverse_final_layers = final_layers[::-1]

        for layer in reverse_final_layers[:-1]:
            R_final = self.lrp_eval(layer, final_act.pop(0), R_final)
            R_final_list.append(R_final)

        act1 = img_act.pop(0)
        # act1 = torch.squeeze(act1, dim=0)
        act2 = pp_act.pop(0)
        # act2 = torch.squeeze(act2, dim=0)
        act = torch.cat((act1, act2), dim=1)

        R = self.lrp_eval(final_layers[0], act, R_final)
        R_final_list.append(R)

        R_img = R[0:512]
        R_img_list = []
        for layer in reverse_img_layers:
            R_img = self.lrp_eval(layer, img_act.pop(0), R_img)
            R_img_list.append(R_img)

        R_pp = R[512:]
        R_pp_list = []
        for layer in reverse_pp_layers:
            R_pp = self.lrp_eval(layer, pp_act.pop(0).to(device=device, dtype=torch.float), R_pp)
            R_pp_list.append(R_pp)

        return {"image": R_img_list, "pp": R_pp_list, "final": R_final_list}

    def lrp_eval(self, layer, a, R):
        if isinstance(layer, nn.Linear):
            a = a.squeeze()

        if isinstance(layer, nn.ReLU):
            return R

        a = a.data.requires_grad_(True)
        z = self.eps + layer.forward(a)
        s = (R / (z + 1e-9)).data  # 1e-9 to prevent divide by 0
        (z * s).sum().backward()
        c = a.grad
        R = a * c
        return R


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    BATCH_SIZE = 1  # Minibatch size to use
    NUM_MELT_POOL_CLASSES = 24  # Number of different melt pool shape classes
    NUM_PROCESS_PARAM = 9  # Number of process parameters
    EPS = 1

    # The base directory to images
    DATA_DIR = '../../../In-situ Meas Data/In-situ Meas Data/Melt Pool Camera Preprocessed PNG/'
    # DATA_DIR = '../../Melt Pool Camera Preprocessed PNG/'

    MODEL_NAME = 'testV4'  # Name to save trained model

    meltpool_dataset_test = MeltpoolDataset(
        'neural_network_data/test_labels_pp.xlsx',
        DATA_DIR,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    dataloader_test = DataLoader(meltpool_dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    MODEL_PATH = 'trained_models/' + MODEL_NAME + '.pth'

    torch.cuda.empty_cache()

    # Load neural network
    ImgModel = resnet18()
    ImgModel.fc = nn.Linear(512, 512)
    ImgModel.to(device)
    model = MeltPoolNetwork(ImgModel, num_classes=NUM_MELT_POOL_CLASSES, num_param=NUM_PROCESS_PARAM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    lrp_eps = LRP_Combined(model, eps=EPS)
    print(lrp_eps)

    sample = next(iter(dataloader_test))

    R_dict = lrp_eps.evaluate(sample)