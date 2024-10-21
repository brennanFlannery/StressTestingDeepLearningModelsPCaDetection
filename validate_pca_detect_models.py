import os.path
import pandas as pd
import shutil
import sklearn.metrics
from torchsummary import summary
import torchvision.models
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet
from albumentations import *
from albumentations.pytorch import ToTensorV2
import PIL
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys, glob
import time
import math
import tables
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import scipy.ndimage
import torchstain
import torchvision.transforms as trans
from tqdm import tqdm

model_tags = ["densenet", "resnet", "resnext", "efficient_net"]
data = ["pca_detect_dense_bpF", "pca_detect_dense_rpF"]


class hist_data(object):
    def __init__(self, fname, img_transform=None):
        # nothing special here, just internalizing the constructor parameters
        self.fname = fname
        self.img_transform = img_transform

        with tables.open_file(self.fname, 'r') as db:
            self.nitems = db.root.imgs.shape[0]

        self.imgs = None

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes

        with tables.open_file(self.fname, 'r') as db:
            self.imgs = db.root.imgs

            # get the requested image and mask from the pytable
            img = self.imgs[index, :, :, :]

        return img

    def __len__(self):
        return self.nitems

class HistMatch(ImageOnlyTransform):
    def __init__(self,always_apply=True,p=1.0):
        super(HistMatch, self).__init__(always_apply, p)

    def apply(self, image, **params):
        target = histmatch[random.randint(0, len(histmatch) - 1)]
        i = image.copy()
        T = trans.Compose([trans.ToTensor(), trans.Lambda(lambda x: x*255)])

        # try:
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(target))

        image = T(image)
        try:
            t, H, E = torch_normalizer.normalize(I=image, stains=True)
        except:
            t = image.permute(1,2,0)
        t = t.numpy().astype(np.uint8)
        # except:
        #     t = i

        return t

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")

class Dataset(object):
    def __init__(self, fname, img_transform=None):
        # nothing special here, just internalizing the constructor parameters
        self.fname = fname
        # print(fname)
        self.img_transform = img_transform

        with tables.open_file(self.fname, 'r') as db:
            self.classsizes = db.root.classsizes[:]
            self.nitems = db.root.imgs.shape[0]

        self.imgs = None
        self.labels = None

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes
        # index = np.random.randint(0,100)
        with tables.open_file(self.fname, 'r') as db:
            self.imgs = db.root.imgs
            self.labels = db.root.labels

            # get the requested image and mask from the pytable
            img = self.imgs[index, :, :, :]
            label = self.labels[index]

        img_new = img

        if self.img_transform:
            img_new = self.img_transform(image=img)['image']

        return img_new, label, img

    def __len__(self):
        return self.nitems


img_transform_val = Compose([
    ToTensorV2()
])
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

model_dict = {"densenet": ("", DenseNet(growth_rate=32, block_config=(2, 2, 2, 2), num_init_features=64, bn_size=4,
                     drop_rate=0,num_classes=2).to(device)),
              "resnet": ("__rn", torchvision.models.resnet18(num_classes=2).to(device)),
              "resnext": ("__rnx", torchvision.models.ResNet(torchvision.models.resnet.Bottleneck, [1, 1, 1, 1], num_classes=2).to(device)),
              "efficient_net": ("__en", torchvision.models.efficientnet_b0().to(device))}

results = {"Metric": ["Sensitivity", "Specificity", "F1 Score"]}
for data_source in data:
    dataset = Dataset(f"./{data_source}_val.pytable", img_transform=img_transform_val)
    dataLoader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    for modeltype in model_tags:
        tag, model = model_dict[modeltype]
        if modeltype == "efficient_net":
            model.classifier[1] = nn.Linear(in_features=1280, out_features=2).to(device)
        for model_start in ["pca_detect_dense_bpF", "pca_detect_dense_rpF"]:
            checkpoint = torch.load(f"{model_start}_densenet_best_model{tag}.pth")
            model.load_state_dict(checkpoint["model_dict"])
            pred = []
            gt = []
            for ii, (X, label, img_orig) in enumerate(tqdm(dataLoader, desc=f"data_{data_source[-3:-1]}-{modeltype}-{model_start[-3:-1]}")):
                X = X.type('torch.FloatTensor').to(device)
                label = label.type('torch.LongTensor').to(
                    device)
                prediction = model(X)
                p = prediction.detach().cpu().numpy()
                cpredflat = np.argmax(p, axis=1).flatten().tolist()
                yflat = label.cpu().numpy().flatten().tolist()

                pred.extend(cpredflat)
                gt.extend(yflat)

            # Calculate Sen, Spe, F1 score
            cm = sklearn.metrics.confusion_matrix(gt, pred)
            se = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            sp = cm[1,1]/(cm[1,0]+cm[1,1])
            f1 = sklearn.metrics.f1_score(gt, pred)

            results[f"data_{data_source[-3:-1]}-{modeltype}-{model_start[-3:-1]}"] = [se, sp, f1]

results_df = pd.DataFrame(results)
results_df.to_excel("PathModeltypesValidation.xlsx")