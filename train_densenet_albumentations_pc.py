# v3.classification
# 17/8/2019
# modified augmentation approach to use albumentations:
# https://github.com/albu/albumentations
# https://albumentations.readthedocs.io/
#
import os.path
import pandas as pd
import shutil

import sklearn.metrics
from torchsummary import summary
import torchvision.models
from sklearn.metrics import roc_auc_score
code_mode = "stain_norm_examples"
dataname = "pca_detect_dense_rpF"
model_tag = "_rn"
gpuid = 0

# %%

# %%
# --- densenet params
# these parameters get fed directly into the densenet class, and more description of them can be discovered there
num_classes = 2  # number of classes in the data mask that we'll aim to predict
in_channels = 3  # input channel of the data, RGB = 3

growth_rate = 32
block_config = (2, 2, 2, 2)
num_init_features = 64
bn_size = 4
drop_rate = 0

# --- training params
batch_size = 16
patch_size = 224  # currently, this needs to be 224 due to densenet architecture
num_epochs = 500
phases = ["train", "val"]  # how many phases did we create databases for?
validation_phases = [
    "val"]  # when should we do valiation? note that validation is *very* time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
# additionally, using simply [], will skip validation entirely, drastically speeding things up
# + {}
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

# -

# helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent + .00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# specify if we should use a GPU (cuda) or only the CPU
print(torch.cuda.get_device_properties(gpuid))
torch.cuda.set_device(gpuid)
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

# this defines our dataset class which will be used by the dataloader
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


# Load Histogram matching pytable
# hm_file = "Z:/home/bxf169/BladderToProstateConversion/TumorSegmentationTraining/rp_bp_hist_norm.pytable"
hm_file = "./rp_bp_hist_norm.pytable"
histmatch = hist_data(hm_file, img_transform=None)


# %%
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


# +
# build the model according to the paramters specified above and copy it to the GPU. finally print out the number of trainable parameters

# model = DenseNet(growth_rate=growth_rate, block_config=block_config,
#                  num_init_features=num_init_features,
#                  bn_size=bn_size,
#                  drop_rate=drop_rate,
#                  num_classes=num_classes).to(device)

# model = torchvision.models.swin_t()

model = torchvision.models.efficientnet_b0().to(device)
model.classifier[1] = nn.Linear(in_features=1280, out_features=2).to(device)

# model = torchvision.models.SwinTransformer(patch_size=[4, 4],
#         embed_dim=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=[7, 7],
#         stochastic_depth_prob=0.2, num_classes=2).to(device)

# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform(m.weight.data)
# model.apply(weights_init)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(initialize_weights)

# model = torchvision.models.resnet18(num_classes=2).to(device)
model = torchvision.models.resnet18(num_classes=2).to(device)
# model = torchvision.models.resnext50_32x4d()
# torchvision.models.resnet._ovewrite_named_param({}, "groups", 32)
# torchvision.models.resnet._ovewrite_named_param({}, "width_per_group", 4)
# model = torchvision.models.ResNet(torchvision.models.resnet.Bottleneck, [1, 1, 1, 1], num_classes=2).to(device)
# model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), #these represent the default parameters
#                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=3)

print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")


# %%


# +
# https://github.com/albu/albumentations/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb
img_transform_train = Compose([
    HistMatch(),
    # VerticalFlip(p=.5),
    # HorizontalFlip(p=.5),
    # HueSaturationValue(hue_shift_limit=15, sat_shift_limit=10, val_shift_limit=3, always_apply=False, p=0.9),
    # RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    # Rotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    # ElasticTransform(always_apply=True, approximate=True, alpha=150, sigma=8,alpha_affine=50),
    # RandomSizedCrop((patch_size, patch_size), patch_size, patch_size),
    ToTensorV2()
])
img_transform_val = Compose([
    ToTensorV2()
])
transform_dict = {"train": img_transform_train, "val": img_transform_val}
dataset = {}
dataLoader = {}
for phase in phases:  # now for each of the phases, we're creating the dataloader
    # interestingly, given the batch size, i've not seen any improvements from using a num_workers>0

    dataset[phase] = Dataset(f"./{dataname}_{phase}.pytable", img_transform=transform_dict[phase])
    dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size,
                                   shuffle=True, num_workers=0, pin_memory=True)
    print(f"{phase} dataset size:\t{len(dataset[phase])}")

# %%
# +
if code_mode == "stain_norm_examples":
# visualize a single example to verify that it is correct
    for i in range(len(dataset["train"])):
        for j in range(4):
            (img, label, img_old) = dataset["train"][i]
            img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            plt.imsave(f"StainNormExamples_RP/{i}_{j}_norm.png", dpi=600, arr=img)

        plt.imsave(f"StainNormExamples_RP/{i}_old.png", dpi=600, arr=img_old)

        bla = 1

# %%
optim = torch.optim.Adam(model.parameters(), lr=0.001)  # adam is going to be the most robust, though perhaps not the best performing, typically a good place to start
# optim = torch.optim.SGD(model.parameters(),
#                           lr=.1,
#                           momentum=0.9,
#                           weight_decay=0.0005)

# +
# we have the ability to weight individual classes, in this case we'll do so based on their presense in the trainingset
# to avoid biasing any particular class
nclasses = dataset["train"].classsizes.shape[0]
class_weight = dataset["train"].classsizes
class_weight = torch.from_numpy(1 - class_weight / class_weight.sum()).type('torch.FloatTensor').to(device)

print(class_weight)  # show final used weights, make sure that they're reasonable before continouing
criterion = nn.CrossEntropyLoss(weight=class_weight)
# %%
# +

if code_mode == "train":
    # def trainnetwork():
    best_loss_on_test = np.Infinity
    best_f1_on_test = 0
    train_loss_array = []
    val_loss_array = []
    accuracy_array = []
    sensitivity_array = []
    specificity_array = []
    precision_array = []
    start_time = time.time()
    act = nn.Softmax()
    for epoch in range(num_epochs):
        # zero out epoch based performance variables
        all_acc = {key: 0 for key in phases}
        all_auc = {key: 0 for key in phases}
        all_loss = {key: torch.zeros(0).to(device) for key in phases}  # keep this on GPU for greatly improved performance
        cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}
        preds = []
        gt = []
        for phase in phases:  # iterate through both training and validation states
            if phase == 'train':
                model.train()  # Set model to training mode
            else:  # when in eval mode, we don't want parameters to be updated
                model.eval()  # Set model to evaluate mode

            for ii, (X, label, img_orig) in enumerate(dataLoader[phase]):  # for each of the batches
                X = X.type('torch.FloatTensor').to(device)  # [Nbatch, 3, H, W]
                label = label.type('torch.LongTensor').to(
                    device)  # [Nbatch, 1] with class indices (0, 1, 2,...num_classes)

                with torch.set_grad_enabled(
                        phase == 'train'):  # dynamically set gradient computation, in case of validation, this isn't needed
                    # disabling is good practice and improves inference time
                    prediction = act(model(X)) # [N, Nclass]
                    loss = criterion(prediction, label)

                    if phase == "train":  # in case we're in train mode, need to do back propogation
                        optim.zero_grad()
                        loss.backward()
                        # for name, param in model.named_parameters():
                        #     print(name, param.grad)

                        optim.step()
                        train_loss = loss

                    all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))

                    if phase in validation_phases:  # if this phase is part of validation, compute confusion matrix
                        p = prediction.detach().cpu().numpy()
                        cpredflat = np.argmax(p, axis=1).flatten()
                        yflat = label.cpu().numpy().flatten()
                        CM = scipy.sparse.coo_matrix((np.ones(yflat.shape[0], dtype=np.int64), (yflat, cpredflat)),
                                                     shape=(num_classes, num_classes), dtype=np.int64).toarray()
                        cmatrix[phase] = cmatrix[phase] + CM
        #                     cmatrix[phase]=cmatrix[phase]+confusion_matrix(yflat,cpredflat, labels=range(nclasses))

            all_acc[phase] = (cmatrix[phase] / cmatrix[phase].sum()).trace()
            all_loss[phase] = all_loss[phase].cpu().numpy().mean()


        print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f, train acc: %.4f test acc: %.4f' % (timeSince(start_time, (epoch + 1) / num_epochs),
                                                                       epoch + 1, num_epochs,
                                                                       (epoch + 1) / num_epochs * 100, all_loss["train"],
                                                                       all_loss["val"], all_acc["train"], all_acc["val"]), end="")
        CM = cmatrix['val']
        FP = CM.sum(axis=0) - np.diag(CM)
        FN = CM.sum(axis=1) - np.diag(CM)
        TP = np.diag(CM)
        TN = CM.sum() - (FP + FN + TP)

        val_loss_array.append(all_loss['val'])
        accuracy_array.append(CM.diagonal().sum() / cmatrix[phase].sum())
        train_loss_array.append(all_loss['train'])
        sensitivity_array.append(TP / (TP + FN))
        precision_array.append(TP / (TP + FN))
        specificity_array.append(TN / (TN + FP))
        f1 = 2 * (precision_array[-1] * sensitivity_array[-1]) / (precision_array[-1] + sensitivity_array[-1])
        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            #     if f1 > best_f1_on_test:
            best_loss_on_test = all_loss["val"]
            #         best_f1_on_test = f1
            print("  **")
            state = {'epoch': epoch + 1,
                     'model_dict': model.state_dict(),
                     'optim_dict': optim.state_dict(),
                     'best_loss_on_test': all_loss,
                     'in_channels': in_channels,
                     'growth_rate': growth_rate,
                     'block_config': block_config,
                     'num_init_features': num_init_features,
                     'bn_size': bn_size,
                     'drop_rate': drop_rate,
                     'num_classes': num_classes}

            if len(model_tag) > 0:
                torch.save(state, f"{dataname}_densenet_best_model_{model_tag}.pth")
            else:
                torch.save(state, f"{dataname}_densenet_best_model.pth")

        else:
            print("")
    # %%
    epochs = range(num_epochs)
    plt.plot(epochs, train_loss_array, 'r', label='train loss')
    # np.convolve(val_loss_array,np.ones(10)/10, mode='same')
    plt.plot(epochs, val_loss_array, 'b', label='val loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Train-Val Loss for Densenet Tumor Classification")
    plt.legend()
    plt.show()

    plt.plot(epochs, accuracy_array, 'r', label='Accuracy')
    plt.plot(epochs, sensitivity_array, 'b', label='Sensitivity')
    plt.plot(epochs, specificity_array, 'g', label='Specificity')
    plt.xlabel("Epoch")
    plt.ylabel("Percentage")
    plt.title("Accuracy Metrics for Densenet Tumor Classification")
    plt.legend()
    plt.show()

    cm = cmatrix['val']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    # %%
    print(best_f1_on_test)
    # %%
    # +
    # #%load_ext line_profiler
    # #%lprun -f trainnetwork trainnetwork()

    # +
    # At this stage, training is done...below are snippets to help with other tasks: output generation + visualization
    # -


if code_mode == "test":
    # ----- generate output
    # load best model
    dataname = "pca_detect_dense_rpF"
    modelname_b = "pca_detect_dense_bpF"
    modelname_r = "pca_detect_dense_rpF"
    model_tag = "_rn"
    save_tiles = False
    if len(model_tag) > 0:
        checkpoint_b = torch.load(f"{modelname_b}_densenet_best_model_{model_tag}.pth")
        checkpoint_r = torch.load(f"{modelname_r}_densenet_best_model_{model_tag}.pth")
    else:
        checkpoint_b = torch.load(f"{modelname_b}_densenet_best_model.pth")
        checkpoint_r = torch.load(f"{modelname_r}_densenet_best_model.pth")
    # checkpoint = torch.load(f"{dataname}_densenet_best_model.pth")
    if model_tag == "":
        model_b = DenseNet(growth_rate=growth_rate, block_config=block_config,
                 num_init_features=num_init_features,
                 bn_size=bn_size,
                 drop_rate=drop_rate,
                 num_classes=num_classes).to(device)
        model_r = DenseNet(growth_rate=growth_rate, block_config=block_config,
                 num_init_features=num_init_features,
                 bn_size=bn_size,
                 drop_rate=drop_rate,
                 num_classes=num_classes).to(device)
    elif model_tag == "_rn":
        model_b = torchvision.models.resnet18(num_classes=2).to(device)
        model_r = torchvision.models.resnet18(num_classes=2).to(device)
    elif model_tag == "_svt":
        model_b = torchvision.models.SwinTransformer(patch_size=[4, 4],
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=[7, 7],
                stochastic_depth_prob=0.2, num_classes=2).to(device)
        model_r = model_b.copy()
    elif model_type == "_en":
        model_b = torchvision.models.efficientnet_b0().to(device)
        model_b.classifier[1] = nn.Linear(in_features=1280, out_features=2).to(device)
        model_r = torchvision.models.efficientnet_b0().to(device)
        model_r.classifier[1] = nn.Linear(in_features=1280, out_features=2).to(device)

    model_b.load_state_dict(checkpoint_b["model_dict"])
    model_r.load_state_dict(checkpoint_r["model_dict"])

    for phase in ["val"]:  # now for each of the phases, we're creating the dataloader

        dataset[phase] = Dataset(f"./{dataname}_{phase}.pytable", img_transform=transform_dict[phase])
        dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size,
                                       shuffle=False, num_workers=0, pin_memory=True)
        print(f"{phase} dataset size:\t{len(dataset[phase])}")
    # %%
    num_classes = 2
    datatype = "val"
    bpred = []
    rpred = []
    gt = []
    for ii, (X, label, img_orig) in enumerate(dataLoader[datatype]):  # for each of the batches
        X = X.type('torch.FloatTensor').to(device)  # [Nbatch, 3, H, W]
        label = label.type('torch.LongTensor').to(device)  # [Nbatch, 1] with class indices (0, 1, 2,...num_classes)
        prediction_b = model_b(X)
        prediction_r = model_r(X)  # [N, Nclass]
        # [N, Nclass]
        p_b = prediction_b.detach().cpu().numpy()
        p_r = prediction_r.detach().cpu().numpy()
        cpredflat_b = np.argmax(p_b, axis=1).flatten().tolist()
        cpredflat_r = np.argmax(p_r, axis=1).flatten().tolist()
        yflat = label.cpu().numpy().flatten().tolist()

        bpred.extend(cpredflat_b)
        rpred.extend(cpredflat_r)
        gt.extend(yflat)

    counter = 0
    filenames = []
    for ii, (X, label, img_orig) in enumerate(dataLoader[datatype]):
        X = X.detach().cpu().numpy()
        for i in range(np.shape(X)[0]):
            xi = X[i].transpose(1,2,0)
            if save_tiles:
                plt.imsave(f"RP_AnalysisTiles/{counter}.png", xi)
            filenames.append(f"{counter}.png")
            counter += 1

    data_dict = {"file": filenames, "ground_truth": gt, "Mb_prediction": bpred, "Mr_prediction": rpred}
    data_df = pd.DataFrame(data_dict)
    csv_name = f"{dataname}_predictions_{model_tag}.csv" if len(model_tag) > 1 else f"{dataname}_predictions.csv"
    data_df.to_csv(csv_name)

    # Compute metrics for both BX and RP models
    bx_cm = sklearn.metrics.confusion_matrix(gt, bpred)
    rp_cm = sklearn.metrics.confusion_matrix(gt, bpred)
    metrics = {"f1_score": {"bx": sklearn.metrics.f1_score(gt, bpred), "rp": sklearn.metrics.f1_score(gt, rpred)},
               "sensetivity": {"bx": bx_cm[0,0]/(bx_cm[0,0]+bx_cm[0,1]), "rp": rp_cm[0,0]/(rp_cm[0,0]+rp_cm[0,1])},
               "specificty": {"bx": bx_cm[1,1]/(bx_cm[1,0]+bx_cm[1,1]), "rp": rp_cm[1,1]/(rp_cm[1,0]+rp_cm[1,1])}}
    for metric in metrics.keys():
        bx_m = metrics[metric]["bx"]
        rp_m = metrics[metric]["rp"]
        print(f"{metric} |  Biopsy Model: {bx_m}, Radical Pros Model: {rp_m}")


if code_mode == "organize":
    folder = "RP_AnalysisTiles/*.png"
    files = glob.glob(folder)
    results_sheet = pd.read_csv("pca_detect_dense_rpF_predictions.csv")
    random_sample = random.sample(files, 400)
    random_sample_indeces = [int(os.path.split(x)[1].split(".")[0]) for x in random_sample]
    random_sample_df = results_sheet[results_sheet.index.isin(random_sample_indeces)]
    random_sample_df.to_csv("RP_AnalysisTiles/Selected/pca_detect_dense_rpF_predictions_sampled.csv")
    for file in random_sample:
        dst = os.path.split(file)[0] + "/Selected/" + os.path.split(file)[1]
        shutil.copyfile(file, dst)

# Flip prediction ofr biopsy dataset when using model trained on WSIs
# for i in [0, 1]:
#     temp = cmatrix['val'][i, 0]
#     cmatrix['val'][i, 0] = cmatrix['val'][i, 1]
#     cmatrix['val'][i, 1] = temp

# accuracy = cmatrix[phase].diagonal().sum() / cmatrix[phase].sum()
# print(f'Model Accuracy: {accuracy}')
# cm = cmatrix['val']
#
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#
# disp.plot()
# plt.show()
#
# # %%
# # Binary classification results
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title("UPenn Classification Results")
# plt.show()
#
# CM = cm
# FP = CM[0, 1]
# FN = CM[1, 0]
# TP = CM[1, 1]
# TN = CM[0, 0]
# sensitivity = TP / (TP + FN)
# precision = TP / (TP + FP)
# specificity = TN / (TN + FP)
# f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
# print("Upenn Val Results")
# print(f"Sensitivity: {sensitivity}")
# print(f"Specificity: {specificity}")
# print(f"F1: {f1}")
# # %%
# # High grade vs Low grade accuracy
# # cm_hlg = np.zeros((2, 2))
# # cm_hlg[0, 0] = np.sum(cm[0:3, 0:3])
# # cm_hlg[0, 1] = np.sum(cm[0:3, 3:])
# # cm_hlg[1, 0] = np.sum(cm[3:, 0:3])
# # cm_hlg[1, 1] = np.sum(cm[3:, 3:])
# # disp = ConfusionMatrixDisplay(confusion_matrix=cm_hlg)
# # disp.plot()
# # plt.title("High Grade vs Low Grade Classification")
# # plt.show()
#
# # CM = cm_hlg
# # FP = CM[0, 1]
# # FN = CM[1, 0]
# # TP = CM[1, 1]
# # TN = CM[0, 0]
# # sensitivity = TP / (TP + FN)
# # precision = TP / (TP + FP)
# # specificity = TN / (TN + FP)
# # f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
# # print("High Grade vs Low-Grade")
# # print(f"Sensitivity: {sensitivity}")
# # print(f"Specificity: {specificity}")
# # print(f"F1: {f1}")
#
# # Cancer vs non-cancer accuracy
# # cm_cnc = np.zeros((2, 2))
# # cm_cnc[0, 0] = np.sum(cm[0:2, 0:2])
# # cm_cnc[0, 1] = np.sum(cm[0:2, 2:])
# # cm_cnc[1, 0] = np.sum(cm[2:, 0:2])
# # cm_cnc[1, 1] = np.sum(cm[2:, 2:])
# # disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnc)
# # disp.plot()
# # plt.title("Cancer vs Non-Cancer Classification")
# # plt.show()
# #
# # CM = cm_cnc
# # FP = CM[0, 1]
# # FN = CM[1, 0]
# # TP = CM[1, 1]
# # TN = CM[0, 0]
# # sensitivity = TP / (TP + FN)
# # precision = TP / (TP + FP)
# # specificity = TN / (TN + FP)
# # f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
# # print("Cancer vs Non-Cancer")
# # print(f"Sensitivity: {sensitivity}")
# # print(f"Specificity: {specificity}")
# # print(f"F1: {f1}")
# # %%
# # All classification Accuracy
# CM = cmatrix['val']
# FP = CM.sum(axis=0) - np.diag(CM)
# FN = CM.sum(axis=1) - np.diag(CM)
# TP = np.diag(CM)
# TN = CM.sum() - (FP + FN + TP)
# sensitivity = TP / (TP + FN)
# precision = TP / (TP + FP)
# specificity = TN / (TN + FP)
# # f1 = [(2*i*j)/(i+j) for (i,j) in zip(recall, sensitivity)]
# f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
# print(sensitivity)
# print(specificity)
# print(f1)
# # %%
# # test = np.array([1, 2, 3]) * np.array([4, 5, 6]) / np.array([2, 2, 2])
# # print(test)
# # %%
