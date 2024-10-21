# %%

# coding: utf-8

# v2
# 7/11/2018

# %%


import argparse
import os
import glob
import numpy as np
import cv2
import torch
import sys
import time
import math
from pathlib import Path
from torchvision.models import DenseNet
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
os.add_dll_directory("C:/Users/bxf169/openslide-win64-20171122/openslide-win64-20171122/bin")
import tables
from WSI_handling import wsi
from unet import UNet
from shapely.geometry import Polygon
from albumentations import *
from albumentations.pytorch import ToTensorV2
import random
import torchstain
import torchvision.transforms as trans
# %%
class arguments:
    def __init__(self):
        self.model = 'E:/Brennan/bxf169/BladderToProstateConversion/pca_detect_dense_rpF_densenet_best_model.pth'
        self.batchsize = 10
        self.gpuid = 0
        self.resolution = 2
        self.annotation = 'wsi'
        self.color = 'None'
        #         self.basepath = 'TumorSegmentationTraining/UpennBiopsies'
        #         self.basepath = '../../../data/UPenn_Prostate_Histology/Progressor_nonProgressorProstate/histologyImages/UPenn'
        self.basepath = 'E:/Brennan/RTOG_0521'
        self.input_pattern = '*.svs'
        self.force = False
        self.outdir = 'E:/Brennan/RTOG_0521/MR_overlays_NORM/Masks'
        self.patchsize = 224


# %%
# %%
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
hm_file = "rp_bp_hist_norm.pytable"
histmatch = hist_data(hm_file, img_transform=None)


# %%
class HistMatch(ImageOnlyTransform):
    def __init__(self,always_apply=True,p=1.0, sample = "All"):
        super(HistMatch, self).__init__(always_apply, p)
        if sample == "Biopsy":
            sample_range = (0,49)
        elif sample == "Surgical":
            sample_range = (50,99)
        else:
            sample_range = (0,99)
        self.sample_range = sample_range
    def apply(self, image, **params):
        target = histmatch[random.randint(self.sample_range[0], self.sample_range[1])]
        i = image.copy()
        T = trans.Compose([trans.ToTensor(), trans.Lambda(lambda x: x*255)])

        # try:
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(target))

        image = T(image)
        t, H, E = torch_normalizer.normalize(I=image, stains=True)
        t = t.numpy().astype(np.uint8)
        # except:
        #     t = i

        return t

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")

img_transform = Compose([
       HistMatch(sample="Surgical"),
       ToTensorV2()
    ])
# -----helper function to split data into batches
def divide_batch(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


args = arguments()
# %%
# %%
if not (args.input_pattern):
    parser.error('No images selected with input pattern')

# %%


OUTPUT_DIR = args.outdir

# %%


batch_size = args.batchsize
patch_size = args.patchsize
base_stride_size = patch_size // 2

# %%
# %%
# ----- load network
device = torch.device(args.gpuid if torch.cuda.is_available() else 'cpu')
print(device)
# %%


checkpoint = torch.load(args.model, map_location=lambda storage,
                                                        loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
print(checkpoint["num_init_features"])
model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                 num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                 drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_dict"])
model.eval()

# %%


print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")


# %%

# %%
# ----- get file list

def run_model(img_dims, patch_size, stride_size, base_stride_size, batch_size, args, img, annotation):
    x_start = int(img_dims[0])
    y_start = int(img_dims[1])
    w_orig = img.get_coord_at_mpp(img_dims[2] - x_start, input_mpp=img['mpp'], output_mpp=args.resolution)
    h_orig = img.get_coord_at_mpp(img_dims[3] - y_start, input_mpp=img['mpp'], output_mpp=args.resolution)

    w = int(w_orig + (patch_size - (w_orig % patch_size)))
    h = int(h_orig + (patch_size - (h_orig % patch_size)))

    base_edge_length = base_stride_size * int(math.sqrt(batch_size))

    # need to make sure we don't end up with a last row/column smaller than patch_size
    h = h + patch_size if ((h + base_stride_size) % base_edge_length) < patch_size else h
    w = w + patch_size if ((w + base_stride_size) % base_edge_length) < patch_size else w
    x_points = range(x_start, h, patch_size)
    y_points = range(y_start, w, patch_size)
    grid_points = [(x, y) for x in x_points for y in y_points]

    output = np.zeros([h + patch_size, w + patch_size], dtype='uint8')

    for i, batch_points in enumerate(grid_points):
        # get the tile of the batch
        xc = img.get_coord_at_mpp(batch_points[0], input_mpp=args.resolution, output_mpp=img['mpp'])
        yc = img.get_coord_at_mpp(batch_points[1], input_mpp=args.resolution, output_mpp=img['mpp'])
        big_patch = img.get_tile(args.resolution, (yc, xc), (patch_size, patch_size))
        white_mask = np.mean(big_patch, axis=2) > 220
        tissue_mask = ~(np.mean(big_patch, axis=2) < 35) & ~(np.mean(big_patch, axis=2) > 220)
        if np.sum(tissue_mask*1)/(np.shape(tissue_mask)[0]**2) > .30:
            # ---- get results
            if img_transform:
                try:
                    batch_arr = img_transform(image=big_patch)['image']
                    batch_arr = batch_arr[None,::].type('torch.FloatTensor').to(device)
                except:
                    big_patch = np.expand_dims(big_patch, axis=0)
                    big_patch_gpu = torch.from_numpy(big_patch).type('torch.FloatTensor').to(device)
                    batch_arr = big_patch_gpu.permute(0, 3, 1, 2)
            else:
                big_patch = np.expand_dims(big_patch, axis=0)
                big_patch_gpu = torch.from_numpy(big_patch).type('torch.FloatTensor').to(device)
                batch_arr = big_patch_gpu.permute(0, 3, 1, 2)

            # plt.imshow(big_patch)
            # plt.show()
            output_batch = model(batch_arr)
            output_batch = output_batch.argmax(axis=1)

            # --- pull from GPU and append to rest of output
            output_batch = output_batch.detach().cpu().numpy()
            # ---- create tile masks
            output_batch = np.array(
                [np.zeros((patch_size, patch_size)) if out == 0 else np.ones((patch_size, patch_size)) for out in
                 output_batch])
            #         output_batch = np.array([np.zeros((patch_size, patch_size)) if out == 1 else np.ones((patch_size, patch_size)) for out in
            #                 output_batch]) # for biopsy data, tumor and benign are flipped
            output_batch = np.squeeze(output_batch)
            reconst = ((output_batch > 0) & (~white_mask)) * 1
            output[batch_points[0]:(batch_points[0] + patch_size),
            batch_points[1]:(batch_points[1] + patch_size)] = reconst
        else:
            output[batch_points[0]:(batch_points[0] + patch_size),
            batch_points[1]:(batch_points[1] + patch_size)] = 0
    if (args.annotation.lower() != 'wsi'):
        # in case there was extra padding to get a multiple of patch size, remove that as well
        _, mask = img.get_annotated_region(args.resolution, args.color, annotation, return_img=False)
        output = output[0:mask.shape[0], 0:mask.shape[1]]  # remove padding, crop back
        output = np.bitwise_and(output > 0, mask > 0) * 255
    else:
        output = output * 255
        output = output[:h_orig, :w_orig]
        if args.input_pattern == "*.mrxs":
            # For core-plus images, .mrxs files are dumb and load weirdly, have to cut out random black space
            small_image = img.get_wsi(desired_mpp=8)
            tissue_mask = ~(np.mean(small_image, axis=2) < 35)
            coords = np.where(tissue_mask)
            coords = np.array([list(t) for t in coords])
            img_limits = [np.min(coords[0]), np.min(coords[1]), np.max(coords[0]), np.max(coords[1])]
            # plt.imshow(small_image[img_limits[0]:img_limits[2], img_limits[1]:img_limits[3]])
            # plt.show()
            for n, item in enumerate(img_limits):
                img_limits[n] = img.get_coord_at_mpp(img_limits[n], input_mpp=8, output_mpp=args.resolution)
            output = output[img_limits[0]:img_limits[2], img_limits[1]:img_limits[3]]
    return output


# %%
# %%

OUTPUT_DIR = args.outdir
print(OUTPUT_DIR)
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# %%
# %%
files = []
basepath = args.basepath  #
basepath = basepath + os.sep if len(
    basepath) > 0 else ""  # if the user supplied a different basepath, make sure it ends with an os.sep

# %%
files = glob.glob(args.basepath + '/' + args.input_pattern)
print(files)
# %%
# %%

for fname in files:
    fname = fname.strip()

    if (args.annotation.lower() != 'all'):

        newfname_class = "%s/%s_class_normv2.png" % (OUTPUT_DIR, Path(fname).stem)

        if not args.force and os.path.exists(newfname_class):
            print("Skipping as output file exists")
            continue
        print(f"working on file: \t {fname}")
        print(f"saving to : \t {newfname_class}")

        start_time = time.time()
        cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))
    xml_fname = Path(fname).with_suffix('.xml')
    if not os.path.exists(xml_fname):
        # xml_fname = Path(fname).with_suffix('.json')
        xml_fname = None

    #     if os.path.exists(xml_fname) and os.path.exists(fname):
    if os.path.exists(fname):
        img = wsi(fname, xml_fname)
        stride_size = int(base_stride_size * (args.resolution / img["mpp"]))

        if (args.annotation.lower() == 'all'):
            annotations_todo = len(img.get_points(args.color, [])[0])
            print(f"working on file: \t {fname}")

            for k in range(0, annotations_todo):
                print('Working on annotation ' + str(k))
                start_time = time.time()
                newfname_class = "%s/%s_%d_class_normv2.png" % (OUTPUT_DIR, Path(fname).stem, k)

                if args.force or not os.path.exists(newfname_class):
                    cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))
                    img_dims = img.get_dimensions_of_annotation(args.color, k)
                    output = run_model(img_dims, patch_size, stride_size, base_stride_size, batch_size, args, img,
                                       annotation=k)
                    cv2.imwrite(newfname_class, output)

                output = None
                print('Elapsed time = ' + str(time.time() - start_time))

        else:

            if (args.annotation.lower() == 'wsi'):
                img_dims = [0, 0, img["img_dims"][0][0], img["img_dims"][0][1]]
            else:
                img_dims = img.get_dimensions_of_annotation(args.color, args.annotation)

            if img_dims:
                print("Starting to run model...")
                output = run_model(img_dims, patch_size, stride_size, base_stride_size, batch_size, args, img,
                                   annotation=args.annotation)
                # plt.imshow(output)
                # plt.show()
                cv2.imwrite(newfname_class, output)
                output = None
                print('Elapsed time = ' + str(time.time() - start_time))

            else:
                print('No annotation of color')
    else:
        print('Could not find ' + str(xml_fname))