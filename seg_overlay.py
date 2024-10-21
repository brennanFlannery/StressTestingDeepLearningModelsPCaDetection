import os
import glob
import numpy as np
import cv2
os.add_dll_directory("C:/Users/bxf169/openslide-win64-20171122/openslide-win64-20171122/bin")

import sys
from WSI_handling import wsi
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import seaborn as sb
Image.MAX_IMAGE_PIXELS = 1000000000
import skimage.morphology as morphology

# Directories of the gt images and the segmentation masks
segPath = "E:/Brennan/RTOG_0521/MR_overlays_NORM/Masks"
imgPath = "E:/Brennan/RTOG_0521"
savePath = "E:/Brennan/RTOG_0521/MR_overlays_NORM/Overlays"
# GT image file extension
ext = ".svs"
# Get list of appropriate files
img_files = glob.glob(imgPath + os.path.sep + "*" + ext)
mask_files = glob.glob(segPath + os.path.sep + "*.png")

resolution = 2  # mpp of the segmentations (usually 2)
overlay_figure = False
contour_figure = True
high_res = True
high_res_mpp = 1
num_file_limit = 100 # Set to 0 if not using

if num_file_limit > 0:
    img_files = img_files[:num_file_limit]
    mask_files = mask_files[:num_file_limit]


def imfill(image):
    im_th = np.array((image > 0) * 255).astype(np.uint8).copy()
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out

for i in range(len(img_files)):
    img_file = img_files[i]
    print(f"Working on {img_file}")
    mask_file = mask_files[i]
    file = os.path.split(img_file)[1].split(".")[0]

    img = wsi(img_file)
    mask = (plt.imread(mask_file) > 0) * 1

    small_image = img.get_wsi(desired_mpp=8)

    # Limit segmentation to areas where tissue is present (helpful to remove artifacts)
    tissue_mask = ~(np.mean(small_image, axis=2) < 35) & ~(np.mean(small_image, axis=2) > 220)
    tissue_mask = imfill(tissue_mask)
    new_size = (np.shape(mask)[1], np.shape(mask)[0])
    tissue_mask = (cv2.resize(np.uint8(tissue_mask), new_size) > 0) * 1
    mask[tissue_mask == 0] = 0

    mask = (imfill(mask).astype(np.int32) > 0) * 1

    footprint = morphology.disk(20)
    res = morphology.white_tophat(mask, footprint)
    mask = mask - res

    if overlay_figure:
        big_image = img.get_wsi(desired_mpp=resolution)

        mask_overlap = np.zeros((np.shape(big_image)[0], np.shape(big_image)[1], 3))
        mask_overlap[:,:,0] = (mask > 0) * 255 # Biopsy is red
        overlay = cv2.addWeighted(mask_overlap.astype(np.uint8), 1, big_image, .8, 0)
        newfname = savePath + "/" + file + "_seg_overlay.png"
        cv2.imwrite(newfname, cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
        print(f"Written overlay file {newfname}")

    if contour_figure:
        if high_res:
            big_image = img.get_wsi(desired_mpp=high_res_mpp)
            new_size = (np.shape(big_image)[1], np.shape(big_image)[0])
            mask = (cv2.resize(np.uint8(mask), new_size).astype(np.int32) > 0) * 1
            newfname = savePath + "/" + file + "_seg_highres_contours.png"
        else:
            big_image = img.get_wsi(desired_mpp=resolution)
            newfname = savePath + "/" + file + "_seg_contours.png"

        # mask_overlap = bmask + (3 * rmask) + (5 * gt) # make sure every combination has unique values
        mask_overlap = mask  # make sure every combination has unique values

        vals = np.unique(mask_overlap)[1:] # remove 0, we dont care about it since its just background
        # all the unique values and their colors
        color_wheel = {1: (255,0,0), 3: (0,255,0), 4: (255,255,0), 5: (0,0,255), 6: (255,0,255), 8: (0,255,255), 9: (255,255,255)}

        # plt.imshow(big_image)
        for val in vals:
            mask = mask_overlap.copy()
            mask[mask != val] = 0
            mask = np.array((mask > 0) * 1)

            if mask.any():
                contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                color = color_wheel[val]
                cv2.drawContours(big_image, contours, -1, color, 5)

        cv2.imwrite(newfname, cv2.cvtColor(big_image,cv2.COLOR_BGR2RGB))
        print(f"Written contour file {newfname}")


