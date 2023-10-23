import torch
import numpy as np
from PIL import Image
import cv2

def split_squares(img, pos):
    h = img.shape[1]
    if(pos == 0):
        return img[:, :, :h]
    else:
        return img[:, :, -h:]

def normalize(img):
    return img/255

def hwc_to_chw(img):
    return np.transpose(img, (2, 0, 1))

def reduce_channel(img):
    if(img[:, :, 0] == img[:, :, 1] and img[:, :, 1] == img[:, :, 2]):
        return img[:, :, 0]

def load_data(img_path):
    if img_path.find("train") != -1:
        gt_path = img_path.replace("train", "train_mask")
    elif img_path.find("val") != -1:
        gt_path = img_path.replace("val", "val_mask")
    else:
        gt_path = img_path.replace("test", "test_mask")

    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path)

    img = crop_and_resize(img)
    gt = crop_and_resize(gt)
    
    return img, gt        

def crop_and_resize(img):
    '''
    Crop and resize image to 256 x 256
    '''
    resized_image1 = cv2.resize(img, (384, 512), interpolation=cv2.INTER_NEAREST)[240:400]
    resized_image2 = cv2.resize(resized_image1, (256, 256), interpolation=cv2.INTER_NEAREST)
    return resized_image2