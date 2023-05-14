#!/usr/bin/env python
# coding: utf-8

# # Keras starter kit [full training set, UNet]

# ## Setup

# In[67]:


import numpy as np
import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

import glob
import time
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from tqdm import tqdm
import cv2

# Data config
# DATA_DIR = '/kaggle/input/vesuvius-challenge-ink-detection/'
DATA_DIR = "."
BUFFER = 64  # Half-size of papyrus patches we'll use as model inputs
Z_LIST = list(range(0, 65, 4))  # Offset of slices in the z direction
Z_DIM = len(Z_LIST)  # Number of slices in the z direction. Max value is 64 - Z_START
SHARED_HEIGHT = 4000  # Height to resize all papyrii

# (y, x)
val_location = (600, 500)
val_zone_size = (1000, 500)

# Model config
BATCH_SIZE = 64
USE_MIXED_PRECISION = False
USE_JIT_COMPILE = False

device = torch.device("cuda")
threshold = 0.25
num_workers = 2
exp = 1e-7


# In[68]:


from scipy.stats import median_abs_deviation

def calculate_MAD(volume):
    all_MAD = median_abs_deviation(volume, axis=[0, 1])
    return all_MAD
    
def calculate_median(volume):
    all_median = np.median(volume, axis=[0, 1])
    return all_median


# In[69]:


all_median = np.array([19581., 19618., 19645., 19710., 19944., 20561., 21908., 22458.,
        18980., 17848., 20094., 21848., 22629., 22993., 23171., 23261.,
        23305.])


# In[70]:


all_MAD = np.array([12424., 12561., 12718., 12864., 12953., 13099., 13550., 14592.,
        13639.,  7500.,  4828.,  3576.,  3067.,  2808.,  2666.,  2588.,
         2550.])


# In[71]:


possible_max_input = ((2 ** 16 - 1) / all_median.min())
possible_max_input


# In[72]:


def resize(img):
    current_height, current_width = img.shape    
    aspect_ratio = current_width / current_height
    new_width = int(SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, SHARED_HEIGHT)
    # (W, H)の順で渡すが結果は(H, W)になっている
    img = cv2.resize(img, new_size)
    return img

def load_mask(split, index):
    img = cv2.imread(f"{DATA_DIR}/{split}/{index}/mask.png", 0) // 255
    img = resize(img)    
    return img


def load_labels(split, index):
    img = cv2.imread(f"{DATA_DIR}/{split}/{index}/inklabels.png", 0) // 255
    img = resize(img)
    return img


# In[73]:


def load_volume(split, index):
    # Load the 3d x-ray scan, one slice at a time
    all = sorted(glob.glob(f"{DATA_DIR}/{split}/{index}/surface_volume/*.tif"))
    z_slices_fnames = [all[i] for i in range(len(all)) if i in Z_LIST]
    assert len(z_slices_fnames) == Z_DIM
    z_slices = []
    for z, filename in  tqdm(enumerate(z_slices_fnames)):
        img = cv2.imread(filename, -1)
        img = resize(img)
        z_slices.append(img)
    return np.stack(z_slices, axis=-1)


# In[74]:


def sample_random_location(shape):
    random_train_x = np.random.randint(low=BUFFER, high=shape[1] - BUFFER - 1, size=())
    random_train_y = np.random.randint(low=BUFFER, high=shape[0] - BUFFER - 1, size=())
    random_train_location = np.stack([random_train_y, random_train_x], axis=-1)
    return random_train_location


def is_in_masked_zone(location, mask):
    return mask[location[0], location[1]] > 0

def is_in_val_zone(location, val_location, val_zone_size):
    x = location[1]
    y = location[0]
    x_match = val_location[1] - BUFFER <= x <= val_location[1] + val_zone_size[1] + BUFFER
    y_match = val_location[0] - BUFFER <= y <= val_location[0] + val_zone_size[0] + BUFFER
    return x_match and y_match


# In[75]:


printed = False

def extract_subvolume(location, volume):
    global printed
    # print(np.unique(volume, return_counts=True, return_index=True))
    x = location[0]
    y = location[1]
    subvolume = volume[x-BUFFER:x+BUFFER, y-BUFFER:y+BUFFER, :].astype(np.float32)
    # print("subvolume[:, :, 0]", subvolume[:, :, 0])
    median = np.full_like(subvolume, all_median).astype(np.float32)
    MAD = np.full_like(subvolume, all_MAD).astype(np.float32)
    # mean = np.mean(subvolume, axis=2)
    # mean = np.stack([mean for i in range(Z_DIM)], axis=2) + exp
    # MAD = median_abs_deviation(subvolume, axis=2)
    # print("MAD", MAD[0, 0, :])
    # print("mean", mean)
    # print("median", median[0, 0, :])
    
    subvolume = (subvolume - median) / MAD
    
    if not printed:
        print("subvolume after taking care of median and MAD", subvolume)
        printed = True
    
    return subvolume


# In[76]:


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

class SubvolumeDataset(Dataset):
    def __init__(self, locations, volume, labels, buffer, is_train: bool, return_location: bool = False):
        self.locations = locations
        self.volume = volume
        self.labels = labels        
        self.buffer = buffer
        self.is_train = is_train
        self.return_location = return_location

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        label = None
        location = np.array(self.locations[idx])
        y, x = location[0], location[1]

        subvolume = extract_subvolume(location, self.volume)        
        # print("subvolume", subvolume)
        # print("labels", labels)
        # subvolume = subvolume.numpy()
        subvolume = subvolume
        
        if self.labels is not None:
            label = self.labels[y - self.buffer:y + self.buffer, x - self.buffer:x + self.buffer]
            # print("label", label)
            # n_category = 2
            # label = np.eye(n_category)[label]
            label = np.stack([label], axis=-1)
            # label = label.numpy()
            # print("label.shape", label.shape
        
        if self.is_train and label is not None:            
            
            # print("label", label.dtype)
            # print("subvolume in dataset (before aug)", subvolume)            
            performed = A.Compose([            
                A.ToFloat(max_value=possible_max_input),
                A.RandomBrightnessContrast(),
                A.HorizontalFlip(),
                A.VerticalFlip(),  
            #     # A.Normalize(
            #     #     mean=[mean],
            #     #     std=[std],
            #     # ),
                A.FromFloat(max_value=possible_max_input),
            ])(image=subvolume, mask=label)
            subvolume = performed["image"]            
            label = performed["mask"]
            # print("subvolume in dataset (after aug)", subvolume)
            # print("label", label.dtype)
            # print("subvolume", subvolume.dtype)
            # →C, H, W
            subvolume = torch.from_numpy(subvolume.transpose(2, 0, 1).astype(np.float64))
            # print(performed)
            # print(subvolume.shape, label.shape)
            # H, W, C → C, H, W
            label = torch.from_numpy(label.transpose(2, 0, 1).astype(np.uint8)) 
        else:
            performed = A.Compose([  
                A.ToFloat(max_value=possible_max_input),                
                # A.Normalize(
                #     mean=[mean],
                #     std=[std],
                # ),
                A.FromFloat(max_value=possible_max_input),
            ])(image=subvolume)
            subvolume = performed["image"]
            subvolume = torch.from_numpy(subvolume.transpose(2, 0, 1).astype(np.float64))
            if label is not None:
                label = torch.from_numpy(label.transpose(2, 0, 1).astype(np.uint8)) 
        if self.return_location:
            return subvolume, location
        return subvolume, label        


# In[77]:


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(                
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        def transpose_conv_block(in_channels, out_channels):
            return nn.Sequential(                
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels if i == 2 else 64 * 2**(i - 1), 64 * 2**i, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64 * 2**i),
                nn.ReLU(),
                nn.Conv2d(64 * 2**i, 64 * 2**i, kernel_size=3, padding=1),
                nn.BatchNorm2d(64 * 2**i),
                nn.ReLU(),
            )
            for i in range(2, 5)
        ])


        self.middle = nn.Sequential(
            conv_block(1024, 512),
            conv_block(512, 1024),
        )
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                transpose_conv_block(2 ** (i + 7), 2 ** (i + 6)),
                transpose_conv_block(2 ** (i + 6), 2 ** (i + 5)),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )
            for i in range(4, 1, -1)
        ])
        self.final_decoder = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )
        self.activation = nn.Identity()

    def forward(self, x):
        # print("input:", x)
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        x = self.middle(x)
        
        # print("encoder ok", x)
        for i, layer in enumerate(self.decoder):            
            # print(f"decoder will {i}: ok", x.shape)
            x = torch.cat([x, skip_connections[-i-1]], dim=1)  # Concatenate along channel dimension
            # print(f"decoder with skip connection {i}: ok", x.shape)            
            x = layer(x)            
            # print(f"decoder {i}: ok", x)
        # print("decoder ok")
        x = self.final_decoder(x)
        x = self.activation(x)
        # print("final out", x)
        return x


# In[78]:


device = torch.device("cuda")


# In[79]:


model = UNet(Z_DIM, 2)
model = nn.DataParallel(model)
# model.load_state_dict(torch.load(f"/kaggle/input/ink-detection/model.pt"))
model.load_state_dict(torch.load("model.pt"))
model = model.to(device)


# ## Load up the training data

# In[80]:


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

def compute_predictions_map(split, index):
    global all_MAD, all_median
    
    print(f"Load data for {split}/{index}")

    test_volume = load_volume(split=split, index=index)        
    test_mask = load_mask(split=split, index=index)    
    
    # all_MAD = calculate_MAD(test_volume)
    # all_median = calculate_median(test_volume)

    test_locations = []
    stride = BUFFER // 2
    for y in range(BUFFER, test_volume.shape[0] - BUFFER, stride):
        for x in range(BUFFER, test_volume.shape[1] - BUFFER, stride):
            test_locations.append((y, x))

    print(f"{len(test_locations)} test locations (before filtering by mask)")

    # filter locations inside the mask
    test_locations = [loc for loc in test_locations if is_in_masked_zone(loc, test_mask)]
    
    print(f"{len(test_locations)} test locations (after filtering by mask)")

    test_ds = SubvolumeDataset(test_locations, test_volume, None, BUFFER, is_train=False, return_location=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=num_workers)

    # shape: (X, Y, C)
    predictions_map = np.zeros_like(test_volume[:, :, 0]).transpose((1, 0))[:, :, np.newaxis].astype(np.float64)
    
    print("test_volume.shape", test_volume.shape)
    print("predictions_map.shape", predictions_map.shape)

    print(f"Compute predictions")

    model.eval()  # set model to evaluation mode
    with torch.no_grad():    
        for patch_batch, loc_batch in tqdm(test_loader):
            loc_batch = loc_batch.to(device).long()
            patch_batch = patch_batch.to(device).float()
            predictions = model(patch_batch)
            # print("predictions", predictions)
            predictions = nn.Softmax(dim=1)(predictions)            
            predictions: torch.Tensor = predictions[:, 1, :, :].unsqueeze(dim=1)
            # print("predictions", predictions)
            # print("Softmaxed predictions where conf is gt threshold", predictions[predictions.gt(threshold)])
            # →(BATCH, W, H, C)
            predictions = torch.permute(predictions, (0, 3, 2, 1))
            predictions = predictions.cpu().numpy()  # move predictions to cpu and convert to numpy
            for (y, x), pred in zip(loc_batch, predictions):
                # print("index: ", index ,"x, y, pred", x.item(), y.item(), pred[BUFFER, BUFFER, :].item(), file=open('log.out', 'a'))
                predictions_map[
                    x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
                ][pred > threshold] = 1
    print("predictions_map", predictions_map, file=open("predictions_map", "w"))
    return predictions_map


# In[81]:


from skimage.transform import resize as resize_ski
import pathlib


# In[82]:


def rle(predictions_map, threshold):
    flat_img = (np.where(predictions_map.flatten() >= threshold, 1, 0)).astype(np.uint8)
    
    # Add padding at the beginning and end
    flat_img = np.pad(flat_img, pad_width=1, mode='constant', constant_values=0)

    starts = np.where((flat_img[:-1] == 0) & (flat_img[1:] == 1))[0]
    ends = np.where((flat_img[:-1] == 1) & (flat_img[1:] == 0))[0]

    lengths = ends - starts
    
    print(lengths.shape)

    return " ".join(map(str, np.c_[starts, lengths].flatten()))


# In[83]:


def update_submission(predictions_map, index):
    rle_ = rle(predictions_map, threshold=threshold)
    print(f"{index}," + rle_, file=open('submission.csv', 'a'))


# In[84]:


print("Id,Predicted", file=open('submission.csv', 'w'))
kind = "test"
folder = pathlib.Path(DATA_DIR) / kind
threshold = 0.25
for p in list(folder.iterdir()):
    index = p.stem
    predictions_map = compute_predictions_map(split=kind, index=index)
    original_size = cv2.imread(DATA_DIR + f"/{kind}/{index}/mask.png", 0).shape[:2]
    # W, H, C → H, W, C
    predictions_map = predictions_map.transpose((1, 0, 2))    
    predictions_map = resize_ski(predictions_map, (original_size[0], original_size[1], 1)).squeeze(axis=-1)    
    print("original predictions_map size", predictions_map.shape)    
    # H, W → W, H
    update_submission(predictions_map, index)
    plt.imsave(f"{index}.png", predictions_map, cmap="gray")

