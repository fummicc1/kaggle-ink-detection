#!/usr/bin/env python
# coding: utf-8

# # Keras starter kit [full training set, UNet]

# ## Setup

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torchvision
import albumentations as A
import pathlib
from albumentations.pytorch import ToTensorV2

import glob
import time
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from tqdm import tqdm

# Data config
# DATA_DIR = '/kaggle/input/vesuvius-challenge-ink-detection/'
DATA_DIR = "."
BUFFER = 128  # Half-size of papyrus patches we'll use as model inputs
Z_DIM = 16  # Number of slices in the z direction. Max value is 64 - Z_START
Z_START = 0  # Offset of slices in the z direction
SHARED_HEIGHT = 4000  # Height to resize all papyrii

# Model config
BATCH_SIZE = 32
USE_MIXED_PRECISION = False
USE_JIT_COMPILE = False


# In[ ]:


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )

        def transpose_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 2 else 64 * 2 ** (i - 1),
                        64 * 2**i,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(64 * 2**i),
                    nn.ReLU(),
                    nn.Conv2d(64 * 2**i, 64 * 2**i, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64 * 2**i),
                    nn.ReLU(),
                )
                for i in range(2, 4)
            ]
        )

        self.middle = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512),
        )

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    transpose_conv_block(2 ** (i + 7), 2 ** (i + 6)),
                    transpose_conv_block(2 ** (i + 6), 2 ** (i + 5)),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                )
                for i in range(3, 1, -1)
            ]
        )
        self.final_decoder = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        x = self.middle(x)

        # print("encoder ok", x.shape)
        for i, layer in enumerate(self.decoder):
            # print(f"decoder will {i}: ok", x.shape)
            x = torch.cat(
                [x, skip_connections[-i - 1]], dim=1
            )  # Concatenate along channel dimension
            # print(f"decoder with skip connection {i}: ok", x.shape)
            x = layer(x)
            # print(f"decoder {i}: ok", x.shape)
        # print("decoder ok")
        x = self.final_decoder(x)
        return x


# In[ ]:


device = torch.device("cuda")


# In[ ]:


model = UNet(Z_DIM, 2)
model = nn.DataParallel(model)
# model.load_state_dict(torch.load(f"/kaggle/input/ink-detection/model.pt"))
model.load_state_dict(torch.load(f"model.pt"))
model = model.to(device)


# ## Load up the training data

# In[ ]:


def resize(img):
    current_width, current_height = img.size
    aspect_ratio = current_width / current_height
    new_width = int(SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, SHARED_HEIGHT)
    img = img.resize(new_size)
    return img


def is_in_masked_zone(location, mask):
    return mask[location[0], location[1]]


def load_mask(split, index):
    img = Image.open(f"{DATA_DIR}/{split}/{index}/mask.png").convert("1")
    img = resize(img)
    return torch.tensor(np.array(img), dtype=torch.bool)


def load_labels(split, index):
    img = Image.open(f"{DATA_DIR}/{split}/{index}/inklabels.png").convert("1")
    img = resize(img)
    return torch.tensor(np.array(img), dtype=torch.bool)


def load_volume(split, index):
    # Load the 3d x-ray scan, one slice at a time
    z_slices_fnames = sorted(
        glob.glob(f"{DATA_DIR}/{split}/{index}/surface_volume/*.tif")
    )[Z_START : Z_START + Z_DIM]
    z_slices = []
    for z, filename in tqdm(enumerate(z_slices_fnames)):
        img = Image.open(filename).convert("1")
        img = resize(img)
        z_slice = np.array(img, dtype="float32")
        z_slices.append(z_slice)
    return np.stack(z_slices, axis=-1)


# In[ ]:


def extract_subvolume(location, volume):
    x = location[0]
    y = location[1]
    subvolume = volume[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :]
    subvolume = torch.from_numpy(subvolume).float() / 65535.0
    return subvolume


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


def compute_predictions_map(split, index):
    print(f"Load data for {split}/{index}")

    test_volume = load_volume(split=split, index=index)
    test_mask = load_mask(split=split, index=index)

    test_locations = []
    stride = BUFFER // 2
    for x in range(BUFFER, test_volume.shape[0] - BUFFER, stride):
        for y in range(BUFFER, test_volume.shape[1] - BUFFER, stride):
            test_locations.append((x, y))

    print(f"{len(test_locations)} test locations (before filtering by mask)")

    # filter locations inside the mask
    test_locations = [
        loc for loc in test_locations if is_in_masked_zone(loc, test_mask)
    ]

    class TestDataset(Dataset):
        def __init__(self, test_locations, test_volume):
            self.test_locations = test_locations
            self.test_volume = test_volume

        def __len__(self):
            return len(self.test_locations)

        def __getitem__(self, idx):
            location = torch.tensor(self.test_locations[idx])
            subvolume = extract_subvolume(location, self.test_volume)
            subvolume = torch.permute(subvolume, (2, 1, 0))
            return location, subvolume

    test_ds = TestDataset(test_locations, test_volume)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    predictions_map = np.zeros(test_volume.shape[:2] + (1,), dtype=np.float32)
    predictions_map_counts = np.zeros(test_volume.shape[:2] + (1,), dtype=np.float32)

    print(f"Compute predictions")

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for loc_batch, patch_batch in tqdm(test_loader):
            loc_batch = loc_batch.to(device)
            patch_batch = patch_batch.to(device)
            predictions = model(patch_batch)
            predictions = nn.Softmax2d()(predictions)
            predictions = torch.amax(predictions, dim=1).unsqueeze(dim=1)
            predictions = torch.permute(predictions, (0, 3, 2, 1))
            predictions = (
                predictions.cpu().numpy()
            )  # move predictions to cpu and convert to numpy
            for (x, y), pred in zip(loc_batch, predictions):
                predictions_map[
                    x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
                ] += pred
                predictions_map_counts[
                    x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
                ] += 1
    predictions_map /= predictions_map_counts + 1e-7
    return predictions_map


# In[ ]:


from skimage.transform import resize as resize_ski


# In[ ]:


def rle(predictions_map, threshold):
    flat_img = (np.where(predictions_map.flatten() > threshold, 1, 0)).astype(np.uint8)

    starts = np.where((flat_img[:-1] == 0) & (flat_img[1:] == 1))[0] + 2
    ends = np.where((flat_img[:-1] == 1) & (flat_img[1:] == 0))[0] + 2

    lengths = ends - starts

    return " ".join(map(str, np.c_[starts, lengths].flatten()))


# In[ ]:


# print("Id,Predicted\n", file=open('/kaggle/working/submission.csv', 'w'))
print("Id,Predicted", file=open("submission.csv", "w"))


def update_submission(predictions_map, index):
    threshold = 0.86
    rle_ = rle(predictions_map, threshold=threshold)
    # print(f"{index}," + rle_ + "\n", file=open('/kaggle/working/submission.csv', 'a'))
    print(f"{index}," + rle_, file=open("submission.csv", "a"))


# In[ ]:


folder = pathlib.Path(DATA_DIR) / "test"
for p in folder.iterdir():
    index = p.stem
    predictions_map = compute_predictions_map(split="test", index=index)
    print("compute_predictions_map end")
    original_size = Image.open(DATA_DIR + f"/test/{index}/mask.png").size
    print("original_size end")
    predictions_map = resize_ski(
        predictions_map, (original_size[1], original_size[0])
    ).squeeze()
    # predictions_map.resize((original_size[1], original_size[0]))
    print("resize_ski end")
    update_submission(predictions_map, index)
    print("update_submission end")
