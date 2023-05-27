#!/usr/bin/env python
# coding: utf-8

# # Keras starter kit [full training set, UNet]

# In[1]:


# import sys

# sys.path.append('/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4')
# sys.path.append('/kaggle/input/segmentation-models-pytorch/segmentation_models.pytorch-master')
# sys.path.append('/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
# sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')


# ## Setup

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torchvision
import cupy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import pytorch_lightning.plugins
from skimage.transform import resize as resize_ski
from pytorch_lightning.strategies.ddp import DDPStrategy
import os

from scipy.ndimage import distance_transform_edt

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import glob
import time
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


# Data config
# DATA_DIR = '/kaggle/input/vesuvius-challenge-ink-detection/'
DATA_DIR = "/home/fummicc1/codes/competitions/kaggle-ink-detection"
BUFFER = 112  # Half-size of papyrus patches we'll use as model inputs
Z_LIST = list(range(24, 36))  # Offset of slices in the z direction
Z_DIM = len(Z_LIST)  # Number of slices in the z direction. Max value is 64 - Z_START
SHARED_HEIGHT = 4000  # Max length(width or height) to resize all papyrii

# Model config
BATCH_SIZE = 96

# backbone = "mit_b1"
# backbone = "efficientnet-b5"
backbone = "se_resnext50_32x4d"
# backbone = "resnext50_32x4d"
# backbone = "resnet50"

device = torch.device("cuda")
threshold = 0.45
num_workers = 8
exp = 1e-7
mask_padding = 200

num_epochs = 30
lr = 1e-3

pytorch_lightning.seed_everything(seed=42)
torch.set_float32_matmul_precision("high")


# In[3]:


plt.imshow(Image.open(DATA_DIR + "/train/1/ir.png"), cmap="gray")


# In[4]:


# input shape: (H, W, C)
def rotate90(volume: np.ndarray, reverse=False):
    volume = np.rot90(volume, 1 if not reverse else 3)
    height = volume.shape[0]
    width = volume.shape[1]
    new_height = SHARED_HEIGHT
    new_width = int(new_height * width / height)
    if len(volume.shape) == 2:
        return cv2.resize(volume, (new_width, new_height))
    return resize_ski(volume, (new_height, new_width, volume.shape[2]))


# In[5]:


import cupy as cp

xp = cp

delta_lookup = {
    "xx": xp.array([[1, -2, 1]], dtype=float),
    "yy": xp.array([[1], [-2], [1]], dtype=float),
    "xy": xp.array([[1, -1], [-1, 1]], dtype=float),
}


def operate_derivative(img_shape, pair):
    assert len(img_shape) == 2
    delta = delta_lookup[pair]
    fft = xp.fft.fftn(delta, img_shape)
    return fft * xp.conj(fft)


def soft_threshold(vector, threshold):
    return xp.sign(vector) * xp.maximum(xp.abs(vector) - threshold, 0)


def back_diff(input_image, dim):
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r + 1, n + 1), dtype=float)
    temp2 = xp.zeros((r + 1, n + 1), dtype=float)

    temp1[position[0] : size[0], position[1] : size[1]] = input_image
    temp2[position[0] : size[0], position[1] : size[1]] = input_image

    size[dim] += 1
    position[dim] += 1
    temp2[position[0] : size[0], position[1] : size[1]] = input_image
    temp1 -= temp2
    size[dim] -= 1
    return temp1[0 : size[0], 0 : size[1]]


def forward_diff(input_image, dim):
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r + 1, n + 1), dtype=float)
    temp2 = xp.zeros((r + 1, n + 1), dtype=float)

    size[dim] += 1
    position[dim] += 1

    temp1[position[0] : size[0], position[1] : size[1]] = input_image
    temp2[position[0] : size[0], position[1] : size[1]] = input_image

    size[dim] -= 1
    temp2[0 : size[0], 0 : size[1]] = input_image
    temp1 -= temp2
    size[dim] += 1
    return -temp1[position[0] : size[0], position[1] : size[1]]


def iter_deriv(input_image, b, scale, mu, dim1, dim2):
    g = back_diff(forward_diff(input_image, dim1), dim2)
    d = soft_threshold(g + b, 1 / mu)
    b = b + (g - d)
    L = scale * back_diff(forward_diff(d - b, dim2), dim1)
    return L, b


def iter_xx(*args):
    return iter_deriv(*args, dim1=1, dim2=1)


def iter_yy(*args):
    return iter_deriv(*args, dim1=0, dim2=0)


def iter_xy(*args):
    return iter_deriv(*args, dim1=0, dim2=1)


def iter_sparse(input_image, bsparse, scale, mu):
    d = soft_threshold(input_image + bsparse, 1 / mu)
    bsparse = bsparse + (input_image - d)
    Lsparse = scale * (d - bsparse)
    return Lsparse, bsparse


def denoise_image(
    input_image,
    iter_num=100,
    fidelity=150,
    sparsity_scale=10,
    continuity_scale=0.5,
    mu=1,
):
    image_size = xp.shape(input_image)
    # print("Initialize denoising")
    print("input_size", input_image.shape)
    print("image_size", image_size)
    norm_array = (
        operate_derivative(image_size, "xx")
        + operate_derivative(image_size, "yy")
        + 2 * operate_derivative(image_size, "xy")
    )
    norm_array += (fidelity / mu) + sparsity_scale ** 2
    b_arrays = {
        "xx": xp.zeros(image_size, dtype=float),
        "yy": xp.zeros(image_size, dtype=float),
        "xy": xp.zeros(image_size, dtype=float),
        "L1": xp.zeros(image_size, dtype=float),
    }
    g_update = xp.multiply(fidelity / mu, input_image)
    for i in tqdm(range(iter_num), total=iter_num):
        # print(f"Starting iteration {i+1}")
        g_update = xp.fft.fftn(g_update)
        if i == 0:
            g = xp.fft.ifftn(g_update / (fidelity / mu)).real
        else:
            g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
        g_update = xp.multiply((fidelity / mu), input_image)

        # print("XX update")
        L, b_arrays["xx"] = iter_xx(g, b_arrays["xx"], continuity_scale, mu)
        g_update += L

        # print("YY update")
        L, b_arrays["yy"] = iter_yy(g, b_arrays["yy"], continuity_scale, mu)
        g_update += L

        # print("XY update")
        L, b_arrays["xy"] = iter_xy(g, b_arrays["xy"], 2 * continuity_scale, mu)
        g_update += L

        # print("L1 update")
        L, b_arrays["L1"] = iter_sparse(g, b_arrays["L1"], sparsity_scale, mu)
        g_update += L

    g_update = xp.fft.fftn(g_update)
    g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real

    g[g < 0] = 0
    g -= g.min()
    g /= g.max()
    return g


# In[6]:


def is_in_masked_zone(location, mask):
    return mask[location[0], location[1]] > 0


# In[7]:


def resize(img):
    current_height, current_width = img.shape
    aspect_ratio = current_width / current_height
    new_height = SHARED_HEIGHT
    new_width = int(SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, new_height)
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


# In[8]:


def load_volume(split, index):
    # Load the 3d x-ray scan, one slice at a time
    all = sorted(glob.glob(f"{DATA_DIR}/{split}/{index}/surface_volume/*.tif"))
    z_slices_fnames = [all[i] for i in range(len(all)) if i in Z_LIST]
    assert len(z_slices_fnames) == Z_DIM
    z_slices = []
    for z, filename in tqdm(enumerate(z_slices_fnames)):
        img = cv2.imread(filename, -1)
        img = resize(img)
        # img = (img / (2 ** 8)).astype(np.uint8)
        img = img.astype(np.float32) // 255
        z_slices.append(img)
    return np.stack(z_slices, axis=-1)


# In[9]:


static_all_median = np.array(
    [85.0, 87.0, 88.0, 88.0, 88.0, 86.0, 82.0, 78.0, 74.0, 71.0, 69.0, 69.0]
)


# In[10]:


static_all_MAD = np.array(
    [53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 58.0, 57.0, 53.0, 47.0, 40.0, 33.0]
)


# In[11]:


all_median = static_all_median
all_MAD = static_all_MAD


# In[12]:


printed = True


def extract_subvolume(location, volume):
    global printed
    # print(np.unique(volume, return_counts=True, return_index=True))
    x = location[0]
    y = location[1]
    subvolume = volume[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :].astype(
        np.float32
    )
    # print("subvolume[:, :, 0]", subvolume[:, :, 0])
    # median = np.full_like(subvolume, all_median).astype(np.float32)
    # MAD = np.full_like(subvolume, all_MAD).astype(np.float32)
    # mean = np.mean(subvolume, axis=2)
    # mean = np.stack([mean for i in range(Z_DIM)], axis=2) + exp
    # MAD = median_abs_deviation(subvolume, axis=2)
    # print("MAD", MAD[0, 0, :])
    # print("mean", mean)
    # print("median", median[0, 0, :])

    # subvolume = (subvolume - median) / MAD
    # subvolume = subvolume / median

    if not printed:
        print("subvolume after taking care of median and MAD", subvolume)
        printed = True

    return subvolume


# In[13]:


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

from albumentations.core.transforms_interface import ImageOnlyTransform


class NormalizeTransform(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(NormalizeTransform, self).__init__(always_apply, p)

    def apply(self, img, **params):
        median = np.full_like(img, all_median).astype(np.float32)
        mad = np.full_like(img, all_MAD).astype(np.float32)
        # img = (img - median) / mad
        img = img / median
        # img[img < 0] = 0
        return img


class SubvolumeDataset(Dataset):
    def __init__(
        self,
        locations,
        volume,
        labels,
        buffer,
        is_train: bool,
        return_location: bool = False,
    ):
        self.locations = locations
        self.volume = volume
        self.labels = labels
        self.buffer = buffer
        self.is_train = is_train
        self.return_location = return_location

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        global possible_min_input, possible_max_input
        label = None
        location = np.array(self.locations[idx])
        y, x = location[0], location[1]

        subvolume = extract_subvolume(location, self.volume)
        # print("subvolume", subvolume)
        # print("labels", labels)
        # subvolume = subvolume.numpy()
        subvolume = subvolume

        if self.labels is not None:
            label = self.labels[
                y - self.buffer : y + self.buffer, x - self.buffer : x + self.buffer
            ]
            # print("label", label)
            # n_category = 2
            # label = np.eye(n_category)[label]
            label = np.stack([label], axis=-1)
            # label = label.numpy()
            # print("label.shape", label.shape

        if self.is_train and label is not None:

            # print("label", label.dtype)
            # print("subvolume in train dataset (before aug)", subvolume, file=open("before-train-aug.log", "w"))
            size = int(BUFFER * 2)
            performed = A.Compose(
                [
                    # A.ToFloat(max_value=possible_max_input - possible_min_input),
                    # A.ToFloat(max_value=2**16-1),
                    NormalizeTransform(always_apply=True),
                    A.HorizontalFlip(p=0.5),  # 水平方向に反転
                    A.VerticalFlip(p=0.5),  # 水平方向に反転
                    A.RandomRotate90(p=0.5),
                    # A.RandomBrightnessContrast(p=0.4),
                    A.ShiftScaleRotate(p=0.5, border_mode=0),  # シフト、スケーリング、回転
                    # A.PadIfNeeded(min_height=size, min_width=size, always_apply=True, border_mode=0), # 必要に応じてパディングを追加
                    A.RandomCrop(
                        height=int(size / 1.25), width=int(size / 1.25), p=0.5
                    ),  # ランダムにクロップ, Moduleの中で計算する際に次元がバッチ内で揃っている必要があるので最後にサイズは揃える
                    # A.Perspective(p=0.5), # パースペクティブ変換
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    A.CoarseDropout(
                        max_holes=1,
                        max_width=int(size * 0.1),
                        max_height=int(size * 0.1),
                        mask_fill_value=0,
                        p=0.2,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=[10, 30]),
                            A.GaussianBlur(blur_limit=(3, 5)),
                            A.MotionBlur(blur_limit=5),
                        ],
                        p=0.4,
                    ),
                    A.Resize(BUFFER * 2, BUFFER * 2, always_apply=True),
                    # A.Normalize(
                    #     mean= [0] * Z_DIM,
                    #     std= [1] * Z_DIM
                    # ),
                    # A.FromFloat(max_value=possible_max_input - possible_min_input),
                    ToTensorV2(transpose_mask=True),
                ]
            )(image=subvolume, mask=label)
            subvolume = performed["image"]
            label = performed["mask"]
            # print("subvolume in train dataset (after aug)", subvolume, file=open("after-train-aug.log", "w"))
            # print("label", label.dtype)
            # print("subvolume", subvolume.dtype)
            # →C, H, W
            # subvolume = torch.from_numpy(subvolume.transpose(2, 0, 1).astype(np.float32))
            # print(performed)
            # print(subvolume.shape, label.shape)
            # H, W, C → C, H, W
            # label = torch.from_numpy(label.transpose(2, 0, 1).astype(np.uint8))
        else:
            if label is None:
                performed = A.Compose(
                    [
                        # A.ToFloat(max_value=possible_max_input - possible_min_input),
                        # A.ToFloat(max_value=2**16-1),
                        # A.Normalize(
                        #     mean= [0] * Z_DIM,
                        #     std= [1] * Z_DIM
                        # ),
                        # A.FromFloat(max_value=possible_max_input - possible_min_input),
                        NormalizeTransform(always_apply=True),
                        ToTensorV2(transpose_mask=True),
                    ]
                )(image=subvolume)
                subvolume = performed["image"]
            else:
                # print("subvolume in val dataset (before aug)", subvolume, file=open("before-val-aug.log", "w"))
                performed = A.Compose(
                    [
                        # A.ToFloat(max_value=possible_max_input - possible_min_input),
                        # A.ToFloat(max_value=2**16-1),
                        # A.Normalize(
                        #     mean= [0] * Z_DIM,
                        #     std= [1] * Z_DIM
                        # ),
                        # A.FromFloat(max_value=possible_max_input - possible_min_input),
                        NormalizeTransform(always_apply=True),
                        ToTensorV2(transpose_mask=True),
                    ]
                )(image=subvolume, mask=label)
                label = performed["mask"]
                subvolume = performed["image"]
                # print("subvolume in val dataset (after aug)", subvolume, file=open("after-val-aug.log", "w"))
            # subvolume = torch.from_numpy(subvolume.transpose(2, 0, 1).astype(np.float32))
            # if label is not None:
            # label = torch.from_numpy(label.transpose(2, 0, 1).astype(np.uint8))
        if self.return_location:
            return subvolume, location
        return subvolume, label


# In[14]:


def dice_coef_torch(prob_preds, targets, beta=0.5, smooth=1e-5):
    # No need to binarize the predictions
    # prob_preds = torch.sigmoid(preds)

    # flatten label and prediction tensors
    prob_preds = prob_preds.view(-1).float()
    targets = targets.view(-1).float()

    intersection = (prob_preds * targets).sum()

    dice = ((1 + beta ** 2) * intersection + smooth) / (
        (beta ** 2) * prob_preds.sum() + targets.sum() + smooth
    )

    return dice


class Model(pl.LightningModule):

    training_step_outputs = []
    validation_step_outputs = []
    test_step_outputs = [[], []]

    def __init__(self, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        # self.model = smp.UnetPlusPlus(
        #     encoder_name=encoder_name,
        #     # encoder_weights="imagenet",
        #     encoder_weights="swsl",
        #     # encoder_weights=None,
        #     encoder_depth=5,
        #     decoder_channels=[512, 256, 128, 64, 32],
        #     in_channels=in_channels,
        #     classes=out_classes,
        #     # aux_params={
        #     #     "pooling": "max",
        #     #     "classes": out_classes,
        #     #     "dropout": 0.2,
        #     #     "activation": None,
        #     # },
        #     **kwargs,
        # )
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            # encoder_weights="imagenet",
            encoder_weights="imagenet",
            # encoder_weights=None,
            encoder_depth=5,
            decoder_channels=[512, 256, 128, 64, 32],
            in_channels=in_channels,
            classes=out_classes,
            # aux_params={
            #     "pooling": "max",
            #     "classes": out_classes,
            #     "dropout": 0.2,
            #     "activation": None,
            # },
            **kwargs,
        )

        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.segmentation_loss_fn = smp.losses.TverskyLoss(
            smp.losses.BINARY_MODE, log_loss=False, from_logits=True, smooth=1e-6,
        )
        # smp.losses.FocalLoss()
        # self.segmentation_loss_fn = smp.losses.DiceLoss(
        #     smp.losses.BINARY_MODE,
        #     log_loss=False,
        #     from_logits=True,
        #     smooth=1e-6
        # )
        # self.segmentation_loss_fn = dice_coef_torch
        # self.classification_loss_fn = smp.losses.SoftCrossEntropyLoss()

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        subvolumes, labels = batch

        image, labels = subvolumes.float(), labels.float()
        # print("torch.unique(subvolumes)", torch.unique(subvolumes), file=open("subvolumes_unique", "w"))

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert labels.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert labels.max() <= 1.0 and labels.min() >= 0

        segmentation_out = self.forward(image)
        # print("model out", segmentation_out)
        segmentation_out = segmentation_out.sigmoid()

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.segmentation_loss_fn(segmentation_out, labels)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = segmentation_out
        pred_mask = (prob_mask > threshold).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), labels.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.sum(torch.Tensor([x["loss"] for x in outputs]))

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_loss": loss,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        out = self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()
        return out

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        out = self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return out

    def test_step(self, batch, batch_idx):
        global predictions_map, predictions_map_counts

        patch_batch, loc_batch = batch

        loc_batch = loc_batch.long()
        patch_batch = patch_batch.float()
        predictions: torch.Tensor = self.forward(patch_batch)
        # print("predictions.shape", predictions.shape)
        # print("predictions", predictions)
        predictions = predictions.sigmoid()
        # print("Softmaxed predictions where conf is gt threshold", predictions[predictions.gt(threshold)])
        # print("predictions.shape after sigmoid", predictions.shape)
        # →(BATCH, W, H, C)
        predictions = torch.permute(predictions, (0, 3, 2, 1))
        predictions = (
            predictions.cpu().numpy()
        )  # move predictions to cpu and convert to numpy
        loc_batch = loc_batch.cpu().numpy()
        # print("predictions_map", predictions_map)
        # print("predictions_map_count", predictions_map_counts)
        self.test_step_outputs[0].extend(loc_batch)
        self.test_step_outputs[1].extend(predictions)
        return loc_batch, predictions

    def on_test_epoch_end(self):
        global predictions_map, predictions_map_counts

        locs = np.array(self.test_step_outputs[0])
        preds = np.array(self.test_step_outputs[1])
        print("locs", locs.shape)
        print("preds", preds.shape)

        new_predictions_map = np.zeros_like(predictions_map[:, :, 0])[:, :, np.newaxis]
        new_predictions_map_counts = np.zeros_like(predictions_map_counts[:, :, 0])[
            :, :, np.newaxis
        ]

        for (y, x), pred in zip(locs, preds):
            # print("index: ", index ,"x, y, pred", x.item(), y.item(), pred[BUFFER, BUFFER, :].item(), file=open('log.out', 'a'))
            # print("pred", pred)
            # predictions_map[
            #     x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
            # ] = np.where(predictions_map[
            #     x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
            # ] < pred, pred, predictions_map[
            #     x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
            # ])
            new_predictions_map[
                x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
            ] += pred
            new_predictions_map_counts[
                x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :
            ] += 1
        new_predictions_map /= new_predictions_map_counts + exp
        new_predictions_map = xp.asarray(new_predictions_map[:, :, 0])
        new_predictions_map: xp.ndarray = denoise_image(
            new_predictions_map, iter_num=250
        )
        new_predictions_map = new_predictions_map.get()[:, :, np.newaxis]
        predictions_map = np.concatenate(
            [predictions_map, new_predictions_map], axis=-1
        )
        print("new_predictions_map", new_predictions_map.shape)
        print("predictions_map", predictions_map.shape)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.05, patience=5, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_dataset_iou"},
        }


# In[15]:


class EnsembleModel:
    def __init__(self, test_loader, test_volume):
        super().__init__()
        self.test_loader = test_loader
        self.test_volume = test_volume
        self.list = []
        for fold in [1, 2, 3]:
            _model = Model.load_from_checkpoint(
                f"weights/weights_fold-{fold}.ckpt",
                # f"/kaggle/input/first-ink-detection/weights_fold-{fold}.ckpt",
                encoder_name=backbone,
                in_channels=Z_DIM,
                out_classes=1,
            )
            trainer = pl.Trainer(accelerator="gpu", devices="1", max_epochs=num_epochs,)

            self.list.append((_model, trainer))

    def forward(self):
        global predictions_map, predictions_map_counts, all_median
        predictions_map = (
            np.empty_like(self.test_volume[:, :, 0])
            .transpose((1, 0))[:, :, np.newaxis]
            .astype(np.float64)
        )
        predictions_map_counts = np.empty_like(predictions_map).astype(np.uint8)
        for i, (model, trainer) in enumerate(self.list):
            all_median = static_all_median[i]
            model.test_step_outputs = [[], []]
            # shape: (X, Y, C)
            model.eval()
            trainer.test(
                model=model, dataloaders=self.test_loader, verbose=True,
            )
        predictions_map = predictions_map[:, :, 1:].mean(axis=-1)[:, :, np.newaxis]


# In[16]:


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from skimage.transform import resize as resize_ski
import pathlib

predictions_map = None
predictions_map_counts = None


def compute_predictions_map(split, index):
    global predictions_map
    global predictions_map_counts

    print(f"Load data for {split}/{index}")

    test_volume = rotate90(load_volume(split=split, index=index))
    test_mask = rotate90(load_mask(split=split, index=index))

    test_locations = []
    stride = BUFFER // 2
    for y in range(BUFFER, test_volume.shape[0] - BUFFER, stride):
        for x in range(BUFFER, test_volume.shape[1] - BUFFER, stride):
            test_locations.append((y, x))

    print(f"{len(test_locations)} test locations (before filtering by mask)")

    # filter locations inside the mask
    test_locations = [
        loc for loc in test_locations if is_in_masked_zone(loc, test_mask)
    ]

    print(f"{len(test_locations)} test locations (after filtering by mask)")

    test_ds = SubvolumeDataset(
        test_locations, test_volume, None, BUFFER, is_train=False, return_location=True
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=num_workers)

    # # shape: (X, Y, C)
    # predictions_map = np.zeros_like(test_volume[:, :, 0]).transpose((1, 0))[:, :, np.newaxis].astype(np.float64)
    # predictions_map_counts = np.zeros_like(predictions_map).astype(np.uint8)

    # print("test_volume.shape", test_volume.shape)
    # print("predictions_map.shape", predictions_map.shape)

    # print(f"Compute predictions")

    model = EnsembleModel(test_loader, test_volume)
    model.forward()

    # print("predictions_map", predictions_map, file=open("predictions_map", "w"))
    return predictions_map


# In[17]:


def rle(predictions_map, threshold):
    flat_img = (np.where(predictions_map.flatten() >= threshold, 1, 0)).astype(np.uint8)

    # Add padding at the beginning and end
    flat_img = np.pad(flat_img, pad_width=1, mode="constant", constant_values=0)

    starts = np.where((flat_img[:-1] == 0) & (flat_img[1:] == 1))[0]
    ends = np.where((flat_img[:-1] == 1) & (flat_img[1:] == 0))[0]

    lengths = ends - starts

    return " ".join(map(str, np.c_[starts, lengths].flatten()))


# In[18]:


def update_submission(predictions_map, index):
    rle_ = rle(predictions_map, threshold=threshold)
    print(f"{index}," + rle_, file=open("submission.csv", "a"))
    # print(f"{index}," + rle_, file=open('/kaggle/working/submission.csv', 'a'))


# In[19]:


print("Id,Predicted", file=open("submission.csv", "w"))
kind = "test"
folder = pathlib.Path(DATA_DIR) / kind
for p in list(folder.iterdir()):
    index = p.stem
    predictions_map = compute_predictions_map(split=kind, index=index)
    original_size = cv2.imread(DATA_DIR + f"/{kind}/{index}/mask.png", 0).shape[:2]
    # W, H, C → H, W, C
    predictions_map = predictions_map.transpose((1, 0, 2))
    predictions_map = rotate90(predictions_map, reverse=True)
    predictions_map = resize_ski(
        predictions_map, (original_size[0], original_size[1], 1)
    ).squeeze(axis=-1)
    print("original predictions_map size", predictions_map.shape)
    # H, W → W, H
    update_submission(predictions_map, index)
    predictions_map = np.where(predictions_map >= threshold, 1, 0)
    plt.imsave(
        f"{index}_{str(threshold)}_{str(BUFFER)}.png", predictions_map, cmap="gray"
    )
