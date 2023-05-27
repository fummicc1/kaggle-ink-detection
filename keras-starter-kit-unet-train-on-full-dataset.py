#!/usr/bin/env python
# coding: utf-8

# # Keras starter kit [full training set, UNet]

# ## Setup

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torchvision
import datetime

# import cupy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import pytorch_lightning.plugins
from skimage.transform import resize as resize_ski
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from timm.models.resnet import resnet10t, resnet34d, resnet50d
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


class CFG:
    # Data config
    # DATA_DIR = '/kaggle/input/vesuvius-challenge-ink-detection/'
    # DATA_DIR = '/home/fummicc1/codes/competitions/kaggle-ink-detection'
    DATA_DIR = "/home/fummicc1/codes/Kaggle/kaggle-ink-detection"
    BUFFER = 160  # Half-size of papyrus patches we'll use as model inputs
    # Z_LIST = list(range(0, 20, 5)) + list(range(22, 34))  # Offset of slices in the z direction
    Z_LIST = list(range(28, 37))
    # Z_LIST = list(range(0, 24, 8)) + list(range(24, 36, 2)) + list(range(36, 64, 10))
    Z_DIM = len(
        Z_LIST
    )  # Number of slices in the z direction. Max value is 64 - Z_START
    SHARED_HEIGHT = 4800  # Max height to resize all papyrii

    # Model config
    BATCH_SIZE = 36

    # backbone = "mit_b2"
    # backbone = "efficientnet-b5"
    backbone = "se_resnext50_32x4d"
    # backbone = "resnext50_32x4d"
    # backbone = "resnet50"

    device = torch.device("cuda")
    threshold = 0.5
    num_workers = 8
    exp = 1e-7
    mask_padding = 200

    num_epochs = 20
    lr = 5e-4


# ## Load up the training data

# In[2]:


def resize(img):
    current_height, current_width = img.shape
    aspect_ratio = current_width / current_height
    if CFG.SHARED_HEIGHT is None:
        return img
    new_height = CFG.SHARED_HEIGHT
    new_width = int(CFG.SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, new_height)
    # (W, H)の順で渡すが結果は(H, W)になっている
    img = cv2.resize(img, new_size)
    return img


def load_mask(split, index):
    if index == "2a" or index == "2b":
        mode = index[1]
        index = "2"
    img = cv2.imread(f"{CFG.DATA_DIR}/{split}/{index}/mask.png", 0) // 255
    if index == "2":
        h = 9456
        if mode == "a":
            img = img[h:, :]
        elif mode == "b":
            img = img[:h, :]
    img = np.pad(img, 1, constant_values=0)
    dist = distance_transform_edt(img)
    img[dist <= CFG.mask_padding] = 0
    img = img[1:-1, 1:-1]
    img = resize(img)
    return img


def load_labels(split, index):
    if index == "2a" or index == "2b":
        mode = index[1]
        index = "2"
    img = cv2.imread(f"{CFG.DATA_DIR}/{split}/{index}/inklabels.png", 0) // 255
    if index == "2":
        h = 9456
        if mode == "a":
            img = img[h:, :]
        elif mode == "b":
            img = img[:h, :]
    img = resize(img)
    return img


# In[3]:


# input shape: (H, W, C)
def rotate90(volume: np.ndarray, k=None, reverse=False):
    if k:
        volume = np.rot90(volume, k)
    else:
        volume = np.rot90(volume, 1 if not reverse else 3)
    height = volume.shape[0]
    width = volume.shape[1]
    new_height = CFG.SHARED_HEIGHT
    new_width = int(new_height * width / height)
    if len(volume.shape) == 2:
        return cv2.resize(volume, (new_width, new_height))
    return resize_ski(volume, (new_height, new_width, volume.shape[2]))


# In[4]:


def load_volume(split, index):
    if index == "2a" or index == "2b":
        mode = index[1]
        index = "2"
    # Load the 3d x-ray scan, one slice at a time
    all = sorted(glob.glob(f"{CFG.DATA_DIR}/{split}/{index}/surface_volume/*.tif"))
    z_slices_fnames = [all[i] for i in range(len(all)) if i in CFG.Z_LIST]
    assert len(z_slices_fnames) == CFG.Z_DIM
    z_slices = []
    for z, filename in tqdm(enumerate(z_slices_fnames)):
        img = cv2.imread(filename, -1)
        if index == "2":
            h = 9456
            if mode == "a":
                img = img[h:, :]
            elif mode == "b":
                img = img[:h, :]
        img = resize(img)
        # img = (img / (2 ** 8)).astype(np.uint8)
        img = img.astype(np.float32) // 255
        z_slices.append(img)
    return np.stack(z_slices, axis=-1)


# ## Create a dataset in the input volume
#

# In[5]:


def is_in_masked_zone(location, mask):
    return mask[location[0], location[1]] > 0


# In[6]:


def generate_locations_ds(volume, mask, label=None, skip_zero=False):
    is_in_mask_train = lambda x: is_in_masked_zone(x, mask)

    # Create a list to store train locations
    locations = []

    # Generate train locations
    volume_height, volume_width = volume.shape[:-1]

    for y in range(CFG.BUFFER, volume_height - CFG.BUFFER, int(CFG.BUFFER / 2)):
        for x in range(CFG.BUFFER, volume_width - CFG.BUFFER, int(CFG.BUFFER / 2)):
            if (
                skip_zero
                and label is not None
                and np.all(
                    label[
                        y - CFG.BUFFER : y + CFG.BUFFER, x - CFG.BUFFER : x + CFG.BUFFER
                    ]
                    == 0
                )
            ):
                continue
            if is_in_mask_train((y, x)):
                locations.append((y, x))

    # Convert the list of train locations to a PyTorch tensor
    locations_ds = np.stack(locations, axis=0)
    return locations_ds


# ## Visualize some training patches
#
# Sanity check visually that our patches are where they should be.

# In[7]:


def extract_subvolume(location, volume):
    global printed
    y = location[0]
    x = location[1]
    subvolume = volume[
        y - CFG.BUFFER : y + CFG.BUFFER, x - CFG.BUFFER : x + CFG.BUFFER, :
    ].astype(np.float32)

    return subvolume


# ## SubvolumeDataset

# In[8]:


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

from albumentations.core.transforms_interface import ImageOnlyTransform


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
        label = None
        location = np.array(self.locations[idx])
        y, x = location[0], location[1]

        subvolume = extract_subvolume(location, self.volume)

        if self.labels is not None:
            label = self.labels[
                y - self.buffer : y + self.buffer, x - self.buffer : x + self.buffer
            ]
            label = np.stack([label], axis=-1)

        if self.is_train and label is not None:
            transformed = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomScale(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(p=0.6),
                    # A.GridDistortion(p=0.1),
                    # A.CoarseDropout(
                    #     max_holes=1,
                    #     max_width=int(self.buffer * 0.3),
                    #     max_height=int(self.buffer * 0.3),
                    #     mask_fill_value=0,
                    #     p=0.5,
                    # ),
                    # A.GridDropout(p=0.15),
                    A.Resize(height=self.buffer * 2, width=self.buffer * 2),
                ]
            )(image=subvolume, mask=label)
            subvolume = transformed["image"]
            label = transformed["mask"]
            subvolume = np.transpose(subvolume, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))
            subvolume /= 255.0
            subvolume = (subvolume - 0.3) / 0.22
        else:
            if label is None:
                subvolume = np.transpose(subvolume, (2, 0, 1))
                subvolume /= 255.0
                subvolume = (subvolume - 0.3) / 0.22
            else:
                # print("subvolume in val dataset (before aug)", subvolume, file=open("before-val-aug.log", "w"))
                subvolume = np.transpose(subvolume, (2, 0, 1))
                label = np.transpose(label, (2, 0, 1))
                subvolume /= 255.0
                subvolume = (subvolume - 0.3) / 0.22
        if self.return_location:
            return subvolume, location
        return subvolume, label


# ## Visualize validation dataset patches
#
# Note that they are partially overlapping, since the stride is half the patch size.

# In[9]:


def visualize_dataset_patches(locations_ds, labels, mode: str, fold=0):
    fig, ax = plt.subplots()
    ax.imshow(labels)

    for y, x in locations_ds:
        patch = patches.Rectangle(
            [x - CFG.BUFFER, y - CFG.BUFFER],
            2 * CFG.BUFFER,
            2 * CFG.BUFFER,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(patch)
    plt.savefig(f"fold-{fold}-{mode}.png")
    plt.show()


# ## Compute a trivial baseline
#
# This is the highest validation score you can reach without looking at the inputs.
# The model can be considered to have statistical power only if it can beat this baseline.

# ## Dataset check

# ## Model

# In[10]:


# ref - https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
def fbeta_score(preds, targets, threshold, beta=0.5, smooth=1e-5):
    preds_t = torch.where(preds > threshold, 1.0, 0.0).float()
    y_true_count = targets.sum()

    ctp = preds_t[targets == 1].sum()
    cfp = preds_t[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (
        (1 + beta_squared)
        * (c_precision * c_recall)
        / (beta_squared * c_precision + c_recall + smooth)
    )

    return dice


# In[11]:


class SmpUnetDecoder(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [
            in_channel,
        ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            DecoderBlock(i, s, o, use_batchnorm=True, attention_type=None)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)

        last = d
        return last, decode


class Net(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.output_type = ["inference", "loss"]

        conv_dim = 64
        encoder1_dim = [
            conv_dim,
            64,
            128,
            256,
            512,
        ]
        decoder1_dim = [
            256,
            128,
            64,
            64,
        ]

        self.encoder1 = resnet34d(pretrained=False, in_chans=CFG.Z_DIM)

        self.decoder1 = SmpUnetDecoder(
            in_channel=encoder1_dim[-1],
            skip_channel=encoder1_dim[:-1][::-1],
            out_channel=decoder1_dim,
        )
        # -- pool attention weight
        self.weight1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for dim in encoder1_dim
            ]
        )
        self.logit1 = nn.Conv2d(decoder1_dim[-1], 1, kernel_size=1)

        # --------------------------------
        #
        encoder2_dim = [64, 128, 256, 512]  #
        decoder2_dim = [
            128,
            64,
            32,
        ]
        self.encoder2 = resnet10t(pretrained=False, in_chans=decoder1_dim[-1])

        self.decoder2 = SmpUnetDecoder(
            in_channel=encoder2_dim[-1],
            skip_channel=encoder2_dim[:-1][::-1],
            out_channel=decoder2_dim,
        )
        self.logit2 = nn.Conv2d(decoder2_dim[-1], 1, kernel_size=1)

    def forward(self, batch):
        v = batch
        B, C, H, W = v.shape
        vv = [v[:, i : CFG.Z_DIM] for i in range(0, 2, 4)]
        K = len(vv)
        x = torch.cat(vv, 0)
        # x = v

        # ----------------------
        encoder = []
        e = self.encoder1
        x = e.conv1(x)
        x = e.bn1(x)
        x = e.act1(x)
        encoder.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = e.layer1(x)
        encoder.append(x)
        x = e.layer2(x)
        encoder.append(x)
        x = e.layer3(x)
        encoder.append(x)
        x = e.layer4(x)
        encoder.append(x)
        # print('encoder', [f.shape for f in encoder])

        for i in range(len(encoder)):
            e = encoder[i]
            f = self.weight1[i](e)
            _, c, h, w = e.shape
            f = rearrange(f, "(K B) c h w -> B K c h w", K=K, B=B, h=h, w=w)  #
            e = rearrange(e, "(K B) c h w -> B K c h w", K=K, B=B, h=h, w=w)  #
            w = F.softmax(f, 1)
            e = (w * e).sum(1)
            encoder[i] = e

        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last, decoder = self.decoder1(feature, skip)
        logit1 = self.logit1(last)

        logit1 = F.interpolate(
            logit1, size=(H, W), mode="bilinear", align_corners=False, antialias=True
        )

        # ----------------------
        x = last  # .detach()
        # x = F.avg_pool2d(x,kernel_size=2,stride=2)
        encoder = []
        e = self.encoder2
        x = e.layer1(x)
        encoder.append(x)
        x = e.layer2(x)
        encoder.append(x)
        x = e.layer3(x)
        encoder.append(x)
        x = e.layer4(x)
        encoder.append(x)

        feature = encoder[-1]
        skip = encoder[:-1][::-1]
        last, decoder = self.decoder2(feature, skip)
        logit2 = self.logit2(last)
        logit2 = F.interpolate(
            logit2, size=(H, W), mode="bilinear", align_corners=False, antialias=True
        )
        return logit1, logit2


# In[12]:


def dice_coef_torch(prob_preds, targets, beta=0.5, smooth=1e-5):
    # No need to binarize the predictions
    # prob_preds = torch.sigmoid(preds)

    # flatten label and prediction tensors
    prob_preds = prob_preds.view(-1).float()
    targets = targets.view(-1).float()

    intersection = (prob_preds * targets).sum()

    dice = ((1 + beta**2) * intersection + smooth) / (
        (beta**2) * prob_preds.sum() + targets.sum() + smooth
    )

    return dice


class Model(pl.LightningModule):
    training_step_outputs = []
    validation_step_outputs = []
    test_step_outputs = [[], []]

    def __init__(self, **kwargs):
        super().__init__()

        self.model = Net()

        self.loss1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.7]))
        self.loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.7]))
        # self.segmentation_loss_fn = dice_coef_torch
        # self.classification_loss_fn = smp.losses.SoftCrossEntropyLoss()

    def forward(self, image, stage):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        subvolumes, labels = batch

        image, labels = subvolumes.float(), labels.float()
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # print("labels", labels.max(), labels.min())

        assert labels.max() <= 1.0 and labels.min() >= 0

        logit1, logit2 = self.forward(image, stage)

        # print("segmentation_out", segmentation_out.shape)

        loss = self.loss1(logit1, labels) + self.loss2(logit2, labels)

        prob2 = torch.sigmoid(logit2)

        pred_mask = (prob2 > CFG.threshold).float()

        # print("pred_mask", pred_mask)

        score = fbeta_score(pred_mask, labels, threshold=CFG.threshold)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), labels.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "score": score,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.mean(torch.Tensor([x["loss"] for x in outputs]))
        fbeta_score = torch.mean(torch.Tensor([x["score"] for x in outputs]))

        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_loss": loss.item(),
            f"{stage}_tp": tp.sum().int().item(),
            f"{stage}_fp": fp.sum().int().item(),
            f"{stage}_fn": fn.sum().int().item(),
            f"{stage}_tn": tn.sum().int().item(),
            f"{stage}_score": fbeta_score.item(),
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)

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
        predictions: torch.Tensor = self.forward(patch_batch, "test")

        predictions = torch.permute(predictions, (0, 2, 3, 1)).squeeze(dim=-1)
        predictions = predictions.cpu().numpy()
        loc_batch = loc_batch.cpu().numpy()

        self.test_step_outputs[0].extend(loc_batch)
        self.test_step_outputs[1].extend(predictions)
        return loc_batch, predictions

    def on_test_epoch_end(self):
        global predictions_map, predictions_map_counts

        locs = np.array(self.test_step_outputs[0])
        preds = np.array(self.test_step_outputs[1])
        print("locs", locs.shape)
        print("preds", preds.shape)

        new_predictions_map = np.zeros_like(predictions_map[:, :, 0])
        new_predictions_map_counts = np.zeros_like(predictions_map_counts[:, :, 0])

        for (y, x), pred in zip(locs, preds):
            new_predictions_map[
                y - CFG.BUFFER : y + CFG.BUFFER, x - CFG.BUFFER : x + CFG.BUFFER
            ] += pred
            new_predictions_map_counts[
                y - CFG.BUFFER : y + CFG.BUFFER, x - CFG.BUFFER : x + CFG.BUFFER
            ] += 1

        new_predictions_map /= new_predictions_map_counts + CFG.exp
        new_predictions_map = new_predictions_map[:, :, np.newaxis]
        predictions_map = np.concatenate(
            [predictions_map, new_predictions_map], axis=-1
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=CFG.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.75,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }


# In[13]:


if __name__ == "__main__":
    model = Model()
    model

    # # !export CUDA_LAUNCH_BLOCKING=1
    # # !export TORCH_USE_CUDA_DSA=1

    #

    # In[14]:

    pytorch_lightning.seed_everything(seed=42)
    torch.set_float32_matmul_precision("high")

    masks = load_mask(split="train", index=1)
    labels = load_labels(split="train", index=1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("mask.png")
    ax1.imshow(masks, cmap="gray")
    ax2.set_title("inklabels.png")
    ax2.imshow(labels, cmap="gray")
    plt.show()

    mask_test_a = load_mask(split="test", index="a")
    mask_test_b = load_mask(split="test", index="b")

    mask_train_1 = load_mask(split="train", index="1")
    labels_train_1 = load_labels(split="train", index="1")

    mask_train_2a = load_mask(split="train", index="2a")
    labels_train_2a = load_labels(split="train", index="2a")

    mask_train_2b = load_mask(split="train", index="2b")
    labels_train_2b = load_labels(split="train", index="2b")

    mask_train_3 = load_mask(split="train", index="3")
    labels_train_3 = load_labels(split="train", index="3")

    print(f"mask_test_a: {mask_test_a.shape}")
    print(f"mask_test_b: {mask_test_b.shape}")
    print("-")
    print(f"mask_train_1: {mask_train_1.shape}")
    print(f"labels_train_1: {labels_train_1.shape}")
    print("-")
    print(f"mask_train_2a: {mask_train_2a.shape}")
    print(f"labels_train_2a: {labels_train_2a.shape}")
    print("-")
    print(f"mask_train_2b: {mask_train_2b.shape}")
    print(f"labels_train_2b: {labels_train_2b.shape}")
    print("-")
    print(f"mask_train_3: {mask_train_3.shape}")
    print(f"labels_train_3: {labels_train_3.shape}")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    ax1.set_title("labels_train_1")
    ax1.imshow(labels_train_1, cmap="gray")

    ax2.set_title("labels_train_2a")
    ax2.imshow(labels_train_2a, cmap="gray")

    ax3.set_title("labels_train_2b")
    ax3.imshow(labels_train_2b, cmap="gray")

    ax4.set_title("labels_train_3")
    ax4.imshow(labels_train_3, cmap="gray")
    plt.tight_layout()
    plt.show()

    volume_train_1 = load_volume(split="train", index=1)
    print(f"volume_train_1: {volume_train_1.shape}, {volume_train_1.dtype}")

    volume_train_2a = load_volume(split="train", index="2a")
    print(f"volume_train_2a: {volume_train_2a.shape}, {volume_train_2a.dtype}")

    volume_train_2b = load_volume(split="train", index="2b")
    print(f"volume_train_2b: {volume_train_2b.shape}, {volume_train_2b.dtype}")

    volume_train_3 = load_volume(split="train", index=3)
    print(f"volume_train_3: {volume_train_3.shape}, {volume_train_3.dtype}")

    # volume = np.concatenate([volume_train_1, volume_train_2, volume_train_3], axis=1)
    # volume = np.concatenate([volume_train_1, volume_train_2], axis=1)
    # print(f"total volume: {volume.shape}")

    # In[15]:

    sample_locations = generate_locations_ds(
        volume_train_3, mask_train_3, labels_train_3, skip_zero=True
    )
    sample_ds = SubvolumeDataset(
        sample_locations,
        volume_train_3,
        labels_train_3,
        CFG.BUFFER,
        is_train=True,
    )

    img = sample_ds[148][0][0, :, :]
    plt.imshow(img)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.imshow(labels_train_1)

    y, x = sample_locations[150]
    patch = patches.Rectangle(
        [x - CFG.BUFFER, y - CFG.BUFFER],
        2 * CFG.BUFFER,
        2 * CFG.BUFFER,
        linewidth=2,
        edgecolor="g",
        facecolor="none",
    )
    ax.add_patch(patch)
    plt.show()

    fig, ax = plt.subplots(CFG.Z_DIM, 1, figsize=(12, 24))

    for i in range(CFG.Z_DIM):
        img, _ = sample_ds[148]
        img = img[i, :, :]
        ax[i].hist(img.flatten(), bins=1000)  # Plot histogram of the flattened data
        ax[i].set_title(f"Histogram of Channel {i}")  # Add title to the plot
    fig.tight_layout()
    fig.show()

    # In[16]:

    k_folds = 4
    kfold = KFold(n_splits=k_folds, shuffle=True)
    data_list = [
        (volume_train_2a, labels_train_2a, mask_train_2a),
        (volume_train_1, labels_train_1, mask_train_1),
        (volume_train_2b, labels_train_2b, mask_train_2b),
        (volume_train_3, labels_train_3, mask_train_3),
    ]

    for fold, (train_data, val_data) in enumerate(kfold.split(data_list)):
        print(f"FOLD {fold}")
        print("--------------------------------")
        print("train_data", train_data)
        print("val_data", val_data)
        one = data_list[train_data[0]]
        two = data_list[train_data[1]]
        three = data_list[train_data[2]]
        train_volume = np.concatenate([one[0], two[0], three[0]], axis=1)
        train_label = np.concatenate([one[1], two[1], three[1]], axis=1)
        train_mask = np.concatenate([one[2], two[2], three[2]], axis=1)
        val_volume, val_label, val_mask = data_list[val_data[0]]

        train_locations_ds = generate_locations_ds(
            train_volume, train_mask, train_label, skip_zero=True
        )
        val_location_ds = generate_locations_ds(val_volume, val_mask, skip_zero=False)

        visualize_dataset_patches(train_locations_ds, train_label, "train", fold)
        visualize_dataset_patches(val_location_ds, val_label, "val", fold)

        # Init the neural network
        model = Model()

        # Initialize a trainer
        trainer = pl.Trainer(
            max_epochs=CFG.num_epochs,
            devices="0,1,2",
            accelerator="gpu",
            # strategy="ddp_find_unused_parameters_false",
            # strategy="ddp_fork",
            logger=WandbLogger(name=f"2.5dimension-{datetime.datetime.now()}"),
        )

        # Sample elements randomly from a given list of ids, no replacement.
        train_ds = SubvolumeDataset(
            train_locations_ds, train_volume, train_label, CFG.BUFFER, is_train=True
        )
        val_ds = SubvolumeDataset(
            val_location_ds,
            val_volume,
            val_label,
            CFG.BUFFER,
            is_train=False,
        )

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=CFG.BATCH_SIZE,
            num_workers=CFG.num_workers,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=CFG.BATCH_SIZE,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)
