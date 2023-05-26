#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# for dicsussion, refer to https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/407972#2272189

# In[19]:

# In[ ]:


# import sys

# sys.path.append('/kaggle/input/ink-00/my_lib')
# sys.path.append('/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4')
# sys.path.append('/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
# sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
# sys.path.append('/kaggle/input/segmentation-models-pytorch/segmentation_models.pytorch-master')
# sys.path.append('/kaggle/input/einops/einops-master')


# In[20]:

# In[ ]:


import hashlib
import numpy as np
import pandas as pd
from dotdict import dotdict
from time import time


# In[ ]:


from collections import defaultdict
from glob import glob
import PIL.Image as Image

Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images


# In[ ]:


import cv2


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


from einops import rearrange, reduce, repeat
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from timm.models.resnet import resnet10t, resnet34d


# In[ ]:


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
import os


# In[ ]:


from scipy.ndimage import distance_transform_edt


# In[ ]:


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


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt


# In[21]:

# In[ ]:


def time_to_str(time):
    h = time // 3600
    m = (time % 3600) // 60
    s = time % 60
    return f"{h:02f}, {m:02f}, {s:02f}"


# In[22]:

# In[ ]:


class Config(object):
    mode = [
        "train",  #
        # 'test', 'skip_fake_test',
    ]
    crop_fade = 56
    crop_size = 384
    crop_depth = 5
    infer_fragment_z = [28, 37]
    threshold = 0.5
    lr = 1e-4
    batch_size = 32
    num_workers = 8
    epochs = 20


# In[ ]:


CFG = Config()
CFG.is_tta = True  # True


# In[ ]:


if "train" in CFG.mode:
    CFG.stride = CFG.crop_size // 2
if "test" in CFG.mode:
    CFG.stride = CFG.crop_size // 2


# In[ ]:


def cfg_to_text():
    d = Config.__dict__
    text = [
        f"\t{k} : {v}"
        for k, v in d.items()
        if not (k.startswith("__") and k.endswith("__"))
    ]
    d = CFG.__dict__
    text += [
        f"\t{k} : {v}"
        for k, v in d.items()
        if not (k.startswith("__") and k.endswith("__"))
    ]
    return "CFG\n" + "\n".join(text)


# In[ ]:

#  dataset ##

# --

# In[ ]:


def do_binarise(m, threshold=0.5):
    m = m - m.min()
    m = m / (m.max() + 1e-7)
    m = (m > threshold).astype(np.float32)
    return m


# In[ ]:


def read_data(fragment_id, z0=CFG.infer_fragment_z[0], z1=CFG.infer_fragment_z[1]):
    volume = []
    start_timer = time.time()
    for i in range(z0, z1):
        v = np.array(
            Image.open(f"{data_dir}/{fragment_id}/surface_volume/{i:02d}.tif"),
            dtype=np.uint16,
        )
        v = (v >> 8).astype(np.uint8)
        # v = (v / 65535.0 * 255).astype(np.uint8)
        volume.append(v)
        print(
            f"\r @ read_data(): volume{fragment_id}  {time_to_str(time.time() - start_timer)}",
            end="",
            flush=True,
        )
    # print('')
    volume = np.stack(volume, -1)
    height, width, depth = volume.shape
    # print(f'fragment_id={fragment_id} volume: {volume.shape}')

    # ---
    mask = cv2.imread(f"{data_dir}/{fragment_id}/mask.png", cv2.IMREAD_GRAYSCALE)
    mask = do_binarise(mask)

    if "train" in CFG.mode:
        ir = cv2.imread(f"{data_dir}/{fragment_id}/ir.png", cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(
            f"{data_dir}/{fragment_id}/inklabels.png", cv2.IMREAD_GRAYSCALE
        )
        ir = ir / 255
        label = do_binarise(label)

    if "test" in CFG.mode:
        ir = None
        label = None

    d = dotdict(
        fragment_id=fragment_id,
        volume=volume,
        ir=ir,
        label=label,
        mask=mask,
    )
    return d


# In[ ]:


def read_data1(fragment_id):
    if fragment_id == "2a":
        y = 9456
        d = read_data("2")
        d = dotdict(
            fragment_id="2a",
            volume=d.volume[:y],
            ir=d.ir[:y],
            label=d.label[:y],
            mask=d.mask[:y],
        )
    elif fragment_id == "2b":
        y = 9456
        d = read_data("2")
        d = dotdict(
            fragment_id="2b",
            volume=d.volume[y:],
            ir=d.ir[y:],
            label=d.label[y:],
            mask=d.mask[y:],
        )
    else:
        d = read_data(fragment_id)
    return d


# In[ ]:


def load_mask(split, index):
    img = cv2.imread(f"{data_dir}/{split}/{index}/mask.png", 0) // 255
    return img


def load_labels(split, index):
    img = cv2.imread(f"{data_dir}/{split}/{index}/inklabels.png", 0) // 255
    return img


# In[ ]:


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


# In[ ]:


def run_check_data():
    d = read_data1(valid_id[0])  # valid_id[0]
    print("")
    print("fragment_id:", d.fragment_id)
    print("volume:", d.volume.shape, d.volume.min(), d.volume.max())
    print("mask  :", d.mask.shape, d.mask.min(), d.mask.max())
    if "train" in CFG.mode:
        print("ir    :", d.ir.shape, d.ir.min(), d.ir.max())
        print("label :", d.label.shape, d.label.min(), d.label.max())


# un_check_data()

# ref - https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288

# In[ ]:

# In[ ]:


def extract(location, volume):
    global printed
    x = location[0]
    y = location[1]
    subvolume = volume[y : y + CFG.crop_size, x : x + CFG.crop_size, :].astype(
        np.float32
    )
    return subvolume


# In[ ]:

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


from albumentations.core.transforms_interface import ImageOnlyTransform


# In[ ]:


def run_check_net():
    height, width = CFG.crop_size, CFG.crop_size
    depth = CFG.infer_fragment_z[1] - CFG.infer_fragment_z[0]
    batch_size = 3


# In[ ]:


# x = np.arange(0,3)
# y = np.arange(0,4)
# x,y = np.meshgrid(x,y)
# xy  = np.stack([x,y],-1).reshape(-1,2)
# x, y, xy


# In[ ]:


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
        x, y = location[0], location[1]

        subvolume = extract(location, self.volume)

        if self.labels is not None:
            label = self.labels[y : y + self.buffer * 2, x : x + self.buffer * 2]
            label = np.stack([label], axis=-1)

        if self.is_train and label is not None:
            transformed = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    A.RandomScale(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(p=0.5),
                    A.Resize(height=self.buffer * 2, width=self.buffer * 2),
                ]
            )(image=subvolume, mask=label)
            subvolume = transformed["image"]
            label = transformed["mask"]
            subvolume = np.transpose(subvolume, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))
            subvolume /= 255.0
            subvolume = (subvolume - 0.45) / 0.225
        else:
            # print("subvolume in val dataset (before aug)", subvolume, file=open("before-val-aug.log", "w"))
            subvolume = np.transpose(subvolume, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))
            subvolume /= 255.0
            subvolume = (subvolume - 0.45) / 0.225
        d = {
            "volume": subvolume,
            "label": label,
        }
        if len(subvolume.shape) < 3:
            print("shhape 2 location:", location)
        elif subvolume.shape[1] != CFG.crop_size or subvolume.shape[2] != CFG.crop_size:
            print("location()", location)
        return d


# In[ ]:


def collate_fn(batch):
    keys = batch[0].keys()

    collated_batch = {}
    for key in keys:
        if batch[0][key] is not None:
            collated_batch[key] = torch.stack(
                [torch.from_numpy(sample[key]) for sample in batch]
            )
        else:
            collated_batch[key] = None

    return collated_batch


# In[ ]:


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

        self.encoder1 = resnet34d(pretrained=False, in_chans=CFG.crop_depth)

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
        vv = [
            v[:, i : i + CFG.crop_depth]
            for i in [
                0,
                2,
                4,
            ]
        ]
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

        logit1 = F.interpolate(
            logit1, size=(H, W), mode="bilinear", align_corners=False, antialias=True
        )

        output = {
            "logit1": logit1,
            "logit2": logit2,
        }
        return output


# In[ ]:

# #### infer here !!!!<br>
# https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0

# In[ ]:


def mask_to_rle(mask):
    m = mask.reshape(-1)
    # m = np.where(mask > threshold, 1, 0).astype(np.uint8)

    s = np.array((m[:-1] == 0) & (m[1:] == 1))
    e = np.array((m[:-1] == 1) & (m[1:] == 0))

    s_index = np.where(s)[0] + 2
    e_index = np.where(e)[0] + 2
    length = e_index - s_index
    rle = " ".join(map(str, sum(zip(s_index, length), ())))
    return rle


# In[ ]:

# In[ ]:


def metric_to_text(ink, label, mask):
    text = []

    p = ink.reshape(-1)
    t = label.reshape(-1)
    pos = np.log(np.clip(p, 1e-7, 1))
    neg = np.log(np.clip(1 - p, 1e-7, 1))
    bce = -(t * pos + (1 - t) * neg).mean()
    text.append(f"bce={bce:0.5f}")

    mask_sum = mask.sum()
    # print(f'{threshold:0.1f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
    text.append("p_sum  th   prec   recall   fpr   dice   score")
    text.append("-----------------------------------------------")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = ink.reshape(-1)
        t = label.reshape(-1)
        p = (p > threshold).astype(np.float32)
        t = (t > 0.5).astype(np.float32)

        tp = p * t
        precision = tp.sum() / (p.sum() + 0.0001)
        recall = tp.sum() / t.sum()

        fp = p * (1 - t)
        fpr = fp.sum() / (1 - t).sum()

        beta = 0.5
        #  0.2*1/recall + 0.8*1/prec
        score = (
            beta * beta / (1 + beta * beta) * 1 / recall
            + 1 / (1 + beta * beta) * 1 / precision
        )
        score = 1 / score

        dice = 2 * tp.sum() / (p.sum() + t.sum())
        p_sum = p.sum() / mask_sum

        # print(fold, threshold, precision, recall, fpr,  score)
        text.append(
            f"{p_sum:0.2f}, {threshold:0.2f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}"
        )
    text = "\n".join(text)
    return text


# In[ ]:

# In[ ]:


def make_infer_mask():
    s = CFG.crop_size
    f = CFG.crop_fade
    x = np.linspace(-1, 1, s)
    y = np.linspace(-1, 1, s)
    xx, yy = np.meshgrid(x, y)
    d = 1 - np.maximum(np.abs(xx), np.abs(yy))
    d1 = np.clip(d, 0, f / s * 2)
    d1 = d1 / d1.max()
    infer_mask = d1
    return infer_mask


# In[ ]:

# In[ ]:


class Model(pl.LightningModule):
    training_step_outputs = []
    validation_step_outputs = []
    test_step_outputs = [[], []]

    def __init__(self):
        super().__init__()

        self.model = Net()

        self.loss1 = nn.BCEWithLogitsLoss(pos_weight=0.5)
        self.loss2 = nn.BCEWithLogitsLoss(pos_weight=0.5)

    def forward(self, image, stage):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        subvolumes, labels = batch["volume"], batch["label"]

        image, labels = subvolumes.float(), labels.float()
        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # print("labels", labels.max(), labels.min())

        assert labels.max() <= 1.0 and labels.min() >= 0

        segmentation_out = self.forward(image, stage)

        loss = self.loss1(segmentation_out["logit1"], labels) + self.loss2(
            segmentation_out["logit2"], labels
        )

        prob = segmentation_out["logit2"].sigmoid()

        score = fbeta_score(prob, labels, CFG.threshold)

        pred_mask = (prob > CFG.threshold).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), labels.long(), mode="binary"
        )

        m = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "score": score,
        }
        return m

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.mean(torch.Tensor([x["loss"] for x in outputs]))
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
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=CFG.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.05,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }


# In[ ]:

# In[ ]:


def train_one(d, val_d):
    net = Model()

    # get coord
    crop_size = CFG.crop_size
    stride = CFG.stride
    H, W, D = d.volume.shape

    ##pad #assume H,W >size
    px, py = W % stride, H % stride
    if (px != 0) or (py != 0):
        px = stride - px
        py = stride - py
        pad_volume = np.pad(d.volume, [(0, py), (0, px), (0, 0)], constant_values=0)
        pad_label = np.pad(d.label, [(0, py), (0, px)], constant_values=0)
    else:
        pad_volume = d.volume
        pad_label = d.label

    pH, pW, _ = pad_volume.shape
    x = np.arange(0, pW - crop_size + 1, stride)
    y = np.arange(0, pH - crop_size + 1, stride)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x, y], -1).reshape(-1, 2)
    print("H,W,pH,pW,len(xy)", H, W, pH, pW, len(xy))

    val_H, val_W, val_D = val_d.volume.shape

    ##pad #assume H,W >size
    val_px, val_py = val_W % stride, val_H % stride
    if (val_px != 0) or (val_py != 0):
        val_px = stride - val_px
        val_py = stride - val_py
        pad_val_volume = np.pad(
            val_d.volume, [(0, val_py), (0, val_px), (0, 0)], constant_values=0
        )
        pad_val_label = np.pad(
            val_d.label, [(0, val_py), (0, val_px)], constant_values=0
        )
    else:
        pad_val_volume = val_d.volume
        pad_val_label = val_d.label

    val_pH, val_pW, _ = pad_val_volume.shape
    val_x = np.arange(0, val_pW - crop_size + 1, stride)
    val_y = np.arange(0, val_pH - crop_size + 1, stride)
    val_x, val_y = np.meshgrid(val_x, val_y)
    val_xy = np.stack([val_x, val_y], -1).reshape(-1, 2)

    print(
        "val_H,val_W,val_pH,val_pW,len(val_xy)",
        val_H,
        val_W,
        val_pH,
        val_pW,
        len(val_xy),
    )

    train_ds = SubvolumeDataset(
        locations=xy,
        volume=pad_volume,
        labels=pad_label,
        buffer=crop_size // 2,
        is_train=True,
    )
    val_ds = SubvolumeDataset(
        val_xy,
        pad_val_volume,
        pad_val_label,
        crop_size // 2,
        is_train=False,
    )

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator="gpu",
        devices="0,1,2,3",
        logger=WandbLogger(name=f"2.5d-stack-unet-{datetime.datetime.now()}"),
        # strategy='ddp_find_unused_parameters_true',
    )
    trainer.fit(
        net,
        train_loader,
        val_loader,
    )


# In[ ]:

if __name__ == "__main__":
    print(cfg_to_text())

    if "train" in CFG.mode:
        data_dir = "/home/fummicc1/codes/competitions/kaggle-ink-detection/train"
        valid_id = [
            "1",
            "2b",
        ]
        train_id = ["2a", "3"]
    if "test" in CFG.mode:
        data_dir = "/home/fummicc1/codes/competitions/kaggle-ink-detection/test"
        valid_id = glob(f"{data_dir}/*")
        valid_id = sorted(valid_id)
        valid_id = [f.split("/")[-1] for f in valid_id]

        # https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/410985
        a_file = f"{data_dir}/a/mask.png"
        with open(a_file, "rb") as f:
            hash_md5 = hashlib.md5(f.read()).hexdigest()
        is_skip_test = hash_md5 == "0b0fffdc0e88be226673846a143bb3e0"
        print("is_skip_test:", is_skip_test)

    # In[ ]:

    for t, fragment_id in enumerate(train_id):
        d = read_data1(fragment_id)
        val_d = read_data1(valid_id[0])
        train_one(d, val_d)
