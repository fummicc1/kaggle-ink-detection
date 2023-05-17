#!/usr/bin/env python
# coding: utf-8

# # Keras starter kit [full training set, UNet]

# ## Setup

# In[203]:


import numpy as np
import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
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
DATA_DIR = '/home/fummicc1/codes/competitions/kaggle-ink-detection'
BUFFER = 80 # Half-size of papyrus patches we'll use as model inputs
Z_LIST = list(range(0, 61, 4))  # Offset of slices in the z direction
Z_DIM = len(Z_LIST)  # Number of slices in the z direction. Max value is 64 - Z_START
SHARED_LENGTH = 4000  # Max length(width or height) to resize all papyrii

# Model config
BATCH_SIZE = 128

backbone = "se_resnext50_32x4d"
# backbone = "resnet50"

device = torch.device("cuda")
threshold = 0.25
num_workers = 8
exp = 1e-7
mask_padding = 300

num_epochs = 50
lr = 1e-3

pytorch_lightning.seed_everything(seed=42)
torch.set_float32_matmul_precision('high')


# In[204]:


# plt.imshow(Image.open(DATA_DIR + "/train/1/ir.png"), cmap="gray")
# plt.imshow(Image.open(DATA_DIR + "/train/2/ir.png"), cmap="gray")
# plt.imshow(Image.open(DATA_DIR + "/train/3/ir.png"), cmap="gray")
plt.imshow(Image.open(DATA_DIR + "/test/a/mask.png"), cmap="gray")
# plt.imshow(Image.open(DATA_DIR + "/test/b/mask.png"), cmap="gray")


# ## Load up the training data

# In[ ]:





# In[205]:


def resize(img):
    current_height, current_width = img.shape    
    aspect_ratio = current_width / current_height
    if current_height > current_width:
        new_height = SHARED_LENGTH
        new_width = int(SHARED_LENGTH * aspect_ratio)
    else:
        new_width = SHARED_LENGTH
        new_height = int(SHARED_LENGTH / aspect_ratio)        
        
    new_size = (new_width, new_height)
    # (W, H)の順で渡すが結果は(H, W)になっている
    img = cv2.resize(img, new_size)
    return img

def load_mask(split, index):
    img = cv2.imread(f"{DATA_DIR}/{split}/{index}/mask.png", 0) // 255
    img = np.pad(img, 1)[1:-1, 1:-1]
    dist = distance_transform_edt(img)
    img[dist <= mask_padding] = 0
    img = resize(img)    
    return img


def load_labels(split, index):
    img = cv2.imread(f"{DATA_DIR}/{split}/{index}/inklabels.png", 0) // 255
    img = resize(img)
    return img


mask = load_mask(split="train", index=1)
labels = load_labels(split="train", index=1)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("mask.png")
ax1.imshow(mask, cmap='gray')
ax2.set_title("inklabels.png")
ax2.imshow(labels, cmap='gray')
plt.show()


# In[177]:


mask_test_a = load_mask(split="test", index="a")
mask_test_b = load_mask(split="test", index="b")

mask_train_1 = load_mask(split="train", index=1)
labels_train_1 = load_labels(split="train", index=1)

mask_train_2 = load_mask(split="train", index=2)
labels_train_2 = load_labels(split="train", index=2)

mask_train_3 = load_mask(split="train", index=3)
labels_train_3 = load_labels(split="train", index=3)

print(f"mask_test_a: {mask_test_a.shape}")
print(f"mask_test_b: {mask_test_b.shape}")
print("-")
print(f"mask_train_1: {mask_train_1.shape}")
print(f"labels_train_1: {labels_train_1.shape}")
print("-")
print(f"mask_train_2: {mask_train_2.shape}")
print(f"labels_train_2: {labels_train_2.shape}")
print("-")
print(f"mask_train_3: {mask_train_3.shape}")
print(f"labels_train_3: {labels_train_3.shape}")


# In[178]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title("labels_train_1")
ax1.imshow(labels_train_1, cmap='gray')

ax2.set_title("labels_train_2")
ax2.imshow(labels_train_2, cmap='gray')

ax3.set_title("labels_train_3")
ax3.imshow(labels_train_3, cmap='gray')
plt.tight_layout()
plt.show()


# In[179]:


def load_volume(split, index):
    # Load the 3d x-ray scan, one slice at a time
    all = sorted(glob.glob(f"{DATA_DIR}/{split}/{index}/surface_volume/*.tif"))
    z_slices_fnames = [all[i] for i in range(len(all)) if i in Z_LIST]
    assert len(z_slices_fnames) == Z_DIM
    z_slices = []
    for z, filename in  tqdm(enumerate(z_slices_fnames)):
        img = cv2.imread(filename, -1)
        img = resize(img)
        img = (img / (2 ** 8)).astype(np.uint8)
        z_slices.append(img)
    return np.stack(z_slices, axis=-1)


# In[180]:


volume_train_1 = load_volume(split="train", index=1)
print(f"volume_train_1: {volume_train_1.shape}, {volume_train_1.dtype}")

volume_train_2 = load_volume(split="train", index=2)
print(f"volume_train_2: {volume_train_2.shape}, {volume_train_2.dtype}")

volume_train_3 = load_volume(split="train", index=3)
print(f"volume_train_3: {volume_train_3.shape}, {volume_train_3.dtype}")

# volume = np.concatenate([volume_train_1, volume_train_2, volume_train_3], axis=1)
# volume = np.concatenate([volume_train_1, volume_train_2], axis=1)
# print(f"total volume: {volume.shape}")


# In[181]:


# # labels = np.concatenate([labels_train_1, labels_train_2, labels_train_3], axis=1)
# labels = np.concatenate([labels_train_1, labels_train_2], axis=1)
# print(f"labels: {labels.shape}, {labels.dtype}")
# # mask = np.concatenate([mask_train_1, mask_train_2, mask_train_3], axis=1)
# mask = np.concatenate([mask_train_1, mask_train_2], axis=1)
# print(f"mask: {mask.shape}, {mask.dtype}")


# In[182]:


# # Free up memory
# del labels_train_1
# del labels_train_2
# del labels_train_3
# del mask_train_1
# del mask_train_2
# del mask_train_3


# ## Visualize the training data
# 
# In this case, not very informative. But remember to always visualize what you're training on, as a sanity check!

# In[183]:


# fig, axes = plt.subplots(1, 2, figsize=(15, 3))
# for z, ax in enumerate(axes):
#     ax.imshow(volume[:, :, z], cmap='gray')
#     ax.set_xticks([]); ax.set_yticks([])
# fig.tight_layout()
# plt.show()


# ## Create a dataset in the input volume
# 

# In[184]:


def is_in_masked_zone(location, mask):
    return mask[location[0], location[1]] > 0


# In[185]:


def generate_locations_ds(volume, mask):
    is_in_mask_train = lambda x: is_in_masked_zone(x, mask)

    # Create a list to store train locations
    locations = []

    # Generate train locations
    volume_height, volume_width = volume.shape[:-1]

    for y in range(BUFFER, volume_height - BUFFER, BUFFER // 2):
        for x in range(BUFFER, volume_width - BUFFER, BUFFER // 2):
            if is_in_mask_train((y, x)):
                locations.append((y, x))

    # Convert the list of train locations to a PyTorch tensor
    locations_ds = np.stack(locations, axis=0)
    return locations_ds


# ## Visualize some training patches
# 
# Sanity check visually that our patches are where they should be.

# In[186]:


possible_min_input = possible_max_input = all_median = all_MAD = None


# In[187]:


from scipy.stats import median_abs_deviation


def calculate_all_MAD(volume):
    global all_MAD
    all_MAD = median_abs_deviation(volume, axis=[0, 1])
    print("all_MAD", all_MAD)


# In[188]:


def calculate_all_median(volume):
    global all_median
    all_median = np.median(volume, axis=[0, 1])
    print("all_median", all_median)


# In[189]:


def calculate_possibles(all_median, all_MAD):
    global possible_max_input, possible_min_input
    # possible_max_input = ((2 ** 8 - 1) - all_median.min()) / all_MAD.min()
    # possible_min_input = ((0) - all_median.min()) / all_MAD.min()
    possible_max_input = (2 ** 8 - 1) / all_median.min()
    possible_min_input = 0 / (all_median.min() + exp)


# In[190]:


print("all_median", all_median)
"all_median", all_median


# In[191]:


print("all_MAD", all_MAD)
"all_MAD", all_MAD


# In[192]:


printed = True

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
    
    # subvolume = (subvolume - median) / MAD
    # subvolume = subvolume / median
    
    if not printed:
        print("subvolume after taking care of median and MAD", subvolume)
        printed = True
    
    return subvolume


# ## SubvolumeDataset

# In[193]:


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

from albumentations.core.transforms_interface import ImageOnlyTransform

class MedianImageValue(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(MedianImageValue, self).__init__(always_apply, p)

    def apply(self, img, **params):
        median = np.full_like(img, all_median).astype(np.float32)
        return img / median


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
            label = self.labels[y - self.buffer:y + self.buffer, x - self.buffer:x + self.buffer]
            # print("label", label)
            # n_category = 2
            # label = np.eye(n_category)[label]
            label = np.stack([label], axis=-1)
            # label = label.numpy()
            # print("label.shape", label.shape
        
        if self.is_train and label is not None:            
            
            # print("label", label.dtype)
            # print("subvolume in data/et (before aug)", subvolume, file=open("before-aug.log", "w")) 
            size = int(BUFFER * 2)
            performed = A.Compose([
                # A.ToFloat(max_value=possible_max_input - possible_min_input),                
                A.HorizontalFlip(p=0.5), # 水平方向に反転
                A.VerticalFlip(p=0.5), # 水平方向に反転
                A.RandomRotate90(p=0.5),
                # A.RandomBrightnessContrast(p=0.4),
                A.ShiftScaleRotate(p=0.5, border_mode=0), # シフト、スケーリング、回転
                # A.PadIfNeeded(min_height=size, min_width=size, always_apply=True, border_mode=0), # 必要に応じてパディングを追加
                A.RandomCrop(height=int(size // 1.1), width=int(size // 1.1), p=0.5), # ランダムにクロップ, Moduleの中で計算する際に次元がバッチ内で揃っている必要があるので最後にサイズは揃える
                # A.Perspective(p=0.5), # パースペクティブ変換                
                # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.CoarseDropout(max_holes=1, max_width=int(size * 0.1), max_height=int(size * 0.1), 
                                mask_fill_value=0, p=0.2),
                A.GaussianBlur(p=0.25, blur_limit=(3, 5)),
                A.MedianBlur(p=0.25, blur_limit=5),
                A.Resize(BUFFER * 2, BUFFER * 2, always_apply=True),                
                # A.Normalize(
                #     mean= [0] * Z_DIM,
                #     std= [1] * Z_DIM
                # ),
                # A.FromFloat(max_value=possible_max_input - possible_min_input),
                MedianImageValue(always_apply=True),
                ToTensorV2(transpose_mask=True),                
            ])(image=subvolume, mask=label)            
            subvolume = performed["image"]
            label = performed["mask"]
            # print("subvolume in dataset (after aug)", subvolume, file=open("after-aug.log", "w"))
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
                performed = A.Compose([            
                    # A.ToFloat(max_value=possible_max_input - possible_min_input),
                    # A.Normalize(
                    #     mean= [0] * Z_DIM,
                    #     std= [1] * Z_DIM
                    # ),
                    # A.FromFloat(max_value=possible_max_input - possible_min_input),
                    MedianImageValue(always_apply=True),
                    ToTensorV2(transpose_mask=True),
                ])(image=subvolume)
            else:
                performed = A.Compose([            
                    # A.ToFloat(max_value=possible_max_input - possible_min_input),
                    # A.Normalize(
                    #     mean= [0] * Z_DIM,
                    #     std= [1] * Z_DIM
                    # ),
                    # A.FromFloat(max_value=possible_max_input - possible_min_input),
                    MedianImageValue(always_apply=True),
                    ToTensorV2(transpose_mask=True),
                ])(image=subvolume, mask=label)
                label = performed["mask"]
            subvolume = performed["image"]
            # subvolume = torch.from_numpy(subvolume.transpose(2, 0, 1).astype(np.float32))
            # if label is not None:
                # label = torch.from_numpy(label.transpose(2, 0, 1).astype(np.uint8)) 
        if self.return_location:
            return subvolume, location
        return subvolume, label        


# ## Visualize validation dataset patches
# 
# Note that they are partially overlapping, since the stride is half the patch size.

# In[194]:


def visualize_dataset_patches(locations_ds, labels, mode: str, fold = 0):
    fig, ax = plt.subplots()
    ax.imshow(labels)

    for y, x in locations_ds:
        patch = patches.Rectangle([x - BUFFER, y - BUFFER], 2 * BUFFER, 2 * BUFFER, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(patch)
    plt.savefig(f"fold-{fold}-{mode}.png")
    plt.show()    


# ## Compute a trivial baseline
# 
# This is the highest validation score you can reach without looking at the inputs.
# The model can be considered to have statistical power only if it can beat this baseline.

# In[195]:


def trivial_baseline(dataset):
    total = 0
    matches = 0.
    for _, batch_label in tqdm(dataset):
        batch_label = torch.tensor(batch_label)
        matches += torch.sum(batch_label.float())
        total += torch.numel(batch_label)
    return 1. - matches / total

# score = trivial_baseline(val_ds).item()
# print(f"Best validation score achievable trivially: {score * 100:.2f}% accuracy")


# ## Model

# In[198]:


class Model(pl.LightningModule):
    
    training_step_outputs = []
    validation_step_outputs = []
    test_step_outputs = [[], []]

    def __init__(self, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name, 
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
        self.segmentation_loss_fn = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.classification_loss_fn = smp.losses.SoftCrossEntropyLoss()

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        subvolumes, labels = batch
        
        image, mask = subvolumes.float(), labels.float()
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
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        segmentation_out = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.segmentation_loss_fn(segmentation_out, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = segmentation_out.sigmoid()
        if stage == "valid":
            # print("prob_mask", prob_mask.max().item(), prob_mask.min().item())
            pass
        pred_mask = (prob_mask > threshold).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

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
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
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
        predictions = predictions.cpu().numpy()  # move predictions to cpu and convert to numpy
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
        new_predictions_map_counts = np.zeros_like(predictions_map_counts[:, :, 0])[:, :, np.newaxis]
        
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
            new_predictions_map_counts[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER, :] += 1
        new_predictions_map /= (new_predictions_map_counts + exp)
        predictions_map = np.concatenate([predictions_map, new_predictions_map], axis=-1)
        print("new_predictions_map", new_predictions_map.shape)
        print("predictions_map", predictions_map.shape)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_dataset_iou"}


# In[199]:


k_folds = 3
kfold = KFold(
    n_splits=k_folds,
    shuffle=True
)
data_list = [
    (volume_train_1, labels_train_1, mask_train_1),
    (volume_train_2, labels_train_2, mask_train_2),
    (volume_train_3, labels_train_3, mask_train_3),
]

all_volume = np.concatenate([volume_train_1, volume_train_2, volume_train_3], axis=1)

calculate_all_MAD(all_volume)
calculate_all_median(all_volume)
calculate_possibles(all_median, all_MAD)

for fold, (train_data, val_data) in enumerate(kfold.split(data_list)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    one = data_list[train_data[0]]
    two = data_list[train_data[1]]
    train_volume = np.concatenate([one[0], two[0]], axis=1)
    train_label = np.concatenate([one[1], two[1]], axis=1)
    train_mask = np.concatenate([one[2], two[2]], axis=1)
    val_volume, val_label, val_mask = data_list[val_data[0]]    
    
    train_locations_ds = generate_locations_ds(train_volume, train_mask)
    val_location_ds = generate_locations_ds(val_volume, val_mask)   
    
    visualize_dataset_patches(train_locations_ds, train_label, "train", fold)
    visualize_dataset_patches(val_location_ds, val_label, "val", fold)
    
    # Init the neural network
    model = Model(
        encoder_name=backbone,
        in_channels=Z_DIM,
        out_classes=1,
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        devices="auto",
        accelerator="auto",
        log_every_n_steps=BATCH_SIZE // 4,
    )
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_ds = SubvolumeDataset(
        train_locations_ds,
        train_volume,
        train_label,
        BUFFER,
        is_train=True
    )
    val_ds = SubvolumeDataset(
        val_location_ds,
        val_volume,
        val_label,
        BUFFER,
        is_train=False,
    )
    
    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=False,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)


# In[ ]:


print("all_median", all_median)
all_median


# In[ ]:


print("all_MAD", all_MAD)
all_MAD

