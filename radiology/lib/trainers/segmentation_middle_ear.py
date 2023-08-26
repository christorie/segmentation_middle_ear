# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# lib/trainers is the module to define the pre and post transforms to train the network/model

import logging

import torch
from monai.optimizers import Novograd
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscreted,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics

logger = logging.getLogger(__name__)


class Segmentation(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        roi_size=(96, 96, 64),
        target_spacing=(0.21, 0.21, 0.3125),
        num_samples=4,
        description="Train mid ear Segmentation model",
        **kwargs,
    ):
        self._network = network
        self.roi_size = roi_size
        self.target_spacing = target_spacing
        self.num_samples = num_samples
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=4e-4,)
      
    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def lr_scheduler_handler(self, context: Context):
        return None

    def train_data_loader(self, context, num_workers=0, shuffle=False):
        return super().train_data_loader(context, num_workers, True)

    def train_pre_transforms(self, context: Context):
          return [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            SpatialPadd(keys=("image", "label"), spatial_size=self.roi_size),
            ScaleIntensityd(keys="image"),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=self.roi_size,
                pos=1,
                neg=1,
                num_samples=self.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            EnsureTyped(keys=("image", "label"), device=context.device),
            RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.3, 1.5)), 
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.5),  
            RandRotate90d(keys=("image", "label"), prob=0.1, max_k=3), 
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5), 
            SelectItemsd(keys=("image", "label"))
        ]

    def train_post_transforms(self, context: Context):
        return [
            EnsureTyped(keys="pred", device=context.device),
            AsDiscreted(
                keys=("pred", "label"),
                argmax=(True, False),
                to_onehot=len(self._labels) + 1,
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            Spacingd(keys=("image", "label"), pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=("image", "label")),
            SelectItemsd(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=self.roi_size, sw_batch_size=8)

    def norm_labels(self):
        new_label_nums = {}
        for idx, (key_label, _) in enumerate(self._labels.items(), start=1):
            if key_label != "background":
                new_label_nums[key_label] = idx
            if key_label == "background":
                new_label_nums["background"] = 0
        return new_label_nums

    def train_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "train_mean_dice", "train")

    def val_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "val_mean_dice", "val")

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(
                TensorBoardImageHandler(
                    log_dir=context.events_dir,
                    batch_transform=from_engine(["image", "label"]),
                    output_transform=from_engine(["pred"]),
                    interval=20,
                    epoch_level=True,
                )
            )
        return handlers
