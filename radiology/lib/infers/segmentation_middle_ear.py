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

# lib/infers is the module where researchers define the inference class
# (i.e. type of inferer, pre transforms for inference, etc)

from typing import Callable, Sequence

from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class Segmentation(BasicInferTask):
    """
    This provides Inference Engine for pre-trained segmentation (VNet) model for middle ear CT images.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(0.21, 0.21, 0.3125),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) segmentation over 3D Images",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            Spacingd(keys="image", pixdim=self.target_spacing),
            ScaleIntensityd(keys="image"),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(roi_size=self.roi_size)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        largest_cc = False if not data else data.get("largest_cc", False)
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            AsDiscreted(keys="pred", argmax=True),
        ]
        if largest_cc:
            t.append(KeepLargestConnectedComponentd(keys="pred"))
        t.append(Restored(keys="pred", ref_image="image"))
        return t
