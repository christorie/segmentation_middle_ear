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

# lib/configs is the module to define the image selection techniques

import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import UNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import strtobool # ,download_file

logger = logging.getLogger(__name__)


class Segmentation(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "ossicle chain": 1,
            "tympanic cavity": 2
        }

        # Number of input channels
        self.number_intensity_ch = 1 

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"{name}.pt"),                                        
        ]

        # Target space for image
        self.target_spacing = (0.21, 0.21, 0.3125)                                              
        
        # Setting ROI size                                                                      
        self.roi_size = (96, 96, 64)                                                            

        # Network
        self.network = UNet(

            spatial_dims=3,                                 
            kernel_size = 3,                                
            in_channels=self.number_intensity_ch,           
            out_channels=len(self.labels.keys()) + 1 ,      
            channels=[16, 32, 64, 128, 256],                
            strides=[2, 2, 2, 2],                           
            num_res_units=2,                                
            norm="instance",                                
            dropout = 0.1                                   
                                                            
        )
        

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.Segmentation(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
            config={"largest_cc": True},  # Largest Connected Component
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0]

        task: TrainTask = lib.trainers.Segmentation(
            model_dir=output_dir,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            load_path=load_path,
            publish_path=self.path[0],
            description="Train middle ear Segmentation Model",
            labels=self.labels,
            disable_meta_tracking=False,
        )
        return task
