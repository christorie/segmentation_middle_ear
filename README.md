# segmentation_middle_ear

This application is based on the radiology [MONAI sample app](https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps/radiology) and was modified for the segmentation of middle ear ossicles and the tympanic cavity based on CT Scans. It includes modifications for the training of a V-Net shaped CNN and the inference on cropped CT volumes of the middle ear. Also the pre-trained segmentation_middle_ear model is provided which was used for the presented segmentations and the validations in the paper linked below.

<img src="https://github.com/christorie/segmentation_middle_ear/blob/main/media/ossicles.gif" width="400" height="300" style="float: left;"/> <img src="https://github.com/christorie/segmentation_middle_ear/blob/main/media/tympanicCavity.gif" width="400" height="300" align="right"/>


# Installation
You can clone this repository to your local drive (e.g. /yourdir/) with: 
```
cd /yourdir
git clone https://github.com/christorie/segmentation_middle_ear.git
cd segmentation_middle_ear
```
## Running in a docker image
You can then run the segmentation_middle_ear model via a docker container, to ensure that all the required packages are installed. Therefore the official [projectmonai/monailabel:0.6.0](https://hub.docker.com/layers/projectmonai/monailabel/0.6.0/images/sha256-7066b51948bfd73cd10f4857e1d614522481f8bcea48598f94c14480f5335b1f?context=explore) image will be used.

Make sure docker is installed on your system (Linux/WSL2). After cloning the repository and navigating to the target folder (e.g. yourdir/segmentation_middle_ear), you can run the start_docker_ml060.sh shell script with:
```
youruser@workstation: /yourdir/segmentation_middle_ear# . start_docker_ml060.sh
```
This will start a docker container of the image linked above and map the repository to the folder ```/segmentation_middle_ear/``` within the container. In addition the port 8000 is exposed for the usage of the monailabel server. 

Within the container the monailabel server can be started with: 
```
cd /segmentation_middle_ear/
root@b6cbae0d7beb:/segmentation_middle_ear# monailabel start_server --app radiology --studies sample_data --conf models segmentation_middle_ear
```
## Running locally
You can also run the segmentation_middle_ear model on your local system. Therefore follow the [MONAILabel installation](https://docs.monai.io/projects/label/en/latest/installation.html#install-monai-label) instructions and make sure that at least the following is installed:
````
MONAILabel version: 0.6.0
Python version: 3.9.5
Numpy version: 1.23.5
PyTorch version: 1.13.1+cu116
````
After cloning the repository you can start the monailabel server locally:
```
cd yourdir/segmentation_middle_ear/
youruser@workstation: /yourdir/segmentation_middle_ear# monailabel start_server --app radiology --studies sample_data --conf models segmentation_middle_ear
```
# Getting started
The segmentation_middle_ear model can be used in the [3D Slicer](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#installing-3d-slicer) with the [MONAILabel Slicer Module](https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer#installing-monai-label-plugin). Make sure both is installed properly on your system with at least the versions:
```
3DSlicer: Slicer-5.0.0
MONAILabel Extension: 77d35f5 (2023-01-04)
```
After starting the monailabel server (container/locally) it can be reached on localhost:8000 with the 3D Slicer. You can then load the sample data and run the segmentation_middle_ear model. If everything works fine the visualization should look similar to this:

<img src="https://github.com/christorie/segmentation_middle_ear/blob/main/media/slicerGUI.PNG"/>


You can then create closed surface representations of the segmentations with the 3D Slicer and apply the model to new data. For details about the input data format please refer to the paper linked below.

Training, inference and architecture of the CNN can be modified in the ```segmentation_middle_ear.py``` files located under the ```radiology/lib/``` directory, e.g. ```/segmentation_middle_ear/radiology/lib/trainers/segmentation_middle_ear.py```.

# Paper
tba
