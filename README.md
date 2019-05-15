# Visual-Tracking-using-CNN-and-RNN

A project to create a visual tracker using CNNs and RNNs. CNN will perform the feature extraction for objects and RNN will perform the tracking of the object across frames. Since this was a prototyping project, ImageNet pretrained CNN models namely MobileNet and InceptionV3 were tried as a starting point and then the model was trained on ImageNet videos. 

### Prerequisites

1. python 3.6
2. numpy
3. keras
4. opencv-python

### Installing

For keras you can create a Conda environment as follows.

conda create -n tf_gpu tensorflow-gpu keras-gpu

You can pip install all other dependencies for the project.

## Running the model

1. Download ImageNet videos and keep folder inside 'data' folder
2. Run main.py

## Authors

* **Bharath C Renjith** -[cr-bharath](https://github.com/cr-bharath)
