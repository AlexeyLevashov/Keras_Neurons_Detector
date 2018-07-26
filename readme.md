# Neurons Detector

## Overview
This is a Keras detector of 
neurons on medical images of brain slices of rats. 
This implementation achieve approximately 80% F1 score on 
dataset of 61 annotated images. Solution is tested and
worked well on Linux Ubuntu 14.0 and Windows 10 with
Python3
![image](https://github.com/AlexeyLevashov/Keras_Neurons_Detector/blob/master/pics/img.png?raw=true)
![image](https://github.com/AlexeyLevashov/Keras_Neurons_Detector/blob/master/pics/mask.png?raw=true)
![image](https://github.com/AlexeyLevashov/Keras_Neurons_Detector/blob/master/pics/annotations.png?raw=true)


## Installation
First you should install tensorflow (CPU or GPU version 
if you have Video Card with Cuda Compute Capability 3.0)
https://www.tensorflow.org/install/

Then install all dependencies by <br>
`pip install requirements.txt`


## Dataset
The dataset included to repository: <br>
data/labeled_images <br>
Each image is annotated by LabelImg: <br>
https://github.com/tzutalin/labelImg <br>
Just download binaries for any platform from repo for 
looking and editing the neurons annotations.


## Training 
For training just run: <br>
`python train.py`

Training takes ~2 hours on Tesla P40. <br>
After training the model weights with quality report will
be created in data/trained_weights/vgg: <br>
data/trained_weights/vgg/best_weights.hdf5
data/trained_weights/vgg/best_weights_quality_report.txt


## Testing
For just testing training net on dataset: <br>
`python estimate_quality.py`


## Detection
For making detection on new images run: <br>
`python detect.py <images_mask>` <br>
e.g. <br>
`python detect.py data/validation_images/*.jpg`

This script creates for each image the xml file
with annotated rects in LabelImg format
