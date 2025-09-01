# SpaceTime-SonoNet: Efficient Classification of Ultra-Sound Video Sequences

The goal of this project is to propose a method to automatically detect and classify standard planes (SP)
in liver ultrasound (US) videos. The operator - commonly a nurse - should detect the most informative images 
within each US video, in order to later provide them to physicians for diagnostic purposes. Such frames are known as 
standard planes and are identified by the presence of specific anatomical structures within the image. 
Given the nature of this imaging technique (being highly noisy and subject to device settings and manual skills of 
the operator) and the resulting challenge of recognizing anatomical structures (often not clearly visible even by expert 
physicians), the standard plane detection task is non-trivial and strongly operator-dependent. Nonetheless, 
one aspect that seems to aid expert users is the temporal evolution of the data within the performed motion scan 
(combined with some prior background knowledge of human anatomy). Our aim is hence to develop a deep learning pipeline 
for the automatic classification SP from single frames and sequences of frames within US videos.  

We start by following a 2D approach with a 2D CNN architecture named SonoNet [[1]](#1), which proved to achieve 
state-of-the-art results on US fetal standard plane detection task. As a first later approach concerning the usage of time information, 
instead, we propose to employ a 3D CNN model in order to exploit both spatial and temporal information on a short timescale. 
Specifically, we implemented a 3D extension of the mentioned SonoNet architecture. Extending convolutions 
to the third (temporal) domain should aid the network in solving ambiguous situations where some parts of anatomical 
structures are not clearly visible (or partly occluded) within a single frame, though could appear in nearby frames.
Based on [[2]](#2) we also implemented SonoNet(2+1)D model. It is a 3D version of SonoNet2D, but each 2D convolution layers
is replaced with a SpatioTemporal block, which consists of 2D convolution layer followed by a 1D convolution layer.
In this way, we have a model which is comparable to the SonoNet2D, in terms of trainable parameters, but with a number of non-linear operations which is double with respect to the 3D model, 
potentially leading to best results. 


------

------

## Code Organization

The project is divided into 2 folders:

- **<u>data</u>**: prepare data for 2D model training starting from the raw US dataset.
- **<u>models</u>**: define and train 2D-SonoNet architectures, as well as 3D-SonoNet and (2+1)D-SonoNet extensions.

Let's see them in more detail:
Scripts must be executed in the following order:

> - **_prepare-data2d.py_**: create **<em>data_directory/2d-split</em>** and populate its
> **<em>train</em>** and **<em>test</em>** directories with 7 folders (named from **<em>0</em>** to **<em>6</em>**). 
> Each folder contains only the PNG images passing the time sub-sampling procedure: we take both frames within a video
> sequence for which the SSIM value is lower than the average SSIM throughout the whole video.

### **<u>models</u>**
This folder contains main scripts for running experiments with different models. See the "usage" note at the beginning 
of each of them.
> - **_sononet2d-traintest.py_**: train and test the 2D SonoNet-16/32/64 model.
> - **_sononet2d-traintest_3d_comparable.py_**: trains and evaluates the 2D SonoNet-16/32/64 model using the same dataset as the 3D models for direct comparison.
> - **_sononet3d-traintest.py_**: train and test the 3D SonoNet-16/32/64 model.
> - **_temporal_test.py_**: loads a test video and visualizes the predictions of different models for temporal comparison.
> - **_2d_vs_3d_**: computes per-video accuracy on the test set for both 2D and 3D models, and calculates the average accuracy..

Such scripts use code from the following Python packages:

> **<u>utils</u>**:
> This folder contains python files with many general-purpose utility functions.
>> - **_augments.py_**: defines data augmentation methods for US images.
>> - **_datareader.py_**: defines a class for loading either the 2D or the 3D version of our dataset.
>> - **_datasplit.py_**: defines functions for splitting the dataset into training and validation sets.
>> - **_iterators.py_**: define basic training and testing loops for a single epoch.
>> - **_runner.py_**: defines train and test functions.
>> - **_visualize.py_**: defines a useful function for plotting a confusion matrix and saving it as a PNG image.
>
> **<u>sononet2d</u>**:
> This folder contains the 2D implementation of the SonoNet-16/32/64 model.
>> - **_models.py_**: defines the SonoNet2D class. The number of features in the hidden layers of the network can be 
>> set by choosing between 3 configurations (16, 32, and 64). The network may be used in "classification mode"
>> (the output is given by the adaptation layer) or for "feature extraction" (no adaptation layer is defined and the 
>> output is the set of features in the last convolutional layer): this last functionality is achieved by setting the 
>> _features_only_ parameter to True (useful to check on which image parts the network is focusing its attention). 
>> Finally, by setting the _train_classifier_only_ parameter to True it is possible to freeze learning in all 
>> convolutional layers (only the adaptation layer will be trained).
>> - **_remap-weights.py_**: convert SonoNet weights (downloaded from the [reference repository](https://github.com/rdroste/SonoNet_PyTorch))
>> to be compatible with our implementation of the model.
>
> **<u>sononet3d</u>**:
> This folder contains the 3D and (2+1)D extensions of the standard SonoNet-16/32/64 model implementation.
>> - **_models.py_**: defines the SonoNet3D and SonoNet(2+1)D classes. For the SonoNet3D all 2D convolutional and pooling layers are changed to 
>> their 3D extension. Instead, in the (2+1)D model, the 3D convolutional layers are replaced by a SpatioTemporal block, where the standard convolution
>> is decomposed into a 2D convolution followed by a 1D convovolution. As for the 2D case, the number of features in the hidden layers of the network can be 
>> set by choosing between 3 configurations (16, 32, and 64).
>> 

Results of each experiment are stored in the following folder:

> **<u>logs</u>** / **<u>weights4sononet2d</u>** / **<u>FetalDB</u>**:
>> SonoNet pre-trained weights:
>> - : pretrained weights of all SonoNet configurations (16, 32, and 64 initial 
>> features) from the FetalDB dataset. Each configuration has its own folder (SonoNet-16, SonoNet-32, and SonoNet-64) 
>> where weights are stored in "ckpt_best_loss.pth" file. Such files were obtained from those denoted as "old", which 
>> are the ones provided in [this repository](https://github.com/rdroste/SonoNet_PyTorch) (same weights but not directly 
>> compatible with our model definition).

------

------

## References

<a id="1">[1]</a>
Baumgartner C.F., Kamnitsas K., Matthew J., Fletcher T.P., Smith S., Koch L.M., Kainz B., and Rueckert D. (2017). 
SonoNet: real-time detection and localisation of fetal standard scan planes in freehand ultrasound. 
IEEE transactions on medical imaging, 36(11), pp.2204-2215.
[[link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7974824)]

<a id="2">[2]</a>
Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). 
A closer look at spatiotemporal convolutions for action recognition. 
In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition (pp. 6450-6459).
[[link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf)]
