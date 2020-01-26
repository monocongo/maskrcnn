# maskrcnn
This project provides the ability to train and utilize the 
[Mask R-CNN](https://arxiv.org/abs/1703.06870) algorithm for instance segmentation, 
using an implementation provided by [Matterport](https://github.com/matterport/Mask_RCNN).

### Prerequisites
We assume a Linux development environment running on Ubuntu 18.04.

1. Install [CUDA](https://developer.nvidia.com/cuda-toolkit):
    ```bash
    $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    $ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    $ sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    $ sudo apt-get update
    $ sudo apt-get -y install cuda
    ```
2. Install [cuDNN](https://developer.nvidia.com/rdp/cudnn-download):
    1. Login to the [NVIDIA Developer Network](https://developer.nvidia.com)
    2. Download the [cuDNN Runtime Library for Ubuntu18.04](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/Ubuntu18_04-x64/libcudnn7_7.6.5.32-1%2Bcuda10.2_amd64.deb)
    ```
    $ sudo apt install ./libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
    ```
3. Install [OpenCV](https://opencv.org/):
    ```
    $ sudo apt-get install libopencv-dev python3-opencv
    ```   
### Python Environment
1. Create a new Python virtual environment:
    ```bash
    $ conda config --add channels conda-forge
    $ conda create -n mrcnn python=3 --yes
    $ conda activate mrcnn
    ```
2. Get the Mask R-CNN implementation and install the dependencies:
    ```bash
    $ git clone https://github.com/matterport/Mask_RCNN.git
    $ cd Mask_RCNN/
    $ python setup.py install
    ```
   NOTE: it may be necessary to add the following into the `mrcnn/model.py` file 
   before installing the package:
   ```python
   if K.backend()=='tensorflow':
      keras.backend.set_image_data_format("channels_last")
   ```
3. Install additional libraries we'll use in our project (assumes that `conda-forge` 
is the primary channel):
    ```bash
    $ for pkg in opencv imutils imgaug tensorflow-gpu=1.13 keras=2.2.5
    > do
    > conda install $pkg --yes
    > done
    ```
4. Verify the installation:
    ```bash
    $ python
    >>> import mrcnn
    >>> import cv2
    >>> import imutils
    >>>
    ```
5. Get the Mask R-CNN model file that has been trained on COCO dataset, which will 
be used as the basis of our custom trained model: 
    ```bash
    $ cd
    $ mkdir mrcnn
    $ cd mrcnn
    $ export MRCNN=`pwd`
    $ wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
    ```

### Training Dataset
Acquire a dataset of images and corresponding object segmentation masks. This project 
currently supports two dataset scenarios: 1) a dataset with a directory of image 
files and a corresponding directory of mask image files matching to each image 
file, and 2) a dataset with a directory of image files and an annotations JSON file 
created by the [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html) 
tool. In order to validate a dataset with masks defined via the VIA tool we can 
use the Jupyter notebook `notebook/inspect_dataset_via.ipynb`.


A good dataset to use that includes image mask files is the 
[ISIC 2018 Skin Lesion Analysis Dataset](https://challenge2018.isic-archive.com/), 
and is appropriate for use with the first scenario mentioned above.

Support is planned for datasets pulled from Google's  
[OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset, 
which includes images with segmentation regions annotated in CSV format.

### Training
A training script is provided for training the model using the two supported dataset 
scenarios described above.
###### Usage with masks:
```bash
$ export KERAS_BACKEND=tensorflow
$ python maskrcnn/train.py --images /data/lesions/images --masks /data/lesions/masks \
    --masks_suffix _segmentation.png --pretrained ${MRCNN}/mask_rcnn_coco.h5 \
    --output ${MRCNN}/output --classes /data/lesions/class_labels.txt \
    --epochs 50 --train_split 0.8
```
The above assumes a dataset with image files in JPG format with `*.jpg` extensions 
in the directory `/data/lesions/images` and corresponding mask files in PNG format 
with `*.png` extensions in the directory `/data/lesions/masks`. The mask files should 
share the file ID (file name minus the ".jpg" extension) of the corresponding image 
file, with the mask file name composed of the file ID plus the `masks_suffix` argument. 
For example if the image file is "abc.jpg" and the `masks_suffix` argument is 
"_segmentation.png" then the mask file is expected to be named "abc_segmentation.png". 
We also expect to have a class labels file with one label per line in order to tell 
the model which classes are being mapped to class IDs, with the first class label 
line corresponding to class ID 1, the second to class ID 2, etc.

###### Usage with a VIA annotations JSON file:
```bash
$ export KERAS_BACKEND=tensorflow
$ python maskrcnn/train.py --images /data/lesions/images \
    --viajson /data/lesions/via_annotations.json \
    --pretrained ${MRCNN}/mask_rcnn_coco.h5 \
    --output ${MRCNN}/output --classes /data/lesions/class_labels.txt \
    --epochs 50 --train_split 0.8
```

### Credit
The basis of this project is the original code found in 
[Deep Learning for Computer Vision by Dr. Adrian Rosebrock](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/). 
