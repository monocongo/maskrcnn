# maskrcnn
Instance segmentation using Mask R-CNN

### Python Environment
1. Create a new Python virtual environment:
    ```bash
    $ conda create -n maskrcnn python=3 --yes
    $ conda activate maskrcnn
    ```
2. Add libraries we'll use in our project:
    ```bash
    $ for pkg in numpy scipy h5py scikit-learn pillow imgaug imutils BeautifulSoup4 lxml tensorflow keras
    > do
    > conda install $pkg --yes
    > done
    ```
3. Get the Mask R-CNN implementation and install the dependencies:
    ```bash
    $ git clone https://github.com/matterport/Mask_RCNN.git
    $ cd Mask_RCNN/
    $ python setup.py install
    ```
4. Verify the installation:
    ```bash
    $ python
    >>> import mrcnn
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
Acquire a dataset of images and corresponding object segmentation masks. In this 
example we will utilize a single class of object taken from Google's 
[OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset. 

