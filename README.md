# maskrcnn
Instance segmentation using Mask R-CNN

### Python environment
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
    $ git clone https://github.com/matterport/Mask_RCNN
    $ cd Mask_RCNN/
    $ git checkout 1aca439c37849dcd085167c4e69d3abcd9d368d7
    $ pip install -r requirements.txt
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
